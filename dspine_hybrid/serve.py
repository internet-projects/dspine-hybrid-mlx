"""OpenAI-compatible local server for one Qwen3.8-27B build decoded with DSpine.

    dspine-hybrid-serve --fast --target mlx-community/Qwen3.8-27B-4bit \\
        --draft netprojects/Qwen3.8-27B-DSpine-b16-MLX --port 8791

The HTTP side is mlx-lm's own server (``mlx_lm.server``, MIT): its request parsing, chat
templates, tool-call and reasoning parsing, streaming and usage reporting are used unchanged.
Only the generation thread is replaced: instead of mlx-lm's decode loop it runs this kit's
generator, so every answer is drafted by DSpine and checked by the target. Endpoints are the
ones mlx-lm serves (``/v1/chat/completions``, ``/v1/completions``, ``/v1/models``, ``/health``).

Up to ``--parallel N`` requests (default 8) are decoded together: every round checks all of them in
one target pass (``batch_gen``), with ``--tree`` (or ``--fast``) on a draft tree per greedy request.
A request that is alone runs on the one-request generator, which is faster for a single user, and
joins the rounds between two of its own rounds when another request arrives (``--no-alone`` turns
this off). With ``--parallel 1`` requests are decoded one at a time and others wait in a queue.
Prompts of finished requests are kept, so a follow-up turn only reads what is new. Not supported:
images, logprobs, logit bias and repetition/presence/frequency penalties (ignored with a
warning), and a per-request model name (the loaded target always answers).
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time

import mlx.core as mx


class _Stop(Exception):
    """Raised inside the token callback to end a generation early."""


class _Emitter:
    """Turns the tokens of one request into mlx-lm ``Response`` items on its queue."""

    def __init__(self, S, rqueue, ctx, tok, stop_sequences, stop_ids, max_tokens):
        self.S, self.rqueue, self.ctx = S, rqueue, ctx
        self.detok = tok.detokenizer
        self.matcher = stop_sequences.matcher()
        self.stop_ids = set(stop_ids)
        self.max_tokens = int(max_tokens)
        self.n = 0
        self.done = False

    def _put(self, token, reason):
        self.rqueue.put(self.S.Response(self.detok.last_segment, token, 0.0, reason, ()))

    def feed(self, new) -> bool:
        """Send ``new`` tokens on. True when the generation must be cut here (a stop word, or the
        client went away); the generator ends by itself on stop ids and on the token limit."""
        for t in new:
            if self.done:
                return False
            t = int(t)
            self.n += 1
            if t in self.stop_ids:
                self.detok.finalize()
                self._put(t, "stop")
                self.done = True
                return False
            self.detok.add_token(t)
            if self.matcher.advance(t):
                self.detok.finalize()
                self._put(t, "stop")
                self.done = True
                return True
            if self.n >= self.max_tokens:
                self.detok.finalize()
                self._put(t, "length")
                self.done = True
                return False
            self._put(t, None)
        return bool(self.ctx._should_stop)

    def close(self, fallback_token=0):
        """After the generator ended: the final marker if none was sent, then the end of stream."""
        if not self.done:
            self.detok.finalize()
            self._put(int(fallback_token), "length" if self.n >= self.max_tokens else "stop")
            self.done = True
        self.rqueue.put(None)


def _cost_file(a, rows: int):
    """Where this setup's measured target costs are kept between starts (None: not kept)."""
    import hashlib
    import os
    import platform

    if getattr(a, "remeasure", None) is None:        # embedded use and tests: measure, keep nothing
        return None
    root = os.environ.get("DSPINE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "dspine-hybrid")
    key = json.dumps([str(a.target), str(a.draft), "dspine", a.draft_bits, a.block_size, rows, a.parallel,
                      bool(a.fast), bool(a.tree), bool(a.fast_verify), bool(a.adaptive), a.confidence,
                      mx.__version__, platform.machine(), platform.mac_ver()[0], 2])
    return os.path.join(root, "cost-" + hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")


def _saved_costs(path, days: float = 14.0):
    """The cost tables of an earlier start: dict(single=[...] or None, batch={B: [...]}), or None."""
    import os

    try:
        if path is None or time.time() - os.path.getmtime(path) > days * 86400:
            return None
        with open(path) as f:
            d = json.load(f)
        return dict(single=d.get("single"), batch={int(k): v for k, v in d["batch"].items()})
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_costs(path, single, batch) -> None:
    import os

    if path is None or batch is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(dict(single=single, batch={str(k): v for k, v in batch.items()}), f)
        os.replace(path + ".tmp", path)
    except OSError as e:
        logging.warning("could not save the measured costs to %s: %s", path, e)


def _args_namespace(a) -> argparse.Namespace:
    """The fields mlx-lm's server reads from its own command line."""
    template_args = {} if a.thinking else {"enable_thinking": False}
    return argparse.Namespace(
        model=a.target, adapter_path=None, draft_model=None, num_draft_tokens=0,
        trust_remote_code=False, chat_template="", use_default_chat_template=False,
        temp=a.temperature, top_p=a.top_p, top_k=a.top_k, min_p=a.min_p, max_tokens=a.max_tokens,
        chat_template_args=template_args, allowed_origins=a.allowed_origins,
        decode_concurrency=1, prompt_concurrency=1, prefill_step_size=2048,
        prompt_cache_size=1, prompt_cache_bytes=None, kv_bits=None, kv_group_size=64,
        quantized_kv_start=0, pipeline=False, log_level=a.log_level,
    )


def build(a):
    """Return (response_generator_class, model_provider) wired to this kit. Imports mlx-lm's server lazily."""
    from mlx_lm import server as S

    from .generate import _load_tokenizer_eos
    from .run import DEFAULT_DRAFT_BITS, apply_memory_rules, load

    class Provider(S.ModelProvider):
        """Holds the one loaded target; the kit loads it inside the generation thread."""

        def __init__(self, cli_args):
            super().__init__(cli_args)
            self.gen = None
            self.loaded = threading.Event()        # set when the generation thread has loaded the model
            self.load_error = None
            self.proposer = None
            self.target = None
            self.stop_ids = ()
            self.cost = None

        def load_kit(self):
            apply_memory_rules(a.memory_limit_gb)
            bits = DEFAULT_DRAFT_BITS if a.draft_bits is None else (None if a.draft_bits >= 16 else a.draft_bits)
            fast = a.fast
            self.cost = None
            if a.parallel > 1:
                # batched rounds: one check size per round; finished requests' caches are kept
                self.batch_tree = bool(a.tree or fast)
                self.batch_copy = bool(a.copy)
                from .batch_gen import measure_batch_cost

                fitted = None
                if a.calibration:
                    from . import calibration as cal

                    fitted = cal.load(a.calibration, "greedy")
                kernels = a.fast_verify or fast
                from .batch_gen import DSpineProposer

                # the one-request generator with everything v0.3 had: a request that is alone runs
                # on it (eb-16: after a 32K prompt 43.6 tok/s against 33 to 37 in the rounds)
                path = _cost_file(a, 16 if kernels else 0)
                saved = None if getattr(a, "remeasure", False) else _saved_costs(path)
                self.gen, self.target, _, _ = load(
                    a.target, a.draft, draft_bits=bits, block_size=a.block_size or (16 if kernels else None),
                    adaptive=a.adaptive or fast, confidence="calibrated" if fast else a.confidence,
                    copy=a.copy, tree=a.tree or fast, fast_verify=kernels,
                    a8_prefill=a.a8_prefill or fast, prompt_cache=True,
                    cost=saved["single"] if saved else None)
                self.proposer = DSpineProposer(self.gen, calibration=fitted)
                rows = int(self.gen.block_size)
                if a.adaptive or fast:
                    if saved and len(saved["batch"]) >= a.parallel and len(saved["batch"][1]) == rows + 1:
                        self.cost = saved["batch"]
                        logging.info("target costs from %s (--remeasure measures them again)", path)
                    else:
                        self.cost = measure_batch_cost(
                            self.target, list(range(1000, 1256)), max_batch=a.parallel, max_T=rows,
                            tree=self.batch_tree, sizes=(1, 2, 4, 8, 16))
                        _save_costs(path, getattr(self.gen, "cost", None), self.cost)
                if self.cost is None and (a.adaptive or fast):
                    self.cost = measure_batch_cost(self.target, list(range(1000, 1256)), max_batch=a.parallel,
                                                   max_T=rows, tree=self.batch_tree, sizes=(1, 2, 4, 8, 16))
            else:
                self._load_single(bits, fast)
            self.model = self.target.model
            self.tokenizer = self.target.tokenizer
            mx.eval(self.model.parameters())           # read the weights now, not inside the first request
            drafter = getattr(self.proposer, "draft", None) or getattr(self.gen, "draft", None)
            if drafter is not None:
                mx.eval(drafter.parameters())
            self.model_key = (a.target, None, a.draft)
            self.is_batchable = False
            self.stop_ids = tuple(_load_tokenizer_eos(self.tokenizer))
            if a.fast_verify or fast:
                logging.info("fast verify kernels: %s", "on" if self.target.fast_verify else "not available")
            if a.a8_prefill or fast:
                logging.info("int8 prefill: %s", "on" if self.target.a8_prefill is not None else "not available")

        def _load_single(self, bits, fast):
            self.gen, self.target, _, _ = load(
                a.target, a.draft, draft_bits=bits, block_size=a.block_size,
                adaptive=a.adaptive or fast, confidence="calibrated" if fast else a.confidence,
                copy=a.copy, tree=a.tree or fast, fast_verify=a.fast_verify or fast,
                a8_prefill=a.a8_prefill or fast, prompt_cache=True)

        # mlx-lm calls these to switch models per request; this server has exactly one
        def load_default(self):
            if self.gen is None:
                self.load_kit()

        def load(self, model_path=None, adapter_path=None, draft_model_path=None):
            self.load_default()
            return self.model, self.tokenizer

        def reset(self):
            self.gen = self.target = self.model = self.tokenizer = None

    class Generator(S.ResponseGenerator):
        """mlx-lm's response generator with the decode loop replaced by the kit's."""

        def _generate(self):
            mp = self.model_provider
            try:
                mp.load_default()
            except BaseException as e:      # main() reports it instead of printing Ready
                mp.load_error = e
                raise
            finally:
                mp.loaded.set()
            logging.info("dspine-hybrid-serve: model loaded, ready")
            if a.parallel > 1:
                self._serve_together(a.parallel)
            while not self._stop:
                request = self._next_request(timeout=0.1)
                if request is None:
                    continue
                self._serve_dspine(request)
            mp.reset()

        def _open(self, request):
            """Tokenize a request and announce it to its handler: (rqueue, args, prompt, ctx, emitter)."""
            rqueue, request, args = request
            mp = self.model_provider
            tok = mp.tokenizer
            prompt, segs, kinds, initial_state, vision = self._tokenize(tok, request, args)
            # where the system prompt (with the tool list) ends: the target keeps a restore point there
            self._system_len = len(segs[0]) if kinds and kinds[0] == "system" and len(segs) > 1 else None
            if vision is not None:
                raise ValueError("images are not supported by dspine-hybrid-serve")
            stop_sequences, text_sm = self._make_state_machine(mp.model_key, tok, args.stop_words)
            ctx = S.GenerationContext(
                has_thinking=tok.has_thinking, has_tool_calling=tok.has_tool_calling,
                tool_parser=tok.tool_parser, text_sm=text_sm, initial_state=initial_state, prompt=prompt)
            ctx.prompt_cache_count = 0
            rqueue.put(ctx)
            lg = args.logits
            if (lg.logit_bias or (lg.repetition_penalty or 0) not in (0, 1.0)
                    or lg.presence_penalty or lg.frequency_penalty or args.logprobs):
                logging.warning("logit bias, penalties and logprobs are ignored by dspine-hybrid-serve")
            return prompt, ctx, _Emitter(S, rqueue, ctx, tok, stop_sequences, mp.stop_ids, args.max_tokens)

        def _serve_together(self, parallel):
            """Decode up to ``parallel`` requests together: one target check per round for all."""
            from .batch_gen import BatchGenerator, Conversation

            mp = self.model_provider
            if getattr(mp, "proposer", None) is None:
                from .batch_gen import DSpineProposer

                mp.proposer = DSpineProposer(mp.gen)
            bg = BatchGenerator(mp.target, mp.proposer, max_batch=parallel, cost=mp.cost,
                                keep_sessions=parallel, keep_tokens=a.cache_tokens, tree=getattr(mp, "batch_tree", False), tree_from=getattr(mp, "tree_from", 2),
                                copy=getattr(mp, "batch_copy", False))
            live = {}                                   # id(conversation) -> (conversation, emitter, rqueue, t0)

            def end(conv):
                em, rqueue, t0, was = live.pop(id(conv))[1:]
                ph = {k: bg.phase_seconds[k] - was[0].get(k, 0.0) for k in bg.phase_seconds}
                n_rounds = max(bg.rounds - was[1], 1)
                rows = sum(m * n for (_, m), n in bg.row_hist.items()) - was[2]
                em.close(mp.stop_ids[0] if mp.stop_ids else 0)
                st = conv.stats
                logging.info("request: prompt %d tokens (%d from cache), %d new tokens, %.1f s (%d decoding together); "
                             "prefill %.2f s, %d rounds, %.2f tokens per round, %.1f ms per round, %s, memory %.1f GB "
                             "(peak %.1f), %d kept prompts with %d tokens; while it ran: %d rounds of %.1f rows, "
                             "ms per round draft %.1f pick %.1f check %.1f land %.1f",
                             len(conv.prompt), conv.cached_tokens, em.n, time.perf_counter() - t0, len(bg.active),
                             st.prefill_seconds, st.rounds, st.committed_tokens / max(st.rounds, 1),
                             1000 * st.decode_seconds / max(st.rounds, 1),
                             "sampled" if conv.sampling is not None else "greedy",
                             mx.get_active_memory() / 2**30, mx.get_peak_memory() / 2**30,
                             len(bg.sessions), sum(len(x["seq"] or ()) for x in bg.sessions),
                             n_rounds, rows / n_rounds, *(1000 * ph[k] / n_rounds for k in ("draft", "pick", "check", "land")))

            def enqueue(request):
                rqueue = request[0]
                try:
                    prompt, ctx, em = self._open(request)
                    s = request[2].sampling
                    if request[2].seed is not None:
                        logging.warning("seed is ignored with --parallel")
                    conv = Conversation(
                        list(prompt), max_tokens=request[2].max_tokens, temperature=float(s.temperature or 0.0),
                        top_k=int(s.top_k or 0), top_p=float(s.top_p if s.top_p is not None else 1.0),
                        min_p=float(s.min_p or 0.0), stop_ids=mp.stop_ids, mark_hint=self._system_len)

                    def on_tokens(new, c=conv, e=em, ctx=ctx):
                        ctx.prompt_cache_count = c.cached_tokens
                        return e.feed(new) and c.cancel()

                    conv.on_tokens = on_tokens
                    conv.on_done = end
                    conv.ctx = ctx
                    live[id(conv)] = (conv, em, rqueue, time.perf_counter(),
                                      (dict(bg.phase_seconds), bg.rounds, sum(m * n for (_, m), n in bg.row_hist.items())))
                    bg.submit(conv)
                except Exception as e:                  # the handler turns this into an error response
                    rqueue.put(e)

            def adopt(conv, em, ctx, rqueue, t0):
                """A request the one-request generator began joins the rounds."""
                def on_tokens(new, c=conv, e=em, ctx=ctx):
                    ctx.prompt_cache_count = c.cached_tokens
                    return e.feed(new) and c.cancel()

                conv.on_tokens = on_tokens
                conv.on_done = end
                conv.ctx = ctx
                live[id(conv)] = (conv, em, rqueue, t0,
                                  (dict(bg.phase_seconds), bg.rounds, sum(m * n for (_, m), n in bg.row_hist.items())))

            # a request that is alone runs on the one-request generator and joins the rounds when a
            # second one arrives (generators with the handoff only: the draft-tree ones)
            alone = getattr(a, "alone", True) and hasattr(mp.gen, "yield_when") and hasattr(mp.proposer, "tail")

            while not self._stop:
                busy = bool(bg.active or bg.waiting)
                request = self._next_request(timeout=None if busy else 0.1)
                if request is not None and not busy and alone:
                    more = self._next_request()
                    if more is None:
                        held = self._serve_dspine(request, together=(bg, adopt))
                        request = held.pop(0) if held else None
                    else:
                        enqueue(request)
                        request = more
                while request is not None:
                    enqueue(request)
                    request = self._next_request()
                if not (bg.active or bg.waiting):
                    continue
                try:
                    bg.step()
                    # the last conversation left in the rounds decodes on alone, as it would have begun
                    if alone and len(bg.active) == 1 and not bg.waiting and not bg.active[0].stopped:
                        request = self._next_request()
                        if request is not None:
                            enqueue(request)
                        else:
                            conv, resume = bg.release()
                            em, rqueue, t0 = live.pop(id(conv))[1:4]
                            for r in self._serve_dspine(None, together=(bg, adopt),
                                                        resumed=(conv, em, conv.ctx, rqueue, t0, resume)):
                                enqueue(r)
                except Exception as e:
                    logging.exception("a round failed; ending the requests that were decoding")
                    for conv, em, rqueue, _, _ in list(live.values()):
                        rqueue.put(e)
                    live.clear()
                    bg = BatchGenerator(mp.target, mp.proposer, max_batch=parallel, cost=mp.cost,
                                keep_sessions=parallel, keep_tokens=a.cache_tokens, tree=getattr(mp, "batch_tree", False), tree_from=getattr(mp, "tree_from", 2),
                                copy=getattr(mp, "batch_copy", False))
                    mx.clear_cache()

        def _serve_dspine(self, request, together=None, resumed=None):
            """One request on the one-request generator. ``together``: (the rounds, adopt) of a server
            that decodes several requests: the kept prompts are shared with the rounds, and when another
            request arrives this one stops between two rounds and joins them. ``resumed``: instead of a
            new request, (conversation, emitter, context, queue, start, resume) of the last conversation
            left in the rounds, which decodes on here. Returns the requests that arrived meanwhile."""
            mp = self.model_provider
            if resumed is None:
                rqueue, args = request[0], request[2]
                seed, max_tokens, sm = args.seed, args.max_tokens, args.sampling
                sampling = dict(temperature=float(sm.temperature or 0.0), top_k=int(sm.top_k or 0),
                                top_p=float(sm.top_p if sm.top_p is not None else 1.0), min_p=float(sm.min_p or 0.0))
            else:
                was, rqueue = resumed[0], resumed[3]
                seed, max_tokens = None, was.max_tokens
                sampling = dict(temperature=was.temperature, top_k=was.top_k, top_p=was.top_p, min_p=was.min_p)
            started = False
            held = []
            bg = together[0] if together else None

            def arrived():
                if not held and not self._stop:
                    r = self._next_request()
                    if r is not None:
                        held.append(r)
                return bool(held)

            try:
                if resumed is None:
                    prompt, ctx, em = self._open(request)
                else:
                    prompt, em, ctx = was.prompt, resumed[1], resumed[2]
                if seed is not None:
                    mx.random.seed(seed)

                def on_tokens(new):
                    ctx.prompt_cache_count = int(getattr(mp.target, "last_reused", 0) or 0)
                    if em.feed(new):
                        raise _Stop          # a stop word or a client that went away: cut it here

                # The generators report the tokens of each round but not the very first token, which
                # they take from the prefill. It is row 0 of the first check, so read it there.
                first = {"sent": resumed is not None}
                target = mp.target

                def send_first(t):
                    if not first["sent"]:
                        first["sent"] = True
                        on_tokens([int(t)])

                def verify(tokens, *x, **k):
                    send_first(tokens[0, 0].item())
                    return type(target).verify(target, tokens, *x, **k)

                def verify_tree(tokens, *x, **k):
                    send_first(tokens[0])
                    return type(target).verify_tree(target, tokens, *x, **k)

                started = True
                t0 = time.perf_counter() if resumed is None else resumed[4]
                out, st = [], None
                block = int(mp.gen.draft.block_size)
                if bg is not None:
                    if resumed is None:
                        bg._load_session(list(prompt))      # a kept prompt this one starts like
                    mp.gen.yield_when = arrived
                target.verify, target.verify_tree = verify, verify_tree
                extra = {}
                if resumed is None:
                    target.mark_hint = self._system_len
                else:
                    extra["resume"] = resumed[5]
                try:
                    out, st = mp.gen.generate(prompt, max_tokens=max_tokens, stop_ids=mp.stop_ids,
                                              on_tokens=on_tokens, **sampling, **extra)
                    clean = True
                    if out:
                        send_first(out[0])      # a one-token answer never reaches a check
                except _Stop:
                    clean = False
                finally:
                    del target.verify, target.verify_tree
                    if bg is not None:
                        mp.gen.yield_when = None
                        if int(mp.gen.draft.block_size) != block:
                            mp.gen.draft.set_block_size(block)
                if not clean:
                    self._forget()
                hand = getattr(mp.gen, "handoff", None) if bg is not None and clean else None
                if hand is not None and not em.done:
                    from .batch_gen import Conversation

                    conv = Conversation(list(prompt), max_tokens=max_tokens, stop_ids=mp.stop_ids, **sampling)
                    conv.tokens, conv.stats = list(out), st
                    if resumed is not None:
                        conv.cached_tokens = was.cached_tokens
                    bg.adopt(conv, hand["anchor"], hand["hidden"], hand["state"], hand["started"])
                    together[1](conv, em, ctx, rqueue, t0)
                    mp.gen.handoff = None
                    logging.info("request: prompt %d tokens, %d new tokens alone so far; another request arrived, "
                                 "decoding together from here", len(prompt), em.n)
                    return held
                dt = time.perf_counter() - t0
                ph = getattr(mp.gen, "phase_seconds", None) or {}
                n_rounds = max(st.rounds, 1) if st is not None else 1
                logging.info("request: prompt %d tokens (%d from cache), %d new tokens, %.1f s (%s); prefill %.2f s, "
                             "%d rounds, %.2f tokens per round, %.1f ms per round, %s, memory %.1f GB (peak %.1f); "
                             "ms per round draft %.1f pick %.1f check %.1f other %.1f",
                             len(prompt), ctx.prompt_cache_count, em.n, dt,
                             "alone" if resumed is None else "alone at the end", st.prefill_seconds if st else 0.0,
                             st.rounds if st else 0, (st.committed_tokens / n_rounds) if st else 0.0,
                             (1000 * st.decode_seconds / n_rounds) if st else 0.0,
                             "sampled" if sampling["temperature"] > 0 else "greedy",
                             mx.get_active_memory() / 2**30, mx.get_peak_memory() / 2**30,
                             *(1000 * ph.get(k, 0.0) / n_rounds for k in ("draft", "pick", "verify", "other")))
                em.close(mp.stop_ids[0] if mp.stop_ids else 0)
                if bg is not None:
                    bg.keep_finished(list(prompt), list(out))
                mx.clear_cache()
            except Exception as e:  # the handler turns this into an error response
                if started:
                    self._forget()
                    if bg is not None:
                        mp.target.cache = None
                        mp.target._drop_session()
                rqueue.put(e)
            return held

        def _forget(self):
            """After a generation cut short the cache may hold an unsettled check: start the next one clean."""
            target = self.model_provider.target
            try:
                if getattr(target, "rewind_to_prompt", lambda: False)():
                    return                      # the prompt (and its marks) stay usable
                target._seq = None
                target.reset()
            except Exception:
                logging.exception("could not reset the target after an interrupted generation")

    class Handler(S.APIHandler):
        """mlx-lm's handler; lists the one model this server runs and stays quiet when a client leaves."""

        def handle_models_request(self):
            self._set_completion_headers(200)
            self.end_headers()
            data = [{"id": str(a.target), "object": "model", "created": self.created, "owned_by": "dspine-hybrid-mlx"}]
            self.wfile.write(json.dumps({"object": "list", "data": data}).encode())
            self.wfile.flush()

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                logging.info("client %s closed the connection", self.client_address[0])

    Generator.Handler = Handler
    return Generator, Provider(_args_namespace(a))


DEFAULT_TARGET = "mlx-community/Qwen3.8-27B-4bit"
DEFAULT_DRAFT = "netprojects/Qwen3.8-27B-DSpine-b16-MLX"
DEFAULT_PORT = 8791


def free_port(host: str, first: int = DEFAULT_PORT, tries: int = 20) -> int:
    """``first`` or the next port after it that nothing listens on."""
    import socket

    for port in range(first, first + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port between {first} and {first + tries - 1}; pass --port")


def settle(a):
    """Fill what the user left out, so the bare command works: fast mode when the kernel wheel is
    installed, several requests at once, a free port."""
    if a.fast is None:
        from . import fast_verify

        a.fast = bool(fast_verify.install())      # False without the kernel wheel
    if a.parallel is None:
        a.parallel = 8      # ds-74: the total keeps rising to 8, and fewer active requests run at their own speed
    if a.port is None:
        a.port = free_port(a.host)
    if getattr(a, "cache_tokens", 0) is None:
        import os

        # what the memory limit leaves after the weights, the drafter and one long request in work
        # (about 24 GB), at about 0.35 GB per 1,000 kept tokens of a 27B target (eb-13 / eb-14)
        limit = getattr(a, "memory_limit_gb", None) or float(os.environ.get("DSPINE_MEMORY_LIMIT_GB", "60"))
        a.cache_tokens = max(16000, int((limit - 24.0) / 0.35 * 1000))
    return a


def main() -> None:
    p = argparse.ArgumentParser(description="OpenAI-compatible server for Qwen3.8-27B with DSpine drafting")
    p.add_argument("--target", default=DEFAULT_TARGET, help="MLX build of Qwen3.8-27B (path or Hugging Face repo)")
    p.add_argument("--draft", default=DEFAULT_DRAFT, help="DSpine drafter (path or Hugging Face repo)")
    p.add_argument("--calibration", default=None, help="acceptance fit from scripts/fit_acceptance.py")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None, help="default: 8791, or the next free port after it")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=None,
                   help="the fastest measured setup: --fast-verify --adaptive --confidence calibrated --tree --a8-prefill "
                        "(default: on when the kernel wheel is installed)")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--confidence", default="vocab", choices=["vocab", "calibrated"])
    p.add_argument("--tree", action="store_true")
    p.add_argument("--copy", action="store_true")
    p.add_argument("--fast-verify", action="store_true")
    p.add_argument("--a8-prefill", action="store_true")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--draft-bits", type=int, default=None)
    p.add_argument("--parallel", type=int, default=None,
                   help="decode up to N requests together, one target check per round for all (default 8)")
    p.add_argument("--alone", action=argparse.BooleanOptionalAction, default=True,
                   help="a request that is alone runs on the one-request generator and joins the rounds when "
                        "another arrives (default on; --no-alone keeps every request in the rounds)")
    p.add_argument("--remeasure", action="store_true",
                   help="measure the target costs again at this start (they are kept for 14 days in "
                        "~/.cache/dspine-hybrid, which makes later starts shorter)")
    p.add_argument("--cache-tokens", type=int, default=None,
                   help="most tokens of finished requests kept for follow-up turns (a 32K prompt holds about 11 GB; "
                        "default: from the memory limit, 45,000 at 40 GB and 100,000 at 60 GB)")
    p.add_argument("--thinking", action="store_true",
                   help="thinking on by default (a request can still set chat_template_kwargs.enable_thinking)")
    p.add_argument("--temperature", type=float, default=0.0, help="default when a request sets none")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048, help="default when a request sets none")
    p.add_argument("--memory-limit-gb", type=float, default=None)
    p.add_argument("--allowed-origins", nargs="+", default=["*"])
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = settle(p.parse_args())
    logging.basicConfig(level=getattr(logging, a.log_level), format="%(asctime)s %(levelname)s %(message)s")
    print(f"Model   {a.target}\nDrafter {a.draft}\nFast mode {'on' if a.fast else 'off'}, up to {a.parallel} "
          f"request(s) at once, up to {a.cache_tokens} tokens of finished requests kept.\nLoading (the first start downloads the weights, about 17 GB for a 4-bit build; "
          f"plan for about 24 GB of free memory) ...", flush=True)

    from mlx_lm import server as S

    Generator, provider = build(a)
    generator = Generator(provider, S.LRUPromptCache(1))
    provider.loaded.wait()                 # the first request must not pay for the load (engine bench eb-14: 41 s)
    if provider.load_error is not None:
        raise SystemExit(f"could not load: {provider.load_error}")
    print(f"\nReady. In your tool, set the OpenAI address to  http://{a.host}:{a.port}/v1  (any API key, model "
          f"\"{a.target}\").\n", flush=True)
    S._run_http_server(a.host, a.port, generator, handler_class=Generator.Handler)


if __name__ == "__main__":
    main()
