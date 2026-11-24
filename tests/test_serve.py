"""The OpenAI-compatible server: mlx-lm's HTTP side with this kit's generator behind it (tiny model, CPU)."""
import argparse
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import mlx.core as mx
import pytest

from dspine_hybrid import serve
from dspine_mlx.generate import DSpineGenerator
from dspine_hybrid.hybrid_target import HybridTargetAdapter
from tiny import TARGET_IDS, VOCAB, make_draft, make_target

EOS, IM_START, THINK, UNTHINK = VOCAB - 1, VOCAB - 3, VOCAB - 4, VOCAB - 5
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    words = [f"w{i}" for i in range(VOCAB)]
    words[EOS], words[IM_START], words[VOCAB - 2] = "<|im_end|>", "<|im_start|>", "<|mask|>"
    for i, w in enumerate(["user", "assistant", "system", "hello", "there", "again"]):
        words[i] = w
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="w10"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers import decoders
    tk.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", unk_token="w10")
    fast.chat_template = TEMPLATE
    return TokenizerWrapper(fast)


class _Server:
    def __init__(self, stop_ids=(EOS,), tree=False, parallel=1):
        from mlx_lm import server as S

        a = argparse.Namespace(
            target="tiny", draft="tiny-draft", thinking=True, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
            max_tokens=24, allowed_origins=["*"], log_level="INFO", memory_limit_gb=None, draft_bits=None,
            fast=False, block_size=None, adaptive=False, confidence="vocab", copy=False, tree=False,
            fast_verify=False, a8_prefill=False, parallel=parallel, cache_tokens=40000)
        Generator, provider = serve.build(a)
        tok = _tokenizer()

        def load_kit():
            model = make_target(0)
            adapter = HybridTargetAdapter(model, tok, TARGET_IDS)
            adapter.enable_prompt_cache(True)
            draft, _, _ = make_draft(10)
            draft.bind(model)
            if tree:
                from dspine_hybrid.lanes import LaneDSpineGenerator

                gen = LaneDSpineGenerator(adapter, draft, block_size=6, tree=True)
            else:
                gen = DSpineGenerator(adapter, draft)
            provider.gen, provider.target = gen, adapter
            provider.model, provider.tokenizer = model, tok
            provider.model_key = ("tiny", None, "tiny-draft")
            provider.stop_ids = tuple(stop_ids)
            provider.batch_tree = bool(tree) and parallel > 1

        provider.load_kit = load_kit
        self.provider = provider
        self.generator = Generator(provider, S.LRUPromptCache(1))
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *x, **k: Generator.Handler(self.generator, *x, system_fingerprint="test", **k))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def post(self, path, body, stream=False):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode()
        if not stream:
            return json.loads(raw)
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and "[DONE]" not in line]

    def close(self):
        self.httpd.shutdown()
        self.generator.stop_and_join()


@pytest.fixture(scope="module")
def srv():
    s = _Server()
    yield s
    s.close()


MSG = [{"role": "user", "content": "hello there w40 w41 w42 w43 w44 w45 w46 w47"}]


def test_chat_completion_and_stream_agree(srv):
    r = srv.post("/v1/chat/completions", {"model": "gpt-4o", "messages": MSG, "max_tokens": 20})
    choice = r["choices"][0]
    text = choice["message"]["content"]
    assert text and choice["finish_reason"] in ("length", "stop")
    assert r["usage"]["completion_tokens"] <= 20 and r["usage"]["prompt_tokens"] > 8
    chunks = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20, "stream": True}, stream=True)
    streamed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
    assert streamed == text
    assert [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")][-1] == choice["finish_reason"]


def test_same_tokens_as_the_generator(srv):
    r = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20})
    tok = srv.provider.tokenizer
    ids = tok.apply_chat_template(MSG, add_generation_prompt=True)
    model = make_target(0)
    adapter = HybridTargetAdapter(model, tok, TARGET_IDS)
    draft, _, _ = make_draft(10)
    draft.bind(model)
    out, _ = DSpineGenerator(adapter, draft).generate(ids, max_tokens=20, stop_ids=(EOS,))
    want = tok.decode([t for t in out if t != EOS])
    assert r["choices"][0]["message"]["content"].split() == want.split()


def test_follow_up_turn_reuses_the_cache(srv):
    r1 = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 16})
    answer = r1["choices"][0]["message"]["content"]
    msgs = MSG + [{"role": "assistant", "content": answer}, {"role": "user", "content": "again w50 w51"}]
    r2 = srv.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 16})
    assert r2["choices"][0]["message"]["content"]
    assert srv.provider.target.last_reused >= r1["usage"]["prompt_tokens"]
    # a different conversation afterwards still works (cache is dropped or restored, never wrong)
    r3 = srv.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "w60 w61 w62 w63"}], "max_tokens": 8})
    assert r3["choices"][0]["message"]["content"]


def test_stop_word_cuts_and_the_next_request_is_clean(srv):
    full = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20})["choices"][0]["message"]["content"]
    words = full.split()
    assert len(words) >= 6
    stop = words[4]
    r = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20, "stop": [stop]})
    cut = r["choices"][0]["message"]["content"]
    assert r["choices"][0]["finish_reason"] == "stop"
    assert cut.split() == words[: words.index(stop)]
    again = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20})["choices"][0]["message"]["content"]
    assert again == full


def test_text_completion_and_models(srv):
    r = srv.post("/v1/completions", {"prompt": "hello there w40 w41 w42 w43", "max_tokens": 8})
    assert r["choices"][0]["text"]
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health", timeout=30) as h:
        assert json.loads(h.read())["status"] == "ok"
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/v1/models", timeout=30) as h:
        assert [m["id"] for m in json.loads(h.read())["data"]] == ["tiny"]


def test_second_conversation_reuses_the_system_prompt(srv):
    srv.provider.load_default()
    system = {"role": "system", "content": " ".join(f"w{40 + i % 50}" for i in range(60))}
    srv.provider.target.mark_min = 8
    ask = lambda q: srv.post("/v1/chat/completions", {"messages": [system, {"role": "user", "content": q}], "max_tokens": 10})
    a1 = ask("hello w70 w71")
    a2 = ask("there w72 w73 w74")
    assert srv.provider.target.last_reused >= 60
    b = _Server()
    b2 = b.post("/v1/chat/completions", {"messages": [system, {"role": "user", "content": "there w72 w73 w74"}], "max_tokens": 10})
    b.close()
    assert a2["choices"][0]["message"]["content"] == b2["choices"][0]["message"]["content"] and a1


def test_tree_generator_streams_every_token(srv):
    """The first token comes from the prefill, not from a round: it must still be streamed, once."""
    plain = srv.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20})
    t = _Server(tree=True)
    try:
        r = t.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20})
        chunks = t.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 20, "stream": True}, stream=True)
        one = t.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 1})
    finally:
        t.close()
    text = r["choices"][0]["message"]["content"]
    assert text.split() == plain["choices"][0]["message"]["content"].split()
    assert r["usage"]["completion_tokens"] == plain["usage"]["completion_tokens"]
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices")) == text
    assert one["choices"][0]["message"]["content"].split() == text.split()[:1]
    assert one["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("tree", [False, True])
def test_parallel_requests_get_the_same_answers(srv, tree):
    """--parallel: requests decoded together answer as they do one at a time (with draft trees too)."""
    asks = [MSG,
            [{"role": "user", "content": "again w60 w61 w62 w63 w64"}],
            [{"role": "user", "content": "hello w70 w71 there w72 w73 w74 w75 w76 w77 w78 w79"}]]
    want = [srv.post("/v1/chat/completions", {"messages": m, "max_tokens": 24})["choices"][0] for m in asks]
    t = _Server(parallel=2, tree=tree)             # three requests, two slots
    got = [None] * len(asks)
    try:
        def one(i):
            if i == 1:
                chunks = t.post("/v1/chat/completions", {"messages": asks[i], "max_tokens": 24, "stream": True}, stream=True)
                got[i] = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
            else:
                r = t.post("/v1/chat/completions", {"messages": asks[i], "max_tokens": 24})
                got[i] = r["choices"][0]["message"]["content"]
                assert r["choices"][0]["finish_reason"] == want[i]["finish_reason"]
                assert r["usage"]["completion_tokens"] > 0

        th = [threading.Thread(target=one, args=(i,)) for i in range(len(asks))]
        [x.start() for x in th]
        [x.join(timeout=120) for x in th]
        words = want[0]["message"]["content"].split()
        cut = t.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 24, "stop": [words[4]]})
        after = t.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 24})
    finally:
        t.close()
    assert got == [w["message"]["content"] for w in want]
    assert cut["choices"][0]["finish_reason"] == "stop"
    assert cut["choices"][0]["message"]["content"].split() == words[: words.index(words[4])]
    assert after["choices"][0]["message"]["content"] == want[0]["message"]["content"]


def test_the_bare_command_gets_working_defaults():
    import socket

    from dspine_hybrid import serve

    a = serve.settle(argparse.Namespace(fast=None, parallel=None, port=None, host="127.0.0.1"))
    assert a.parallel == 8 and isinstance(a.fast, bool) and a.port >= serve.DEFAULT_PORT
    c = serve.settle(argparse.Namespace(fast=False, parallel=1, port=9001, host="127.0.0.1", cache_tokens=None, memory_limit_gb=40))
    assert 45000 <= c.cache_tokens <= 46000
    with socket.socket() as s:                       # a taken port is skipped
        s.bind(("127.0.0.1", a.port))
        s.listen(1)
        assert serve.free_port("127.0.0.1", a.port) > a.port
    b = serve.settle(argparse.Namespace(fast=False, parallel=1, port=9001, host="127.0.0.1"))
    assert (b.fast, b.parallel, b.port) == (False, 1, 9001)


def test_a_lone_request_runs_alone_and_joins_when_another_arrives(srv, caplog):
    """--parallel: one request runs on the one-request generator; a second one arriving meanwhile makes
    it join the rounds. The answers are the ones each gives alone, and follow-up turns reuse the cache."""
    import time

    asks = [MSG, [{"role": "user", "content": "again w60 w61 w62 w63 w64"}]]
    want = [srv.post("/v1/chat/completions", {"messages": m, "max_tokens": 24})["choices"][0]["message"]["content"]
            for m in asks]
    import logging

    caplog.set_level(logging.INFO)
    t = _Server(parallel=2, tree=True)
    got = [None] * len(asks)
    try:
        r1 = t.post("/v1/chat/completions", {"messages": MSG, "max_tokens": 16})
        gen = t.provider.gen
        assert gen.lane_stats and gen.handoff is None          # served by the one-request generator
        assert r1["choices"][0]["message"]["content"].split() == want[0].split()[:16]
        msgs = MSG + [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
                      {"role": "user", "content": "again w50 w51"}]
        r2 = t.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 8})
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= r1["usage"]["prompt_tokens"]

        calls, handed = [], []
        target, generate = t.provider.target, gen.generate
        slow = type(target).prefill

        def prefill(tokens, *x, **k):                          # the first request is still reading its prompt
            if not calls:
                time.sleep(0.5)
            calls.append(1)
            return slow(target, tokens, *x, **k)

        def counted(*x, **k):
            out = generate(*x, **k)
            handed.append(gen.handoff is not None)
            return out

        target.prefill, gen.generate = prefill, counted

        def one(i):
            time.sleep(0.2 * i)
            got[i] = t.post("/v1/chat/completions", {"messages": asks[i], "max_tokens": 24})["choices"][0]["message"]["content"]

        th = [threading.Thread(target=one, args=(i,)) for i in range(len(asks))]
        [x.start() for x in th]
        [x.join(timeout=120) for x in th]
        del target.prefill, gen.generate
        after = t.post("/v1/chat/completions", {"messages": asks[1], "max_tokens": 24})["choices"][0]["message"]["content"]
    finally:
        t.close()
    # the first began alone and was handed over; the one left at the end went back to the generator
    assert handed == [True, False]
    lines = [r.getMessage() for r in caplog.records if r.name == "root"]       # every log line formats
    assert any("(alone at the end)" in x for x in lines) and any("(alone);" in x for x in lines)
    assert got == want and after == want[1]


def test_measured_costs_are_kept_between_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPINE_CACHE_DIR", str(tmp_path))
    a = argparse.Namespace(target="t", draft="d", drafter="dspine", draft_bits=None, block_size=None, parallel=8,
                           fast=True, tree=False, fast_verify=False, adaptive=False, confidence="vocab", remeasure=False)
    path = serve._cost_file(a, 16)
    assert path.startswith(str(tmp_path)) and serve._saved_costs(path) is None
    serve._save_costs(path, [0.0, 0.03], {1: [0.0, 0.03], 2: [0.0, 0.04]})
    assert serve._saved_costs(path) == dict(single=[0.0, 0.03], batch={1: [0.0, 0.03], 2: [0.0, 0.04]})
    assert serve._saved_costs(path, days=0.0) is None                 # too old: measured again
    a.parallel = 4
    assert serve._cost_file(a, 16) != path                             # another setup, another file
    del a.remeasure
    assert serve._cost_file(a, 16) is None
