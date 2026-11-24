"""Requests that separate what makes a long answer slow on a running server (engine bench eb-13):
greedy or sampled, a fresh or a cached prompt, before or after a long prompt. Read the server's
"request:" log lines next to this script's output.

    python scripts/diag_long_requests.py --base http://127.0.0.1:8791/v1 [--tokens 1500] [--long 30000]
"""
import argparse
import json
import time
import urllib.request

CODE = "Write a complete Python module for an LRU cache with TTL support, type hints, docstrings and unit tests. Variant {}."
STORY = "Write a long story about a lighthouse keeper who finds a message in a bottle. Version {}."


def ask(base, content, tokens, temperature):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": tokens, "temperature": temperature}
    if temperature > 0:
        body.update(top_p=0.8, top_k=20)
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.perf_counter() - t0
    n = out["usage"]["completion_tokens"]
    return n, dt, out["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8791/v1")
    p.add_argument("--tokens", type=int, default=1500)
    p.add_argument("--long", type=int, default=30000, help="about this many tokens in the long prompt")
    a = p.parse_args()
    filler = " ".join(f"Section {i}: the committee reviewed item {i} and recorded {i * 7} remarks." for i in range(a.long // 16))
    steps = [
        ("code greedy fresh", CODE.format(1), a.tokens, 0.0),
        ("code greedy same prompt (cached)", CODE.format(1), a.tokens, 0.0),
        ("code sampled same prompt (cached)", CODE.format(1), a.tokens, 0.7),
        ("code sampled fresh", CODE.format(2), a.tokens, 0.7),
        ("story greedy fresh", STORY.format(1), a.tokens, 0.0),
        ("story sampled fresh", STORY.format(2), a.tokens, 0.7),
        ("code greedy short answer fresh", CODE.format(3), 384, 0.0),
        ("LONG prompt summary", filler + "\n\nSummarise the remarks above in five sentences.", 300, 0.0),
        ("code greedy fresh after the long prompt", CODE.format(4), a.tokens, 0.0),
        ("story greedy fresh after the long prompt", STORY.format(3), a.tokens, 0.0),
        ("story sampled fresh after the long prompt", STORY.format(4), a.tokens, 0.7),
        ("LONG prompt again (cached)", filler + "\n\nSummarise the remarks above in five sentences.", 300, 0.0),
        ("LONG prompt, new question", filler + "\n\nList three items with the most remarks.", 300, 0.0),
    ]
    for name, content, tokens, temp in steps:
        n, dt, text = ask(a.base, content, tokens, temp)
        tail = " ".join(text[-160:].split())
        distinct = len(set(text.split())) / max(len(text.split()), 1)
        print(f"[diag] {name}: {n} tokens in {dt:.1f} s = {n / dt:.1f} tok/s wall; distinct words {distinct:.2f}; "
              f"ends: ...{tail}", flush=True)


if __name__ == "__main__":
    main()
