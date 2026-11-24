"""Send the same four requests to a running server one after the other, then all at once.

    python scripts/parallel_check.py http://127.0.0.1:8791 [max_tokens]

Prints total tokens per second for both and whether every answer is the same text.
"""
import json
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "Write a Python class LRUCache with get and put in O(1) and five unit tests.",
    "Explain to a ten year old why the sky is blue and why sunsets are red.",
    "Give me a three day plan for visiting Athens with a family, with one tip per day.",
]


def ask(base, prompt, n, out, i):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": n,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    out[i] = (d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"])


def main():
    base = sys.argv[1].rstrip("/")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/health", timeout=5).read()
            break
        except Exception:
            time.sleep(5)
    else:
        sys.exit("server did not come up")
    warm = [None]
    ask(base, "Say hello.", 16, warm, 0)
    one = [None] * len(PROMPTS)
    t0 = time.perf_counter()
    for i, p in enumerate(PROMPTS):
        ask(base, p, n, one, i)
    t_one = time.perf_counter() - t0
    for rep in range(2):                      # the first pass of a new batch shape warms up
        tog = [None] * len(PROMPTS)
        th = [threading.Thread(target=ask, args=(base, p, n, tog, i)) for i, p in enumerate(PROMPTS)]
        t0 = time.perf_counter()
        [t.start() for t in th]
        [t.join() for t in th]
        t_tog = time.perf_counter() - t0
    if any(x is None for x in tog):
        sys.exit("a request failed")
    n1, n2 = sum(x[1] for x in one), sum(x[1] for x in tog)
    same = sum(a[0] == b[0] for a, b in zip(one, tog))
    print(f"[parallel] one after the other: {n1} tokens in {t_one:.1f} s = {n1 / t_one:.1f} tok/s (prefill included)")
    print(f"[parallel] four at once: {n2} tokens in {t_tog:.1f} s = {n2 / t_tog:.1f} tok/s (x{(n2 / t_tog) / (n1 / t_one):.2f})")
    print(f"[parallel] same text as one after the other: {same}/{len(PROMPTS)}")


if __name__ == "__main__":
    main()
