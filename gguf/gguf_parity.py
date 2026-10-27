"""Token-id parity of an Agaram 2.5B GGUF against the HF tokenizer (2026-10-12). Runs ON spark-1003.

Texts: every turn and reference of openchat v1/v2/v3, plus 300 held-out SFT conversations rendered with the chat
template (Tamil script, Tanglish, English, code, tool-call JSON). Raw texts are compared without special-token
parsing; rendered conversations are compared with special tokens parsed (as llama.cpp's --jinja path would see them).
Reports exact-match counts, the first differing texts, and how many texts contain ZWJ/ZWNJ (the only regex difference
between our pre-tokenizer and llama.cpp's qwen35 pattern).

    python gguf_parity_20261012.py --gguf ~/gguf_2p5b/agaram-2.5b-instruct-f16.gguf --hf ~/r4_eval/models/step000017 \
        --texts ~/gguf_2p5b/parity_texts.jsonl --out ~/gguf_2p5b/parity.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def llama_ids(binary: str, gguf: str, text: str, special: bool) -> list[int]:
    cmd = [binary, "-m", gguf, "--ids", "--log-disable", "--no-bos", "--stdin", "--no-escape"]
    if not special:
        cmd.append("--no-parse-special")
    out = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, check=True).stdout.decode()
    line = [ln for ln in out.splitlines() if ln.startswith("[")][-1]
    return json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--binary", default=str(Path.home() / "llama.cpp/build/bin/llama-tokenize"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    rows = [json.loads(l) for l in open(a.texts, encoding="utf-8")]
    res = {"raw": [0, 0], "chat": [0, 0], "zwj_texts": 0, "diffs": []}
    for i, r in enumerate(rows):
        kind, text = r["kind"], r["text"]
        special = kind == "chat"
        hf = tok.encode(text, add_special_tokens=False) if not special else tok(text, add_special_tokens=False)["input_ids"]
        ll = llama_ids(a.binary, a.gguf, text, special)
        res[kind][1] += 1
        if "‌" in text or "‍" in text:
            res["zwj_texts"] += 1
        if hf == ll:
            res[kind][0] += 1
        elif len(res["diffs"]) < 15:
            j = next((k for k, (x, y) in enumerate(zip(hf, ll)) if x != y), min(len(hf), len(ll)))
            res["diffs"].append({"id": r["id"], "kind": kind, "n_hf": len(hf), "n_ll": len(ll), "first_diff": j,
                                 "hf_tok": tok.convert_ids_to_tokens(hf[j:j + 4]), "ll_ids": ll[j:j + 4],
                                 "context": tok.decode(hf[max(0, j - 3):j + 4])})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)} raw {res['raw']} chat {res['chat']}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "diffs"}), flush=True)
    for d in res["diffs"][:15]:
        print(json.dumps(d, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
