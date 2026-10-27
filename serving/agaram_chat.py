#!/usr/bin/env python3
"""Chat with அகரம் / Agaram 2.5B instruct from the command line.

Settings are the ones every reported chat result used: ChatML via the tokenizer's chat template, no system prompt,
greedy decoding, up to 512 new tokens, stop on <|im_end|> (id 2).

One thing is added outside the model: a crisis-resource footer. The model was trained to point to Tele-MANAS (14416)
and the emergency number 112, but it does not do so every time, and it is not a crisis service. When a user message
contains one of a small set of distress phrases, the wrapper prints the resources itself. The phrase list is short
and will miss many messages; an application should use a proper classifier.

    python serving/agaram_chat.py                                   # interactive, murai-labs/agaram-2.5b-instruct
    python serving/agaram_chat.py --model path/to/checkpoint --prompt "Vanakkam! Nee yaaru?"
"""
from __future__ import annotations

import argparse
import sys

STOP_TOKEN_ID = 2           # <|im_end|>
MAX_NEW_TOKENS = 512
MAX_CONTEXT = 8192
CRISIS_FOOTER = ("[If you are in distress: Tele-MANAS 14416 (free, 24x7, Tamil supported) · emergency 112 · "
                 "நீங்கள் மன அழுத்தத்தில் இருந்தால் Tele-MANAS 14416 அல்லது 112-ஐ அழையுங்கள்]")
CRISIS_PHRASES = (
    # English
    "kill myself", "suicide", "end my life", "want to die", "self harm", "self-harm", "hurt myself", "not safe",
    # Tanglish
    "saaga poren", "saaganum", "sethuduven", "sethu poganum", "uyiroda irukka venaam",
    # Tamil script
    "தற்கொலை", "சாகணும்", "சாகப் போறேன்", "செத்துடுவேன்", "உயிரோட இருக்க வேண்டாம்", "வாழ விருப்பம் இல்ல",
    "என்னை நானே காயப்படுத்த",
)


def needs_footer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in CRISIS_PHRASES)


def load(model_id: str, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = dict(dtype=torch.bfloat16)
    if device is None:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device is not None:
        model = model.to(device)
    model.eval()
    return tok, model


def reply(tok, model, messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[str, bool]:
    import torch

    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"]
    if ids.shape[1] + max_new_tokens > MAX_CONTEXT:
        raise ValueError(f"conversation is {ids.shape[1]} tokens; start a new one (context {MAX_CONTEXT})")
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=STOP_TOKEN_ID,
                             pad_token_id=tok.pad_token_id)
    new = out[0, ids.shape[1]:]
    stopped = bool((new == STOP_TOKEN_ID).any())
    return tok.decode(new, skip_special_tokens=True).strip(), stopped


def show(user: str, text: str, stopped: bool) -> None:
    print(text if stopped else text + "\n[cut at the token limit; the model did not stop on its own]", flush=True)
    if needs_footer(user):
        print(CRISIS_FOOTER, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="murai-labs/agaram-2.5b-instruct", help="HF repo id or local checkpoint dir")
    ap.add_argument("--prompt", default=None, help="one message, then exit")
    ap.add_argument("--device", default=None, help="e.g. cuda or cpu; default device_map=auto")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    tok, model = load(args.model, args.device)
    if args.prompt:
        text, stopped = reply(tok, model, [{"role": "user", "content": args.prompt}], args.max_new_tokens)
        show(args.prompt, text, stopped)
        return 0

    print("அகரம் / Agaram 2.5B · Murai Labs · research model, may be wrong (empty line to quit)", file=sys.stderr)
    history: list[dict] = []
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        text, stopped = reply(tok, model, history, args.max_new_tokens)
        show(user, text, stopped)
        history.append({"role": "assistant", "content": text})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
