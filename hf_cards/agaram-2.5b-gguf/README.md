---
license: apache-2.0
language:
- ta
- en
base_model: murai-labs/agaram-2.5b-instruct
tags:
- gguf
- llama.cpp
- tamil
- tanglish
- agaram
---

# Agaram 2.5B (அகரம்) — GGUF

GGUF files of Agaram 2.5B for llama.cpp and tools built on it. Agaram is a Tamil–English model **trained from scratch**
by Murai Labs and Ramchand Kumaresan; it is not a fine-tune of any other model. Model cards:
[chat](https://huggingface.co/murai-labs/agaram-2.5b-instruct), [base](https://huggingface.co/murai-labs/agaram-2.5b).
Code: [Murai-Labs/Agaram2.5b](https://github.com/Murai-Labs/Agaram2.5b). Paper: [arXiv:2610.18420](https://arxiv.org/abs/2610.18420).
Try the chat model online: [chat demo](https://huggingface.co/spaces/mechramc/agaram-2.5b-chat).

The chat (instruct) GGUFs were rebuilt on 2026-10-26 from the released Agaram 2.5B-Instruct weights with llama.cpp
`b11574`; the source model.safetensors SHA-256 is
`0123c671b233b3e0a6b411e3e88ecb569485f4e20b71719329ac88b75e05bdc6`. The base GGUFs are the 2026-10-12 build, also
llama.cpp `b11574`. File hashes are in `SHA256SUMS`.

| File | Size | Use |
|---|---|---|
| `agaram-2.5b-instruct-f16.gguf` | 5.06 GB | chat, half precision; greedy replies can differ from the HF model's |
| `agaram-2.5b-instruct-Q8_0.gguf` | 2.69 GB | chat, 8-bit; greedy replies can differ from the HF model's |
| `agaram-2.5b-instruct-Q4_K_M.gguf` | 1.56 GB | chat, small 4-bit; answers drift and can be wrong where the full model is right (see below) |
| `agaram-2.5b-f16.gguf` | 5.06 GB | base (continuation model) |
| `agaram-2.5b-Q8_0.gguf` | 2.69 GB | base |
| `agaram-2.5b-Q4_K_M.gguf` | 1.56 GB | base, small |

```
llama-cli -m agaram-2.5b-instruct-Q8_0.gguf --jinja --temp 0
llama-server -m agaram-2.5b-instruct-Q8_0.gguf --jinja
```

- Use `--jinja` so llama.cpp applies the chat template embedded in the file. No system prompt is needed.
- Greedy decoding (`--temp 0`) is what every reported result used.
- The instruct files stop on `<|im_end|>` (id 2). The base files stop on `<|endoftext|>` (id 0).
- Context: 8,192 tokens.

## Requirements: a llama.cpp that knows the `qwen35` pre-tokenizer

Agaram's tokenizer is a byte-level BPE whose pre-tokenizer is Qwen2's split pattern with Unicode combining marks and
the zero-width joiners added to the letter class, so that Tamil vowel signs stay attached to their consonants.
llama.cpp has no entry for this exact pattern. These files use llama.cpp's `qwen35` pre-tokenizer, whose pattern is the
same except that it does not include the zero-width joiners (U+200C, U+200D). Use a llama.cpp build that recognises
`qwen35` (b11574 or newer); older builds, and apps that bundle them, will refuse to load the files or tokenize
differently.

## What was checked

- **Tokenization (chat files, 2026-10-26).** llama.cpp's token ids were compared with the Hugging Face tokenizer on 961
  texts: every prompt and reference answer of our three chat evaluation sets (661 texts in Tamil script, Tanglish and
  English) and 300 held-out training conversations rendered with the chat template (special tokens, tool calls, code,
  JSON). All 961 match exactly for the f16 and Q4_K_M chat files; the Q8_0 file was not tested separately. The f16 base
  file matched on the same texts on 2026-10-12. None of these texts contains a zero-width joiner, the one case where
  the patterns differ, so that case is untested.
- **Replies (chat files, 2026-10-26).** Greedy replies to 8 Tanglish prompts from our evaluation set, compared with the
  HF model's greedy replies (512-token cap): f16 matches character for character on 5 of 8, Q8_0 on 5 of 8, Q4_K_M on
  0 of 8. Where f16 differs, it diverges after 6, 44 and 86 characters. Some differences change the answer: asked how
  many slippers make a pair, Q4_K_M says 12 where the HF model says 2. Prefer Q8_0 or f16 if you can; neither is
  guaranteed to reproduce the HF model's replies. These are small conversion checks, not benchmark scores.
- **Metadata and chat template.** All three chat files use the `qwen35` pre-tokenizer and stop on id 2 (`<|im_end|>`),
  and embed the chat template. The `--jinja` check (the embedded template gives the same prompt as the HF template) was
  run on the previous release's files on 2026-10-12; the chat template and tokenizer files of this release are
  byte-identical to those, but the check was not rerun on these files.

## Model, evaluation, limitations and disclosures

See the [chat model card](https://huggingface.co/murai-labs/agaram-2.5b-instruct). In short: strong at Tamil–English
translation for its size, weak on knowledge (about half of everyday questions answered acceptably), and it is a
research model, not a crisis service.

## Creators

Murai Labs and Ramchand Kumaresan. Licence: Apache-2.0; upstream data terms in the chat model card. Technical report: https://arxiv.org/abs/2610.18420
