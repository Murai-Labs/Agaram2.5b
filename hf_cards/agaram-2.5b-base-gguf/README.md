---
license: apache-2.0
language:
- ta
- en
base_model: murai-labs/agaram-2.5b
tags:
- gguf
- llama.cpp
- tamil
- tanglish
- agaram
- pretrained
---

# Agaram 2.5B (அகரம்) — base model GGUF

GGUF files of the Agaram 2.5B **base** model for llama.cpp and tools built on it. Agaram is a Tamil–English model
**trained from scratch** by Murai Labs and Ramchand Kumaresan; it is not a fine-tune of any other model. Model card:
[base](https://huggingface.co/murai-labs/agaram-2.5b). Code: [Murai-Labs/Agaram2.5b](https://github.com/Murai-Labs/Agaram2.5b).
Paper: [arXiv:2610.18420](https://arxiv.org/abs/2610.18420).

This is a base model: it continues text and does not follow instructions or chat. **To chat, use
[murai-labs/agaram-2.5b-gguf](https://huggingface.co/murai-labs/agaram-2.5b-gguf)** (for example
`ollama run hf.co/murai-labs/agaram-2.5b-gguf:Q8_0`), or try the
[chat demo](https://huggingface.co/spaces/mechramc/agaram-2.5b-chat).

These files were built on 2026-10-12 with llama.cpp `b11574` and were in murai-labs/agaram-2.5b-gguf until
2026-10-29; the files and hashes are unchanged. File hashes are in `SHA256SUMS`.

| File | Size | Use |
|---|---|---|
| `agaram-2.5b-f16.gguf` | 5.06 GB | base, half precision |
| `agaram-2.5b-Q8_0.gguf` | 2.69 GB | base, 8-bit |
| `agaram-2.5b-Q4_K_M.gguf` | 1.56 GB | base, small 4-bit |

```
llama-completion -m agaram-2.5b-Q8_0.gguf -p "சென்னை ஒரு" -n 128 --temp 0
```

- The files stop on `<|endoftext|>` (id 0).
- Context: 8,192 tokens.

## Requirements: a llama.cpp that knows the `qwen35` pre-tokenizer

Agaram's tokenizer is a byte-level BPE whose pre-tokenizer is Qwen2's split pattern with Unicode combining marks and
the zero-width joiners added to the letter class, so that Tamil vowel signs stay attached to their consonants.
llama.cpp has no entry for this exact pattern. These files use llama.cpp's `qwen35` pre-tokenizer, whose pattern is the
same except that it does not include the zero-width joiners (U+200C, U+200D). Use a llama.cpp build that recognises
`qwen35` (b11574 or newer); older builds, and apps that bundle them, will refuse to load the files or tokenize
differently.

## What was checked

- **Tokenization (f16 file, 2026-10-12).** llama.cpp's token ids were compared with the Hugging Face tokenizer on 961
  texts: every prompt and reference answer of our three chat evaluation sets (661 texts in Tamil script, Tanglish and
  English) and 300 held-out training conversations rendered with the chat template. All 961 match exactly. The Q8_0
  and Q4_K_M base files were not tested separately. None of these texts contains a zero-width joiner, the one case
  where the patterns differ, so that case is untested.

## Model, evaluation, limitations and disclosures

See the [base model card](https://huggingface.co/murai-labs/agaram-2.5b).

## Creators

Murai Labs and Ramchand Kumaresan. Licence: Apache-2.0; upstream data terms in the base model card. Technical report: https://arxiv.org/abs/2610.18420
