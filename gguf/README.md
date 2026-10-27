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
Code: [Murai-Labs/Agaram2.5b](https://github.com/Murai-Labs/Agaram2.5b).

Built with llama.cpp `b11574` on 2026-10-12 from the released safetensors. Hashes are in `SHA256SUMS`.

| File | Size | Use |
|---|---|---|
| `agaram-2.5b-instruct-f16.gguf` | 5.06 GB | chat, full precision (reproduces the HF model's greedy replies) |
| `agaram-2.5b-instruct-Q8_0.gguf` | 2.69 GB | chat, near-lossless |
| `agaram-2.5b-instruct-Q4_K_M.gguf` | 1.56 GB | chat, small; answers drift from the full model (see below) |
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

- **Tokenization.** llama.cpp's token ids were compared with the Hugging Face tokenizer on 961 texts: every prompt and
  reference answer of our three chat evaluation sets (661 texts in Tamil script, Tanglish and English) and 300
  held-out training conversations rendered with the chat template (special tokens, tool calls, code, JSON). All 961
  match exactly, for the f16 and Q4_K_M chat files and the f16 base file. None of these texts contains a zero-width
  joiner, the one case where the patterns differ.
- **Replies.** Greedy replies to 8 held-out Tanglish prompts, compared with the HF model's greedy replies: the f16 file
  matches character for character on 7 of 8 (the 8th diverges after 114 characters); Q8_0 matches on 5 of 8; Q4_K_M on
  1 of 8, usually diverging within the first sentence. Prefer Q8_0 or f16 if you can.
- **Chat template.** `--jinja` with the embedded template produces the same prompt as the HF template; the replies to
  two test prompts were identical to the HF model's.

## Model, evaluation, limitations and disclosures

See the [chat model card](https://huggingface.co/murai-labs/agaram-2.5b-instruct). In short: strong at Tamil–English
translation for its size, weak on knowledge (about half of everyday questions answered acceptably), and it is a
research model, not a crisis service.

## Creators

Murai Labs and Ramchand Kumaresan. Licence: Apache-2.0.
