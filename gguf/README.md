# GGUF conversion

How the Agaram 2.5B GGUF files were built and checked. The files and their cards are on Hugging Face:

- chat: [murai-labs/agaram-2.5b-gguf](https://huggingface.co/murai-labs/agaram-2.5b-gguf) (card: `../hf_cards/agaram-2.5b-gguf/README.md`)
- base: [murai-labs/agaram-2.5b-base-gguf](https://huggingface.co/murai-labs/agaram-2.5b-base-gguf) (card: `../hf_cards/agaram-2.5b-base-gguf/README.md`)

Those cards give the build dates, file hashes, usage (llama.cpp, Ollama) and the results of the checks.

| File | What it does |
|---|---|
| `convert_agaram.py` | Wraps llama.cpp's `convert_hf_to_gguf.py` and sets the pre-tokenizer to `qwen35` (see the docstring for the full recipe, including the `<\|im_end\|>` stop token for the chat model) |
| `gguf_parity.py` | Compares a GGUF's token ids with the Hugging Face tokenizer on a set of texts |

Built and checked with llama.cpp `b11574`.
