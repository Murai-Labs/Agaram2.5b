"""Convert Agaram 2.5B (HF safetensors) to GGUF with llama.cpp's converter.

llama.cpp does not recognise Agaram's pre-tokenizer hash (Qwen2's split pattern with combining marks and ZWJ/ZWNJ in the
letter class), so this wrapper tells the converter to use "qwen35", whose pattern differs only in leaving out ZWJ/ZWNJ.
Token parity against the HF tokenizer was 961/961 on our test texts (gguf_parity.py). After converting the instruct
model, set its stop token to <|im_end|> (id 2):

    LLAMA_CPP=~/llama.cpp python convert_agaram.py agaram-2.5b-instruct/ --outtype f16 --outfile agaram-2.5b-instruct-f16.gguf
    python $LLAMA_CPP/gguf-py/gguf/scripts/gguf_set_metadata.py --force agaram-2.5b-instruct-f16.gguf tokenizer.ggml.eos_token_id 2
    $LLAMA_CPP/build/bin/llama-quantize agaram-2.5b-instruct-f16.gguf agaram-2.5b-instruct-Q8_0.gguf Q8_0

Built and checked with llama.cpp b11574.
"""
import os
import runpy
import sys

LLAMA_CPP = os.path.expanduser(os.environ.get("LLAMA_CPP", "~/llama.cpp"))
sys.path.insert(0, LLAMA_CPP)
sys.argv = ["convert_hf_to_gguf.py"] + sys.argv[1:]
from conversion import base  # noqa: E402

base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
runpy.run_path(os.path.join(LLAMA_CPP, "convert_hf_to_gguf.py"), run_name="__main__")
