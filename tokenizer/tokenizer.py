from tokenizers import Tokenizer

import os
from pathlib import Path

# Construct path to tokenizer.json relative to this file to avoid cwd issues
_tokenizer_json_path = Path(__file__).resolve().with_name("tokenizer.json")

tokenizer = Tokenizer.from_file(str(_tokenizer_json_path))

if __name__ == "__main__":
    print(len(tokenizer.get_vocab()))
    print(tokenizer.encode("Hello, world!").ids)