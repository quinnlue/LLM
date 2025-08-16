from tokenizers import Tokenizer

import os

tokenizer = Tokenizer.from_file("gpt1/tokenizer/tokenizer.json")

if __name__ == "__main__":
    print(len(tokenizer.get_vocab()))
    print(tokenizer.encode("Hello, world!").ids)