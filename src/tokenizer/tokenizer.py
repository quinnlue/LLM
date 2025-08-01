from tokenizers import Tokenizer

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

tokenizer = Tokenizer.from_file("src/tokenizer/tokenizer.json")

if __name__ == "__main__":
    print(len(tokenizer.get_vocab()))
    print(tokenizer.encode("Hello, world!").ids)