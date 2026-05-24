import os
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def main():
    vocab_file = "/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/bpe_vocab/TinyStoriesV2/vocab.pkl"
    merge_file = "/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/bpe_vocab/TinyStoriesV2/merge.pkl"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_file, merge_file, special_tokens)

    data_path = "/Users/count_c/code/stanford-cs336/assignment1-basics/data"

    train_file = os.path.join(data_path, "TinyStoriesV2-GPT4-train.txt")
    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    train_encode = tokenizer.encode(train_text)
    train_ids = np.array(train_encode, dtype=np.uint16)
    train_ids.tofile("/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/dataset/TinyStoriesV2_train.bin")

    valid_file = os.path.join(data_path, "TinyStoriesV2-GPT4-valid.txt")
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    valid_encode = tokenizer.encode(valid_text)
    valid_ids = np.array(valid_encode, dtype=np.uint16)
    valid_ids.tofile("/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/dataset/TinyStoriesV2_valid.bin")


if __name__ == '__main__':
    main()
    
    