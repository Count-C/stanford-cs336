import os
from cs336_basics.tokenizer import Tokenizer


def main():
    vocab_file = "/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/bpe_vocab/TinyStoriesV2/vocab.pkl"
    merge_file = "/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/bpe_vocab/TinyStoriesV2/merge.pkl"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_file, merge_file, special_tokens)

    print("------ simple test -----")
    text = """the, and, the, the"""
    text_encode = tokenizer.encode(text)
    text_decode = tokenizer.decode(text_encode)
    print(f"text: {text}")
    print(f"text_encode: {text_encode}")
    print(f"text_decode: {text_decode}")
    print(f"ration: {len(text.encode("utf-8")) / len(text_encode)}")


    print("------ file test -----")
    data_path = "/Users/count_c/code/stanford-cs336/assignment1-basics/data"
    valid_file = os.path.join(data_path, "TinyStoriesV2-GPT4-valid.txt")
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    valid_encode = tokenizer.encode(valid_text)
    valid_decode = tokenizer.decode(valid_encode)
    decode_file = os.path.join(data_path, "TinyStoriesV2-GPT4-valid-decode.txt")
    with open(decode_file, "w", encoding="utf-8") as f:
        f.write(valid_decode)
    print(f"orign_file: {valid_file}")
    print(f"decode_file: {decode_file}")
    print(f"ratio: {len(valid_text.encode("utf-8")) / len(valid_encode)}")



if __name__ == '__main__':
    main()
    
    