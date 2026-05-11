import os
import math
import pickle
import time
import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


INPUT_FILE = "/Users/count_c/code/stanford-cs336/assignment1-basics/data/owt_train.txt"

VOCAB_SIZE = 32000

SPECIAL_TOKENS = ["<|endoftext|>"]

SAVE_PATH = "/Users/count_c/code/stanford-cs336/assignment1-basics/cs336_basics/bpe_vocab/owt"


def find_chunk_boundaries(
    file_path: str,
    split_token: bytes
) -> list[int]:
    
    assert isinstance(split_token, bytes), "Must represent special token as a bytestring"

    chunk_size = 500 * 1024**2

    with open(file_path, "rb") as file:

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()                                                                                           
        file.seek(0)

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_num = math.ceil(file_size / chunk_size)
        chunk_boundaries = [i * chunk_size for i in range(chunk_num + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(
    file_path: str,
    start: int,
    end: int,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    split_pat = "|".join(re.escape(token) for token in special_tokens)

    pretoken_cnt = defaultdict(int)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        for part in re.splititer(split_pat, chunk):
            for match in re.finditer(pat, part):
                word = match.group()
                word_encoded = list(word.encode("utf-8"))
                word_byte = [bytes([w]) for w in word_encoded]
                pretoken_cnt[tuple(word_byte)] += 1
               
    return pretoken_cnt


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = defaultdict(int)
    for i in range(256):
        vocab[i] = bytes([i])
    idx = len(vocab)
    for st in special_tokens:
        vocab[idx] = st.encode("utf-8")
        idx += 1
    return vocab


def merge_new_pretoken(
    old_pretoken: tuple[bytes, ...],
    merge_pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    
    new_pretoken = []
    new_token = merge_pair[0] + merge_pair[1]

    i = 0
    while i < len(old_pretoken) - 1:
        pair = (old_pretoken[i], old_pretoken[i+1])
        if pair != merge_pair:
            new_pretoken.append(old_pretoken[i])
            i += 1
        else:
            new_pretoken.append(new_token)
            i += 2

    if i == len(old_pretoken) - 1:
        new_pretoken.append(old_pretoken[i])

    return tuple(new_pretoken)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    start_time = time.time()
    
    special_tokens.sort(key=len, reverse=True)
    chunk_boundaries = find_chunk_boundaries(input_path, b'\n')

    pretoken_count = defaultdict(int)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(pretokenize, input_path, start, end, special_tokens) 
                   for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
        for future in as_completed(futures):
            result = future.result()
            for pretoken, cnt in result.items():
                pretoken_count[pretoken] += cnt

    pre_tokenize_time = time.time()
    print(f"pretokenize cost: {pre_tokenize_time - start_time} seconds")
            
    vocab = init_vocab(special_tokens)
    merges = []

    pair_count = defaultdict(int)
    pair_pretoken = defaultdict(set)
    for pretoken, cnt in pretoken_count.items():
        for i in range(1, len(pretoken)):
            pair = (pretoken[i-1], pretoken[i])
            pair_count[pair] += cnt
            pair_pretoken[pair].add(pretoken)

    idx = len(vocab)
    while idx < vocab_size:
        # print(idx, len(pair_count))
        max_cnt_pair = None
        max_cnt = -1
        for pair, cnt in pair_count.items():
            if cnt > max_cnt:
                max_cnt = cnt
                max_cnt_pair = pair
            elif cnt == max_cnt and max_cnt_pair is not None:
                if pair > max_cnt_pair:
                    max_cnt_pair = pair

        if max_cnt_pair is None or max_cnt < 0:
            break;
      
        new_token = max_cnt_pair[0] + max_cnt_pair[1]
        merges.append(max_cnt_pair)
        vocab[idx] = new_token
        idx += 1

        changed_pretokens = pair_pretoken[max_cnt_pair].copy()

        for old_pretoken in changed_pretokens:
            pretoken_cnt = pretoken_count[old_pretoken]

            # update pretoken_count
            pretoken_count.pop(old_pretoken)
            new_pretoken = merge_new_pretoken(old_pretoken, max_cnt_pair)
            pretoken_count[new_pretoken] += pretoken_cnt

            # a,b,c,d -> a,bc,d
            # update pair_count and pair_pretoken
            for i in range(1, len(old_pretoken)):
                pair = (old_pretoken[i-1], old_pretoken[i])
                pair_count[pair] -= pretoken_cnt
                pair_pretoken[pair].discard(old_pretoken)
                if pair_count[pair] <= 0:
                    pair_count.pop(pair)
                if len(pair_pretoken[pair]) == 0:
                    pair_pretoken.pop(pair)
            
            for i in range(1, len(new_pretoken)):
                pair = (new_pretoken[i-1], new_pretoken[i])
                pair_count[pair] += pretoken_cnt
                pair_pretoken[pair].add(new_pretoken)
                
    merge_time = time.time()
    print(f"merge cost: {merge_time - pre_tokenize_time} seconds")
    print(f"total cost: {merge_time - start_time} seconds")

    return vocab, merges

    
def save(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    save_path: str
):
    os.makedirs(save_path, exist_ok=True)

    vocab_save_path = os.path.join(save_path, "vocab.pkl")
    with open(vocab_save_path, "wb") as f:
        pickle.dump(vocab, f)
    
    merge_save_path = os.path.join(save_path, "merge.pkl")
    with open(merge_save_path, "wb") as f:
        pickle.dump(merges, f)
        

if __name__ == '__main__':
    vocab, merges = train_bpe(INPUT_FILE, VOCAB_SIZE, SPECIAL_TOKENS)

    save(vocab, merges, SAVE_PATH)
