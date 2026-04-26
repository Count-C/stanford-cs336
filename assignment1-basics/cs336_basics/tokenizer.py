import pickle
import regex as re
from typing import Self, Iterable, Iterator
from tqdm import tqdm


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.decode_vocab = vocab
        self.merges = merges
        self.spcial_tokens = special_tokens

        self.encode_vocab = {v: k for k, v in vocab.items()}

        pretokenize_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretokenize_regex = re.compile(pretokenize_pat)

        self.special_regex = None
        self.special_tokens_set = set()
        if self.spcial_tokens:
            special_pat = "|".join(re.escape(token) for token in self.spcial_tokens)
            self.special_regex = re.compile(f"({special_pat})")
            self.special_tokens_set = set(self.spcial_tokens)


    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> Self:
        
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
                
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        tokens = []
        pretokens = self.__pretokenize(text)
        for pretoken in tqdm(pretokens, desc="Tokenizing"):
            if pretoken in self.special_tokens_set:
                tokens.append(self.encode_vocab[pretoken])
            else:
                merged_pretoken = self.__apply_merge(pretoken)
                for m in merged_pretoken:
                    tokens.append(self.encode_vocab[m])
        return tokens


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
            
        
    def decode(self, ids: list[int]) -> str:
        text_bytes_list = [self.decode_vocab[id] for id in ids]
        return b"".join(text_bytes_list).decode("utf-8")


    def __pretokenize(self, text: str) ->list[bytes]:

        if not self.spcial_tokens:
            words = self.pretokenize_regex.findall(text)
            pretoken = [word.encode("utf-8") for word in words]
        
        pretoken = []
        parts = self.special_regex.split(text)
        for part in parts:
            if part in self.special_tokens_set:
                pretoken.append(part.encode("utf-8"))
            else:
                words = self.pretokenize_regex.findall(part)
                pretoken.extend([word.encode("utf-8") for word in words])
        
        return pretoken
        

    def __apply_merge(self, pretoken: bytes) -> list[bytes]:
        merged_pretoken = [bytes([p]) for p in list(pretoken)]
        for merge in self.merges:
            if len(merged_pretoken) < 2:
                break
            i = 0
            while i < len(merged_pretoken) - 1:
                pair = tuple(merged_pretoken[i:i+2])
                if pair == merge:
                    merged_pretoken[i] = pair[0] + pair[1]
                    merged_pretoken.pop(i+1)
                i += 1
        return merged_pretoken
            