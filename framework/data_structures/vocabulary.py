import re
from typing import List, Union, Optional, Dict, Any


class WordVocabulary:
    def __init__(self, list_of_sentences: Optional[List[Union[str, List[str]]]] = None, allow_any_word: bool = False):
        next_id = 0

        def add_word(w: str):
            nonlocal next_id
            self.words[w] = next_id
            self.inv_words[next_id] = w
            next_id += 1

        self.words: Dict[str, int] = {}
        self.inv_words: Dict[int, str] = {}
        self.to_save = ["words", "inv_words", "_unk_index"]
        self.allow_any_word = allow_any_word
        self.initialized = False

        if list_of_sentences is not None:
            self.initialized = True

            words = {}
            for s in list_of_sentences:
                for w in self.split_sentence(s):
                    words[w] = words.get(w, 0) + 1

            for w in self.words.keys():
                if w in words:
                    words.remove(w)

            for w in sorted(words.keys(), key=words.get, reverse=True):
                add_word(w)

        self._unk_index = self.words.get("<UNK>", self.words.get("<unk>"))
        if allow_any_word:
            assert self._unk_index is not None

    def _process_word(self, w: str) -> int:
        res = self.words.get(w)

        if res is None:
            if not self.allow_any_word:
                assert False, "WARNING: unknown word: '%s'" % w
            res = self._unk_index

        return res

    def __getitem__(self, item: Union[int, str]) -> Union[str, int]:
        if isinstance(item, int):
            return self.inv_words.get(item, "<!INV: %d!>" % item)
        else:
            return self.words.get(item, self._unk_index)

    @staticmethod
    def split_sentence(sentence: Union[str, List[str]]) -> List[str]:
        if isinstance(sentence, list):
            # Already tokenized.
            return sentence

        return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)

    def sentence_to_indices(self, sentence: str) -> List[int]:
        assert self.initialized
        words = self.split_sentence(sentence)
        return [self._process_word(w) for w in words]

    def indices_to_sentence(self, indices: List[int]) -> List[str]:
        assert self.initialized
        return [self[i] for i in indices]

    def __call__(self, seq: Union[List[Union[str, int]], str]) -> List[Union[int, str]]:
        if seq is None or (isinstance(seq, list) and not seq):
            return seq

        if isinstance(seq, str):
            seq = seq.split(" ")

        if isinstance(seq[0], str):
            return self.sentence_to_indices(seq)
        else:
            return self.indices_to_sentence(seq)

    def __len__(self) -> int:
        return len(self.words)

    def state_dict(self) -> Dict[str, Any]:
        return {
            k: self.__dict__[k] for k in self.to_save
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.initialized = True
        self.__dict__.update(state)

