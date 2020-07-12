import torch
import torch.utils.data
import os
import numpy as np
from framework.utils import download
from framework.data_structures import WordVocabulary
from typing import List, Dict, Any, Tuple


class ScanTestState:
    def __init__(self, batch_dim: int = 1):
        self.n_ok = 0
        self.n_total = 0
        self.batch_dim = batch_dim
        self.time_dim = 1 - self.batch_dim

    def step(self, net_out: Tuple[torch.Tensor, torch.Tensor], data: Dict[str, torch.Tensor]):
        out, len = net_out
        ref = data["out"]

        if out.shape[0] > ref.shape[0]:
            out = out[: ref.shape[0]]
        elif out.shape[0] < ref.shape[0]:
            ref = ref[: out.shape[0]]

        unused = torch.arange(0, out.shape[0], dtype=torch.long, device=ref.device).unsqueeze(self.batch_dim) >=\
                 data["out_len"].unsqueeze(self.time_dim)

        ok_mask = ((out == ref) | unused).all(self.time_dim) & (len == data["out_len"])

        self.n_total += ok_mask.nelement()
        self.n_ok += ok_mask.long().sum().item()

    @property
    def accuracy(self):
        return self.n_ok / self.n_total

    def plot(self) -> Dict[str, Any]:
        return {"accuracy/total": self.accuracy}


class Scan(torch.utils.data.Dataset):
    in_sentences = []
    out_sentences = []
    index_table = {}


    URLS = {
        "simple": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt"
        },
        "length": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt"
        },
        "jump": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt"
        },
        "turn_left": {
            "train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_turn_left.txt",
            "test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_turn_left.txt"
        },
    }

    def _load_dataset(self, cache_dir: str):
        if Scan.in_sentences:
            return

        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "scan.pth")

        if not os.path.isfile(cache_file):
            for split_type, split in self.URLS.items():
                if split_type not in Scan.index_table.items():
                    Scan.index_table[split_type] = {}

                for set, url in split.items():
                    fn = os.path.join(cache_dir, os.path.split(url)[-1])

                    print("Downloading", url)
                    download(url, fn, ignore_if_exists=True)

                    this_set = Scan.index_table[split_type].get(set)
                    if this_set is None:
                        this_set = []
                        Scan.index_table[split_type][set] = this_set

                    with open(fn) as f:
                        for line in f:
                            line = line.split("OUT:")
                            line[0] = line[0].replace("IN:", "")
                            line = [l.strip() for l in line]

                            Scan.in_sentences.append(line[0])
                            Scan.out_sentences.append(line[1])

                            this_set.append(len(Scan.in_sentences) - 1)

            print("Constructing vocabularies")
            Scan.in_vocabulary = WordVocabulary(self.in_sentences)
            Scan.out_vocabulary = WordVocabulary(self.out_sentences)

            Scan.in_sentences = [Scan.in_vocabulary(s) for s in Scan.in_sentences]
            Scan.out_sentences = [Scan.out_vocabulary(s) for s in Scan.out_sentences]

            Scan.max_in_len = max(len(l) for l in Scan.in_sentences)
            Scan.max_out_len = max(len(l) for l in Scan.out_sentences)

            print("Done.")
            torch.save({
                "index": Scan.index_table,
                "in_sentences": Scan.in_sentences,
                "out_sentences": Scan.out_sentences,
                "in_voc": Scan.in_vocabulary.state_dict(),
                "out_voc": Scan.out_vocabulary.state_dict(),
                "max_in_len": Scan.max_in_len,
                "max_out_len": Scan.max_out_len
            }, cache_file)
        else:
            data = torch.load(cache_file)
            Scan.index_table = data["index"]
            Scan.in_vocabulary = WordVocabulary(None)
            Scan.out_vocabulary = WordVocabulary(None)
            Scan.in_vocabulary.load_state_dict(data["in_voc"])
            Scan.out_vocabulary.load_state_dict(data["out_voc"])
            Scan.in_sentences = data["in_sentences"]
            Scan.out_sentences = data["out_sentences"]
            Scan.max_in_len = data["max_in_len"]
            Scan.max_out_len = data["max_out_len"]

        for k, t in self.index_table.items():
            print(f"Scan: split {k} data:", ", ".join([f"{k}: {len(v)}" for k, v in t.items()]))

    def __init__(self, sets: List[str]=["train"], split_type: List[str]=["simple"], cache_dir: str="./cache/scan"):
        super().__init__()
        self._load_dataset(cache_dir)

        assert isinstance(sets, List)
        assert isinstance(split_type, List)

        self.my_indices = []
        for t in split_type:
            for s in sets:
                self.my_indices += Scan.index_table[t][s]

    def __len__(self) -> int:
        return len(self.my_indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.my_indices[item]
        in_seq = Scan.in_sentences[index]
        out_seq = Scan.out_sentences[index]

        return {
            "in": np.asarray(in_seq, np.int16),
            "out": np.asarray(out_seq, np.int16),
            "in_len": len(in_seq),
            "out_len": len(out_seq)
        }

    def get_output_size(self):
        return len(self.out_vocabulary)

    def get_input_size(self):
        return len(self.in_vocabulary)

    def start_test(self) -> ScanTestState:
        return ScanTestState()
