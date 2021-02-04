from .task import TaskDataset
import dataset
from .transformer_task import TransformerTask
from models import TransformerEncDecModel
from layers.transformer import Transformer
import torch
from typing import Dict, Any, List, Set
from masked_model import Masks


class TransformerScanTask(TransformerTask):
    ANALYZE_TRAIN_SET = False

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.Scan(["train"], split_type=self.helper.opt.scan.train_split)
        self.valid_sets.iid = dataset.Scan(["test"])

        for s in self.helper.opt.scan.analyze_splits:
            self.valid_sets[s] = dataset.Scan(["test"], split_type=[s])

        for split in ["simple"] + self.helper.opt.scan.analyze_splits:
            self.tasks.append(TaskDataset(split, (lambda split_c: lambda: dataset.Scan(["train"],
                                          split_type=[split_c]))(split), self.valid_sets.iid))

    def post_train(self):
        super().post_train()
        self.helper.summary.log(self.plot_mask_sharing(range(1, len(self.model.masks))))

    def plot_stage_results(self, index: int, name: str) -> Dict[str, Any]:
        res = {f"analysis_results/{name}/{k}": v for k, v in self.validate().items()}
        if index==0:
            res.update(self.do_half_mask_test(index, name))
        return res

    def get_half_mask_masked_layer_names(self, masks: Masks) -> List[Set[str]]:
        res = set()
        size = {"encoder": self.helper.opt.transformer.encoder_n_layers,
                "decoder": self.helper.opt.transformer.decoder_n_layers}

        for k in masks.keys():
            k2 = k.split("_")
            if k2[2] == "layers":
                if int(k2[3]) <= size[k2[1]] // 2:
                    res.add(k)
            elif k.startswith("output_map_"):
                res.add(k)
            else:
                assert False, f"Unknown layer: {k}"

        return [res]

    def create_model(self) -> torch.nn.Module:
        return TransformerEncDecModel(len(self.train_set.in_vocabulary),
                                      len(self.train_set.out_vocabulary), self.helper.opt.state_size,
                                      nhead=self.helper.opt.transformer.n_heads,
                                      num_encoder_layers=self.helper.opt.transformer.encoder_n_layers,
                                      num_decoder_layers=self.helper.opt.transformer.decoder_n_layers,
                                      ff_multipiler=self.helper.opt.transformer.ff_multiplier,
                                      transformer=Transformer)
