import dataset
import torch
from .transformer_task import TransformerTask
from .task import TaskDataset


class DeepmindMathTask(TransformerTask):
    TRAIN_NUM_WORKERS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks, sets=[f"train_{s}" \
                                                     for s in self.helper.opt.dm_math.train_splits])

        self.valid_sets.interpolate = dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks, sets=["interpolate"])

        if len(self.helper.opt.dm_math.tasks)==1:
            self.valid_sets.iid = dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks, sets=[f"test_{s}" \
                                                         for s in self.helper.opt.dm_math.train_splits])
            self.valid_sets.hard = dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks, sets=["test_hard"])
        else:
            for task in self.helper.opt.dm_math.tasks:
                self.valid_sets[f"iid_{task}"] = dataset.DeepmindMathDataset([task], sets=[f"test_{s}" for s in
                                                                             self.helper.opt.dm_math.train_splits])
                self.valid_sets[f"hard_{task}"] = dataset.DeepmindMathDataset([task], sets=["test_hard"])


        extrapolate = dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks, sets=["extrapolate"])
        if len(extrapolate)!=0:
            self.valid_sets.extrapolate = extrapolate

        self.tasks.append(TaskDataset("hard",
                                      dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks, sets=[f"train_{s}" \
                                                     for s in self.helper.opt.dm_math.masks_splits]),
                                      dataset.DeepmindMathDataset(self.helper.opt.dm_math.tasks,
                                                     sets=[f"test_{s}" for s in self.helper.opt.dm_math.masks_splits])
                                      ))

    def create_optimizer(self):
        self.set_optimizer(torch.optim.Adam(self.model.model_parameters.values(), self.helper.opt.lr, eps=1e-9,
                                            betas=(0.9, 0.995)))

    def get_n_mask_samples(self):
        return 8