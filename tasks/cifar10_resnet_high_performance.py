from .cifar10_class_removal import Cifar10ClassRemovalTask
from models import ResNet110
import torch
import torch.utils.data
import dataset
from typing import List, Optional
import torchvision.transforms as transforms


class Cifar10ResnetHPClassRemovalTask(Cifar10ClassRemovalTask):
    TRAIN_NUM_WORKERS = 16
    DATASET = dataset.image.CIFAR10

    def create_model(self):
        torch.backends.cudnn.benchmark = True
        self.enable_grad_clip = False
        return ResNet110(self.train_set.in_channels(), self.train_set.out_channels())

    # def get_n_mask_samples(self):
    #     return 8
    #
    def create_normalizer(self, scale: float = 1):
        s = lambda *args: tuple(a*scale for a in args)
        return transforms.Normalize(s(0.4914, 0.4822, 0.4465), s(0.2023, 0.1994, 0.2010))

    def create_restricted_train_set(self, restrict: Optional[List[int]]) -> torch.utils.data.Dataset:
        return self.DATASET("train", valid_split_size=0, normalize=False, augment= transforms.Compose([
            dataset.image.convert.np_to_pil,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.create_normalizer()
        ]), restrict=restrict)

    def create_datasets(self):
        self.batch_dim = 0
        self.train_set = self.create_restricted_train_set(None)
        self.valid_sets.iid = self.DATASET("test",  normalize=False, augment=transforms.Compose([
            dataset.image.convert.np_to_tensor,
            self.create_normalizer(255)
        ]))
        self.mask_classes = list(range(10))

    def prepare_model_for_analysis(self):
        super().prepare_model_for_analysis()
        self.enable_grad_clip = True

    def clip_gradients(self):
        # Do not clip gradients in the initial training phase
        if self.enable_grad_clip:
            super().clip_gradients()
