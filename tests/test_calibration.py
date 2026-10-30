import random
from collections import Counter

import torch

from commerge.data.base import TaskData
from commerge.data.calibration import calibration_dataset, class_balanced_indices, few_class_indices, random_indices
from commerge.models.llama import LlamaArch


def toy_task(n_train=40, n_val=12, classes=4):
    train = [(i, i % classes) for i in range(n_train)]
    val = [(100 + i, i % classes) for i in range(n_val)]
    return TaskData("toy", classes, [str(c) for c in range(classes)], train, val, val,
                    lambda split: [y for _, y in (train if split == "train" else val)])


def items(ds):
    return [ds[i][0] for i in range(len(ds))]


def test_balanced_and_random_sampling():
    labels = [i % 4 for i in range(40)]
    idx = class_balanced_indices(labels, 8, seed=1)
    assert Counter(labels[i] for i in idx) == {0: 2, 1: 2, 2: 2, 3: 2}
    assert random_indices(labels, 8, seed=1) == random.Random(1).sample(range(40), 8)
    assert random_indices(labels, 100, seed=1) == random.Random(1).sample(range(40), 40)


def test_top_up_completes_a_short_split_from_another_one():
    task = toy_task()
    cal = {"split": "val", "num_samples": 20, "seed": 3, "sampling": "balanced", "top_up": "train"}
    got = items(calibration_dataset(task, cal))
    assert len(got) == 20 and sorted(got[:12]) == list(range(100, 112)) and all(x < 100 for x in got[12:])
    assert items(calibration_dataset(task, {**cal, "num_samples": 6})) == [100 + i for i in class_balanced_indices(task.labels("val"), 6, 3)]


def test_few_classes_keep_at_least_one_class():
    labels = [i % 10 for i in range(100)]
    idx = few_class_indices(labels, range(10), 20, 0.03, random.Random(0))
    assert len({labels[i] for i in idx}) == 1 and len(idx) == 10


def test_llama_noise_restarts_from_its_seed_at_every_merge():
    arch = LlamaArch(noise_std=0.5, noise_seed=7)
    like = torch.zeros(2, 3)
    first = arch._noise(like)
    assert not torch.equal(first, arch._noise(like))
    arch.reset_noise()
    assert torch.equal(first, arch._noise(like))
    assert torch.allclose(first.std(), torch.tensor(0.5), atol=0.5)
