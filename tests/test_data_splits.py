import os

import pytest

from commerge.data.calibration import class_balanced_indices
from commerge.data.vision import NUM_CLASSES, VISION_TASKS, load_vision_task

DATA_ROOT = os.environ.get('COMMERGE_DATA', 'data')

SIZES = {
    'stanford_cars': (8144, 1608, 6433),
    'dtd': (1880, 1880, 1880),
    'eurosat': (21600, 2700, 2700),
    'gtsrb': (26640, 2526, 10104),
    'mnist': (60000, 2000, 8000),
    'resisc45': (18900, 6300, 6300),
    'sun397': (19850, 3970, 15880),
    'svhn': (73257, 5206, 20826),
}
SHUFFLED = ['stanford_cars', 'gtsrb', 'mnist', 'sun397', 'svhn']

needs_data = pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason='vision data folder missing')


@needs_data
@pytest.mark.parametrize('task', VISION_TASKS)
def test_sizes_and_labels(task):
    td = load_vision_task(task, DATA_ROOT, lambda im: im)
    assert td.num_classes == NUM_CLASSES[task] == len(td.class_names)
    for split, n in zip(('train', 'val', 'test'), SIZES[task]):
        assert len(td.split(split)) == n
        labels = td.labels(split)
        assert len(labels) == n and 0 <= min(labels) and max(labels) < td.num_classes


@needs_data
@pytest.mark.parametrize('task', SHUFFLED)
def test_val_test_disjoint(task):
    td = load_vision_task(task, DATA_ROOT, lambda im: im)
    val, test = set(td.val.indices), set(td.test.indices)
    assert not val & test
    assert len(val) + len(test) == len(td.val.dataset)


def test_class_balanced_indices_synthetic():
    labels = [0] * 50 + [1] * 30 + [2] * 20
    idx = class_balanced_indices(labels, 30, seed=1)
    assert len(idx) == len(set(idx)) == 30
    assert sorted(sum(labels[i] == c for i in idx) for c in range(3)) == [10, 10, 10]
    assert idx == class_balanced_indices(labels, 30, seed=1)
    small = [0] * 10 + [1] * 2
    idx = class_balanced_indices(small, 8, seed=0)
    assert len(idx) == len(set(idx)) == 8 and sum(small[i] == 1 for i in idx) == 2


@needs_data
def test_cars_train_is_not_test():
    td = load_vision_task('stanford_cars', DATA_ROOT, lambda im: im)
    train = {p for p, _ in td.split('train').samples}
    test = {p for p, _ in td.split('test').dataset.samples}
    assert not train & test
