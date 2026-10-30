import csv
import os
import re
from typing import Callable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.datasets.folder import make_dataset

from .base import TaskData

VISION_TASKS = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
NUM_CLASSES = {'stanford_cars': 196, 'dtd': 47, 'eurosat': 10, 'gtsrb': 43, 'mnist': 10,
               'resisc45': 45, 'sun397': 397, 'svhn': 10}
VAL_FRACTION = 0.2
ASSETS = os.path.join(os.path.dirname(__file__), 'assets')

GTSRB_CLASSES = [
        'red and white circle 20 kph speed limit',
        'red and white circle 30 kph speed limit',
        'red and white circle 50 kph speed limit',
        'red and white circle 60 kph speed limit',
        'red and white circle 70 kph speed limit',
        'red and white circle 80 kph speed limit',
        'end / de-restriction of 80 kph speed limit',
        'red and white circle 100 kph speed limit',
        'red and white circle 120 kph speed limit',
        'red and white circle red car and black car no passing',
        'red and white circle red truck and black car no passing',
        'red and white triangle road intersection warning',
        'white and yellow diamond priority road',
        'red and white upside down triangle yield right-of-way',
        'stop',
        'empty red and white circle',
        'red and white circle no truck entry',
        'red circle with white horizonal stripe no entry',
        'red and white triangle with exclamation mark warning',
        'red and white triangle with black left curve approaching warning',
        'red and white triangle with black right curve approaching warning',
        'red and white triangle with black double curve approaching warning',
        'red and white triangle rough / bumpy road warning',
        'red and white triangle car skidding / slipping warning',
        'red and white triangle with merging / narrow lanes warning',
        'red and white triangle with person digging / construction / road work warning',
        'red and white triangle with traffic light approaching warning',
        'red and white triangle with person walking warning',
        'red and white triangle with child and person walking warning',
        'red and white triangle with bicyle warning',
        'red and white triangle with snowflake / ice warning',
        'red and white triangle with deer warning',
        'white circle with gray strike bar no speed limit',
        'blue circle with white right turn arrow mandatory',
        'blue circle with white left turn arrow mandatory',
        'blue circle with white forward arrow mandatory',
        'blue circle with white forward or right turn arrow mandatory',
        'blue circle with white forward or left turn arrow mandatory',
        'blue circle with white keep right arrow mandatory',
        'blue circle with white keep left arrow mandatory',
        'blue circle with white arrows indicating a traffic circle',
        'white circle with gray strike bar indicating no passing for cars has ended',
        'white circle with gray strike bar indicating no passing for trucks has ended',
]

EUROSAT_TO_CLIP = {
    'annual crop': 'annual crop land', 'forest': 'forest', 'herbaceous vegetation': 'brushland or shrubland',
    'highway': 'highway or road', 'industrial area': 'industrial buildings or commercial buildings',
    'pasture': 'pasture land', 'permanent crop': 'permanent crop land',
    'residential area': 'residential buildings or homes or apartments', 'river': 'river',
    'sea lake': 'lake or sea',
}


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Vision data not found: expected {path} (downloads are disabled)')
    return path


class PathLabelDataset(Dataset):

    def __init__(self, samples: Sequence, preprocess: Callable):
        self.samples, self.preprocess = list(samples), preprocess
        self.targets = [int(t) for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        return self.preprocess(Image.open(path).convert('RGB')), target


def _folder(path: str, preprocess) -> datasets.ImageFolder:
    return datasets.ImageFolder(_require(path), transform=preprocess)


def _folder_names(ds: datasets.ImageFolder, fn=lambda c: c.replace('_', ' ')) -> List[str]:
    return [fn(c) for c in ds.classes]


def _split_test(test_ds: Dataset, test_targets: Sequence[int], task: str):
    perm = torch.load(os.path.join(ASSETS, f'{task}_shuffled_idxs.pt'), weights_only=False)
    perm = [int(i) for i in np.asarray(perm)]
    assert len(perm) == len(test_ds), (task, len(perm), len(test_ds))
    n_val = int(len(test_ds) * VAL_FRACTION)
    val_idx, test_idx = perm[:n_val], perm[n_val:]
    t = np.asarray(test_targets)
    return (Subset(test_ds, val_idx), Subset(test_ds, test_idx),
            t[val_idx].tolist(), t[test_idx].tolist())


def _digits(): return [str(i) for i in range(10)]


def _cars(root, pre):
    import scipy.io as sio
    d = os.path.join(root, 'stanford_cars')
    meta = sio.loadmat(_require(os.path.join(d, 'devkit', 'cars_meta.mat')), squeeze_me=True)

    def split(annos, images):
        mat = sio.loadmat(_require(os.path.join(d, annos)), squeeze_me=True)
        img_dir = _require(os.path.join(d, images))
        return PathLabelDataset([(os.path.join(img_dir, a['fname']), int(a['class']) - 1) for a in mat['annotations']], pre)
    train = split(os.path.join('devkit', 'cars_train_annos.mat'), 'cars_train')
    test = split('cars_test_annos_withlabels.mat', 'cars_test')
    names = [c.replace('_', ' ') for c in meta['class_names'].tolist()]
    val, tst, lv, lt = _split_test(test, test.targets, 'cars')
    return train, val, tst, dict(train=train.targets, val=lv, test=lt), names


def _dtd(root, pre):
    tr, va, te = (_folder(os.path.join(root, 'dtd', s), pre) for s in ('train', 'val', 'test'))
    return tr, va, te, dict(train=tr.targets, val=va.targets, test=te.targets), _folder_names(va)


def _eurosat(root, pre):
    tr, va, te = (_folder(os.path.join(root, 'EuroSAT_splits', s), pre) for s in ('train', 'val', 'test'))

    def pretty(c):
        out = ' '.join(w.lower() for w in re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', c))
        return out + ' area' if out.endswith('al') else out
    names = [EUROSAT_TO_CLIP[pretty(c)] for c in va.classes]
    return tr, va, te, dict(train=tr.targets, val=va.targets, test=te.targets), names


def _gtsrb(root, pre):
    base = os.path.join(root, 'gtsrb', 'gtsrb')
    train_dir = _require(os.path.join(base, 'GTSRB', 'Training'))
    test_dir = _require(os.path.join(base, 'GTSRB', 'Final_Test', 'Images'))
    classes = sorted(e.name for e in os.scandir(train_dir) if e.is_dir())
    train = PathLabelDataset(make_dataset(train_dir, extensions=('.ppm',),
                                          class_to_idx={c: i for i, c in enumerate(classes)}), pre)
    with open(_require(os.path.join(base, 'GT-final_test.csv'))) as f:
        rows = csv.DictReader(f, delimiter=';', skipinitialspace=True)
        test = PathLabelDataset([(os.path.join(test_dir, r['Filename']), int(r['ClassId'])) for r in rows], pre)
    val, tst, lv, lt = _split_test(test, test.targets, 'gtsrb')
    return train, val, tst, dict(train=train.targets, val=lv, test=lt), GTSRB_CLASSES


def _mnist(root, pre):
    _require(os.path.join(root, 'MNIST', 'raw'))
    tr = datasets.MNIST(root, train=True, download=False, transform=pre)
    te = datasets.MNIST(root, train=False, download=False, transform=pre)
    val, tst, lv, lt = _split_test(te, te.targets.tolist(), 'mnist')
    return tr, val, tst, dict(train=tr.targets.tolist(), val=lv, test=lt), _digits()


def _resisc45(root, pre):
    lists = os.path.join(root, 'resisc45')

    def make(split):
        with open(_require(os.path.join(lists, f'resisc45-{split}.txt'))) as f:
            keep = {line.strip() for line in f}
        return datasets.ImageFolder(_require(os.path.join(lists, 'NWPU-RESISC45')), transform=pre,
                                    is_valid_file=lambda p: os.path.basename(p) in keep)
    tr, va, te = make('train'), make('val'), make('test')
    return tr, va, te, dict(train=tr.targets, val=va.targets, test=te.targets), _folder_names(tr)


def _sun397(root, pre):
    tr = _folder(os.path.join(root, 'SUN397', 'train'), pre)
    te = _folder(os.path.join(root, 'SUN397', 'test'), pre)
    val, tst, lv, lt = _split_test(te, te.targets, 'sun397')
    names = [c[2:].replace('_', ' ') for c in tr.classes]
    return tr, val, tst, dict(train=tr.targets, val=lv, test=lt), names


def _svhn(root, pre):
    d = _require(os.path.join(root, 'svhn'))
    tr = datasets.SVHN(d, split='train', download=False, transform=pre)
    te = datasets.SVHN(d, split='test', download=False, transform=pre)
    val, tst, lv, lt = _split_test(te, te.labels.tolist(), 'svhn')
    return tr, val, tst, dict(train=tr.labels.tolist(), val=lv, test=lt), _digits()


_BUILDERS = dict(stanford_cars=_cars, dtd=_dtd, eurosat=_eurosat, gtsrb=_gtsrb, mnist=_mnist,
                 resisc45=_resisc45, sun397=_sun397, svhn=_svhn)


def load_vision_task(name: str, root: str, preprocess) -> TaskData:
    if name not in _BUILDERS:
        raise ValueError(f'unknown vision task {name!r}; choose from {VISION_TASKS}')
    train, val, test, labels, class_names = _BUILDERS[name](root, preprocess)
    assert len(class_names) == NUM_CLASSES[name], (name, len(class_names))
    return TaskData(name=name, num_classes=NUM_CLASSES[name], class_names=list(class_names),
                    train=train, val=val, test=test, labels_fn=lambda split: labels[split])
