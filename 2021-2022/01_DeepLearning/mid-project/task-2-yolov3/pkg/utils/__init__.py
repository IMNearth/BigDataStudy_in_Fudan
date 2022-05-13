from .datasets import VocDataset
from .gpu import select_device
from .cosine_lr_scheduler import CosineDecayLR

from . import datasets
from . import data_augment

import os, sys
import logging


def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def clean_dir(the_dir, name_keys=["best", "last"]):
    file_names = os.listdir(the_dir)
    for fn in file_names:
        for key in name_keys:
            if key in fn:
                os.remove(os.path.join(the_dir, fn))

