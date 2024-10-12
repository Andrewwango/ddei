from pathlib import Path, WindowsPath
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

from torch import Generator, randperm
from torch.utils.data import Subset
import deepinv as dinv

@dataclass
class Trainer(dinv.Trainer):
    def get_samples_offline(self, iterators, g):
        r"""
        Get the samples for the offline measurements.

        Overrides default method by loading mask from dataloaders and updating physics.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a dictionary containing: the ground truth, the measurement, and the current physics operator.
        """
        x, y, mask = next(iterators[g])
        physics = self.physics[g]
        physics.update_parameters(mask=mask.to(self.device))
        return x.to(self.device), y.to(self.device), physics
    
def patient_random_split(dataset, train_frac=0.8, sax_slices_per_vol=-1, lax_slices_per_vol=3, generator=Generator()):
    """Random split dataset by patients

    :param torch.utils.data.Dataset dataset: full dataset
    :param float train_frac: train fraction, defaults to 0.8
    :param int sax_slices_per_vol: n. short axis slices per volume (per patient). Pass -1 to always pick only the first slice per volume.
    :param int lax_slices_per_vol: n. long axis slices per volume (per patient), defaults to 3 (i.e. all lax views)
    :param _type_ generator: torch random number generator, defaults to Generator()
    """
    path_pid_index = Path(dataset.examples[0].fname).parts.index("P001")

    # Get patient dictionary
    patients = defaultdict(lambda: defaultdict(list))

    for i, sample in enumerate(dataset.examples):
        path_parts = Path(sample.fname).parts
        patients[path_parts[path_pid_index]][path_parts[path_pid_index+1]].append(i)

    # Split patients into test and train
    def randsplit(arr, n, gen=generator):
        idx = randperm(len(arr), generator=gen).tolist()
        return [arr[i] for i in sorted(idx[:n])], [arr[i] for i in sorted(idx[n:])]

    pids = list((patients.keys()))

    pids_train, pids_test = randsplit(pids, int(round(len(pids) * train_frac)))

    # Get associated slice ids
    def choose_slices(lax, sax):
        lax_ids = randsplit(lax, lax_slices_per_vol)[0]
        sax_ids = randsplit(sax, sax_slices_per_vol)[0] if sax_slices_per_vol > -1 else ([sax[0]] if sax else [])
        return lax_ids + sax_ids
    
    slices_train = [choose_slices(patients[pid]["cine_lax.mat"], patients[pid]["cine_sax.mat"]) for pid in pids_train]
    slices_test  = [choose_slices(patients[pid]["cine_lax.mat"], patients[pid]["cine_sax.mat"]) for pid in pids_test]

    slices_train = sum(slices_train, [])
    slices_test  = sum(slices_test, []) # this should equals set(range(len(dataset))) - set(slices_train) when slices_per_vol is high

    # Create datasets
    return Subset(dataset, slices_train), Subset(dataset, slices_test)

@contextmanager
def set_posix_windows():
    posix_backup = PosixPath
    try:
        PosixPath = WindowsPath
        yield
    finally:
        PosixPath = posix_backup