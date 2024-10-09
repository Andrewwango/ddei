import os
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, NamedTuple, Dict, Any, Union, List

import mat73
import scipy
import torch
from natsort import natsorted
from tqdm import tqdm

from deepinv.physics import GaussianNoise

from .transform_utils import *

def loadmat(fname):
    try:
        return mat73.loadmat(fname)
    except TypeError:
        return scipy.io.loadmat(fname)

class RawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    
    
class SliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        set_name: str = "TestSet",
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = True,
        sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        acc_folders: List[str] = ['AccFactor04', 'AccFactor08', 'AccFactor10'],
        mask_folder: str = ""
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
        """

        self.root = root / ('SingleCoil/Cine') / set_name
        # self.root = root / ('TestSet')
        self.transform = transform
        self.dataset_cache_file = Path(dataset_cache_file)
        
        # The list of files names
        self.examples_names = []        
        # The list of actual data samples
        self.examples = []
        
        self.mask_folder = mask_folder

        #! Dataset cache related
        #* If the dataset cache file exists, load it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)

        # Otherwise, create it
        else:
            dataset_cache = {}
        
        #* Iterate through the dataset and create a list of samples
        if dataset_cache.get(self.root) is None:
            folders = []
            #* iterate each accelerate factor to get all folders
            for acc_folder in acc_folders:
                acc_file = natsorted(list(Path(os.path.join(self.root,acc_folder)).iterdir()))
                folders.append(acc_file)
            
            for folder in folders:
                # this block below has been indented under an 
                # external FOR loop that goes across the three folders 

                # of the "folders" object
                for patient in folder:
                    # get all mat file other than mask
                    for file in natsorted(os.listdir(Path(patient))):
                        if not file.endswith("_mask.mat"):
                            file = os.path.join(Path(patient), file)
                            self.examples_names.append(file)
                            
            #* now we got all names, we can read the mat file to get metadata
            for fname in tqdm(self.examples_names):
                metadata, num_slices = self._retrieve_metadata(fname)
                for slice_ind in range(num_slices):
                    self.examples.append(RawDataSample(fname, slice_ind, metadata))
                        
            #* Save the dataset cache
            if use_dataset_cache:
                print('Saving dataset cache file')
                dataset_cache[self.root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            print('Using dataset cache file')
            logging.info(f"Using dataset cache from {self.dataset_cache_file}")
            self.examples = dataset_cache[self.root]
            
        #! Sumsampling
        if sample_rate is None:
            self.sample_rate = 1.0
        elif sample_rate < 1.0:  # sample by slice
            seed = int(datetime.now().strftime("%H%M%S"))
            random.Random(seed).shuffle(self.examples)
            
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        
    @staticmethod
    def _retrieve_metadata(fname: Union[str, Path, os.PathLike]):
        data = loadmat(fname)

        shape = data[list(data.keys())[0]].shape
        if len(shape) == 5:
            metadata = {
                "width": shape[0],
                "height": shape[1],
                "coils": shape[2],
                "slices": shape[3],
                "timeframes": shape[4],
            }
            num_slices = shape[3]
        elif len(shape) == 4:
            metadata = {
                "width": shape[0],
                "height": shape[1],
                "slice": shape[2],
                "timeframes": shape[3],
            }
            num_slices = shape[2]
        else:
            raise ValueError(f"Unexpected shape: {shape}")
        
        return metadata, num_slices
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i:int):
        #* get data sample from self.examples and feed into transform
        fname, slice_index, metadata = self.examples[i]

        data = loadmat(fname)
        if len(data[list(data.keys())[0]].shape) == 5:
            kspace = data[list(data.keys())[0]][:, :, 0, slice_index, :]
        elif len(data[list(data.keys())[0]].shape) == 4:
            kspace = data[list(data.keys())[0]][:, :, slice_index, :]
        else:
            raise ValueError("Unexpected shape")
            
        attrs = metadata.copy()

        mask_name = self.root / (fname.split(os.sep)[-3] if self.mask_folder == "" else self.mask_folder) / fname.split(os.sep)[-2] / (fname.split(os.sep)[-1].split('.')[0] + '_mask.mat')
        data_mask = loadmat(mask_name)
        #mask = data_mask[list(data_mask.keys())[0]]
        mask = next(v for k, v in data_mask.items() if not k.startswith('__'))

        
        
        target = None
        del data, data_mask

        if self.transform:
            sample = self.transform(kspace, mask, target, attrs, fname, slice_index)
        
        return sample


class DeepinvSliceDataset(SliceDataset):
    # TODO deprecate and inherit dinv fastMRI instead
    def __init__(
        self,
        root: str | Path | os.PathLike,
        set_name: str = "TestSet",
        transform: Callable[..., Any] | None = None,
        use_dataset_cache: bool = True,
        sample_rate: float | None = None,
        dataset_cache_file: str | Path | os.PathLike = "dataset_cache.pkl",
        acc_folders: List[str] = ...,
        mask_folder: str = "",
        noise_level: float = 0.,
        generator: torch.Generator = torch.Generator(),
    ):
        super().__init__(
            Path(root),
            set_name,
            transform,
            use_dataset_cache,
            sample_rate,
            dataset_cache_file,
            acc_folders,
            mask_folder,
        )
        self.noise_level = noise_level
        self.generator = generator

    def __getitem__(self, i: int):
        sample = super().__getitem__(i)
        image = sample.image.permute(3, 2, 0, 1)  # HWTC->CTWH
        target = sample.target.permute(3, 2, 0, 1)  # HWTC->CTWH
        kspace = sample.kspace.permute(3, 2, 0, 1)
        mask = sample.mask[..., None].permute(3, 2, 0, 1).float()

        if self.noise_level is not None and self.noise_level > 0:
            kspace = GaussianNoise(sigma=self.noise_level, rng=self.generator)(kspace) * mask

        return target, kspace, mask