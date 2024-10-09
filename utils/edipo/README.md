# EDIPO CMRxRecon code base

Modified CMRxRecon dataloading and NN architecture [code from the EDIPO submission](https://github.com/vios-s/CMRxRECON_Challenge_EDIPO) to the [CMRxRecon 2023 challenge](https://cmrxrecon.github.io/). 

The modifications have been made so that the data and NN model can be integrated with the [DeepInverse library](https://deepinv.github.io/).

Modifications:

`edipo.data.mri_data`:
- Add `set_name` param to `SliceDataset` and add to `self.root path`
- Change path splitting from `/` to `os.sep` in `SliceDataset.__getitem__`
- Add mask_folder and image_folder params to `SliceDataset` and select kspaces and masks from them in `__init__` and `__getitem__`
- `SliceDataset._retrieve_metadata`'s `loadmat` also tries `scipy.io.loadmat`
- Mask loaded from first non-header mat key, not just first key, in `SliceDataset.__getitem__`
- `tqdm` for folders
- Add `DeepinvSliceDataset` that wraps `SliceDataset` and permutes dimensions to get data of shape `B,C,H,W`

`edipo.data.transforms`:
- Add `apply_mask` to `CineDataTransform.__init__` and apply mask in #5 in `__call__`
- Allow 2D+t mask `[w, h, t]` to be passed and applied

`edipo.models.crnn`:
- Originally named `recurrent_cinenet.py` but is really a pure CRNN
- Renamed `CineNet_RNN` to `CRNN`
- `CRNN.forward`: comment `mask = mask.unsqueeze(-1).float()`
- Add `ArtifactRemovalCRNN` which wraps CRNN to apply adjoint and permute dimensions before forward
