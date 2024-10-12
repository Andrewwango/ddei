from munch import DefaultMunch
from argparse import ArgumentParser
import wandb
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

import deepinv as dinv

from utils import Trainer, patient_random_split, ArtifactRemovalCRNN, CRNN, DeepinvSliceDataset, CineNetDataTransform

parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/CMRxRecon", help="Root dir for CMRxRecon data")
parser.add_argument("--mask", type=str, default="TimeVaryingGaussianMask08", help="Subfolder name containing masks")
parser.add_argument("--loss", type=str, default="ddei", help="Name of loss")
parser.add_argument("--model", type=str, default="ArtifactRemovalCRNN", help="Name of model")
parser.add_argument("--noise", type=float, default=0., help="Noise sigma")
parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--seed", type=int, default=1, help="Random seed")

args = parser.parse_args()

config = DefaultMunch(
    data_dir=args.data_dir,
    mask=args.mask,
    loss=args.loss,
    model=args.model,
    noise=args.noise,
    epochs=args.epochs,
    batch_size=args.batch_size,
    seed=args.seed,
)

torch.manual_seed(config.seed)
np.random.seed(config.seed)
generator = torch.Generator().manual_seed(config.seed)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

with wandb.init(project="cmr-experiments", config=config, dir="./wandb"):
    config = wandb.config

    # Define physics
    img_size = (2, 12, 512, 256)
    physics = dinv.physics.DynamicMRI(img_size=(config.batch_size, *img_size), device=device)
    if config.noise > 0:
        physics.noise_model = dinv.physics.GaussianNoise(config.noise, rng=generator)

    # Define data   
    dataset = DeepinvSliceDataset(
        root=config.data_dir, 
        transform=CineNetDataTransform(time_window=12, apply_mask=True, normalize=True), 
        set_name="TrainingSet",
        acc_folders=["FullSample"],
        mask_folder=config.mask,
        dataset_cache_file="dataset_cache.pkl",
        noise_level=config.noise,
        generator=generator
        )

    train_dataset, test_dataset = patient_random_split(
        dataset, 0.8, sax_slices_per_vol=-1, lax_slices_per_vol=3, generator=generator
    )
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=True, generator=generator)
    test_dataloader  = DataLoader(dataset= test_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True, shuffle=False)

    # Define model
    model = ArtifactRemovalCRNN(CRNN(num_cascades=2)).to(device)

    # Define group transforms
    rotate = dinv.transform.Rotate(n_trans=1, interpolation_mode=InterpolationMode.BILINEAR)
    tempad = dinv.transform.ShiftTime(n_trans=1)
    diffeo = dinv.transform.CPABDiffeomorphism(n_trans=1, device=device)
    
    # Define losses
    mcloss = dinv.loss.MCLoss() if "sure" not in config.loss.lower() else dinv.loss.SureGaussianLoss(sigma=config.noise, tau=0.1)
    match config.loss.replace("sure", ""):
        case "sup":
            losses = [dinv.loss.SupLoss()]
        case "ei-r":
            losses = [mcloss, dinv.loss.EILoss(rotate)]
        case "ddei":
            losses = [mcloss, dinv.loss.EILoss(tempad | (diffeo | rotate))]
        case "t-ssdu" | "t-ssdu*":
            losses = [dinv.loss.SplittingLoss(
                split_ratio=0.6, 
                mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator(tensor_size=img_size, split_ratio=0.6, device=device),
                eval_split_input=("*" in config.loss), 
                eval_n_samples=5
            )]
            model = losses[0].adapt_model(model)
        case "phase2phase":
            losses = [dinv.loss.Phase2PhaseLoss(img_size, device=device)]
            model = losses[0].adapt_model(model)
    
    ## SURE must be interleaved to save memory
    if isinstance(mcloss, dinv.loss.SureGaussianLoss):
        losses = [dinv.loss.InterleavedLossScheduler(*losses)]
    
    # Define metrics
    metrics = [
        dinv.metric.PSNR(max_pixel=None, complex_abs=True),
        dinv.metric.SSIM(max_pixel=None, complex_abs=True),
        dinv.metric.NMSE(complex_abs=True)
    ]

    trainer = Trainer(
        model = model,
        physics = physics,
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3),
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = config.epochs,
        losses = losses,
        scheduler = None,
        metrics = metrics,
        online_measurements = False,
        ckp_interval = 1000,
        device = device,
        eval_interval = 25,
        save_path = f"models/{wandb.run.id}",
        plot_images = False,
        wandb_vis = True,
    )

    trainer.train()

    # Evaluate
    results = trainer.test(test_dataloader)
    print(results)
    with open(f"models/results_{config.loss}.json", "w") as f:
        json.dump(results, f)