import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from WGAN.dataset import SpeakerEmbeddingsDataset
from Diffusion.training.train_diffusion import train_diffusion
from Diffusion.training.logger import setup_logger, setup_tensorboard
from Diffusion.init_diffusion import create_diffusion

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        help="Path to speaker embeddings",
        default=Path("dataset/reference_embeddings/utt-level/speaker_vectors.pt"),
        type=Path,
    )
    parser.add_argument(
        "--model_dir",
        help="Path to folder where model will be saved",
        default="../../workspace/models/diffusion",
        type=Path,
    )
    parser.add_argument("--gpu_id", help="GPU to use for training", default=0)
    parser.add_argument("--id", default=None)
    parser.add_argument("--config", type=str, default="./WGAN/configs/train_gan.json")
    args = parser.parse_args()
    return args

def main(args):
    print("Started script for diffusion training.")
    logger = setup_logger()
    writer, timestampStr = setup_tensorboard(logger)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device '{}'".format(device))

    with open(Path(args.config), "r") as f:
        diffusion_parameters = json.load(f)

    print(f"Loaded your config from {args.config}.")
    print("Diffusion parameters")
    print(diffusion_parameters)

    diffusion = create_diffusion(parameters=diffusion_parameters, device=device)
    print("Created diffusion model!")

    # create dataset & dataloader
    dataset = SpeakerEmbeddingsDataset(
        feature_path=args.data_path,
        device=device,
        normalize_data=diffusion_parameters["normalize_data"],
    )
    print(f"Loaded dataset from {args.data_path}.")
    print("Number of dataset samples:    {:06d}".format(len(dataset)))

    print("Starting the training...")
    train_diffusion(
        model=diffusion,
        dataset=dataset,
        parameters=diffusion_parameters,
        logger=logger,
        device=device,
        writer=writer,
        timestampStr=timestampStr,
        models_dir=args.model_dir,
        # save_vis_every=diffusion_parameters["save_every"]
    )
    print(f"Done! Check the model in {args.model_dir}.")


if __name__ == "__main__":
    main(get_args())