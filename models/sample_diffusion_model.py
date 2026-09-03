#!/usr/bin/env python3
import argparse
import logging
import os

import numpy as np
import torch

from Diffusion import EmbeddingsGenerator

def _generate_artificial_embeddings(model_path: str, file_output_dir, n: int, gpu_id=1):
    logging.info(f"Generate {n} artificial speaker embeddings...")
    if gpu_id is not None:
        device=f'cuda:{gpu_id}'
    else:
        device='cpu'
    generator = EmbeddingsGenerator(model_path=model_path, device=device)

    vectors = generator.generate_embeddings(n=n)

    unused_indices = np.arange(len(vectors))

    os.makedirs(file_output_dir, exist_ok=True)
    output_path = os.path.join(file_output_dir, 'vectors_file.pt')
    unused_output_path = os.path.join(file_output_dir, 'unused_indices_file.pt')
    torch.save(vectors, output_path)
    logging.info(f'Saved to {output_path}')
    torch.save(unused_indices, unused_output_path)
    logging.info(f'Saved unused to {output_path}')
    return vectors, unused_indices



def main():
    parser = argparse.ArgumentParser(
        description="Generate artificial speaker embeddings using a trained GAN model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained GAN model file."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the generated embedding files will be stored."
    )

    parser.add_argument(
        "-n",
        "--num_embeddings",
        type=int,
        required=True,
        help="Number of artificial embeddings to generate."
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        required=True,
        help="GPU ID where to load the model"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    _generate_artificial_embeddings(
        model_path=args.model_path,
        file_output_dir=args.output_dir,
        n=args.num_embeddings,
        gpu_id=args.gpu_id
    )


if __name__ == "__main__":
    main()