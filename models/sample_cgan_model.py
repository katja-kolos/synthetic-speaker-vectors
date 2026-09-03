#!/usr/bin/env python3
import argparse
import logging
import os

import numpy as np
import torch

from cWGAN import EmbeddingsGenerator

def _generate_artificial_embeddings_for_labels(generator, labels, file_output_dir, inv_norm=True, prefix=''):
    logging.info(f"Generate {len(labels)} artificial speaker embeddings...")
    labels = torch.LongTensor(labels)

    if inv_norm:
        gan_vectors = generator.generate_embeddings(labels=labels)
    else:
        gan_vectors = generator.generate_embeddings_without_normalization(labels=labels)
    unused_indices = np.arange(len(gan_vectors))


    output_path = os.path.join(file_output_dir, f'{prefix}vectors_file.pt')
    unused_output_path = os.path.join(file_output_dir, f'{prefix}unused_indices_file.pt')
    label_output_path = os.path.join(file_output_dir, f'{prefix}labels.pt')
    torch.save(gan_vectors, output_path)
    logging.info(f'Saved to {output_path}')
    torch.save(unused_indices, unused_output_path)
    logging.info(f'Saved unused to {output_path}')
    torch.save(labels, label_output_path)
    logging.info(f'Saved labels to {label_output_path}')
    return gan_vectors, unused_indices


def _generate_artificial_embeddings(gan_model_path: str, file_output_dir, labels: list=None, num_embeddings=0, inv_norm=True, gpu_id=None):
    os.makedirs(file_output_dir, exist_ok=True)

    if gpu_id is not None:
        device=f'cuda:{gpu_id}'
    else:
        device='cpu'
    generator = EmbeddingsGenerator(gan_path=gan_model_path, device=device)

    if labels:
        gan_vectors, unused_indices = _generate_artificial_embeddings_for_labels(generator, labels, file_output_dir, inv_norm)
        return gan_vectors, unused_indices
    
    else:
        if not num_embeddings:
            num_embeddings = 1000
        
        # {'m': 0, 'f': 1}
        m_labels = [0] * int(num_embeddings / 2)
        f_labels = [1] * int(num_embeddings / 2)
        logging.info(f"Generate {len(m_labels)} 'm' and {len(f_labels)} 'f' artificial speaker embeddings...")
        
        m_gan_vectors, m_unused_indices = _generate_artificial_embeddings_for_labels(generator, m_labels, file_output_dir, inv_norm, prefix='m')
        f_gan_vectors, f_unused_indices = _generate_artificial_embeddings_for_labels(generator, f_labels, file_output_dir, inv_norm, prefix='f')
        return {
            'm': (m_gan_vectors, m_unused_indices),
            'f': (f_gan_vectors, f_unused_indices)
        }


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
        "--inv_norm",
        type=int, #0 or 1
        required=True,
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
        gan_model_path=args.model_path,
        file_output_dir=args.output_dir,
        num_embeddings=args.num_embeddings,
        inv_norm=args.inv_norm,
    )


if __name__ == "__main__":
    main()