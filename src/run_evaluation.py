#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import torch

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)

from evaluation.distribution_analysis import (
    evaluate_cosine_distance, 
    evaluate_wasserstein_distance, 
    evaluate_sinkhorn_distance,
    evaluate_KL_divergence_KDE,
    _summarize_embeddings_)

from evaluation.audio_analysis import (
    calculate_wer,
    calculate_pesq,
)

def evaluate_embeddings(artificial_embeddings_path: str, natural_embeddings_path: str, output_dir: str, suffix: str):
    """
    Evaluates artificial embeddings against natural embeddings using a specified metric.
    
    Args:
        artificial_embeddings_path (str): Path to file containing artificial embeddings.
        natural_embeddings_path (str): Path to file containing natural embeddings.
        output_dir (str): Directory to save evaluation results.
    """
    logging.info("Starting embedding evaluation...")
    logging.info(f"Artificial embeddings: {artificial_embeddings_path}")
    logging.info(f"Natural embeddings: {natural_embeddings_path}")
    os.makedirs(output_dir, exist_ok=True)
    

    # Load embeddings
    artificial_embeddings = torch.load(artificial_embeddings_path, map_location="cpu").numpy()
    logging.info(f"Loaded sample of artificial embeddings of shape: {artificial_embeddings.shape}")
    natural_embeddings = torch.load(natural_embeddings_path, map_location="cpu").numpy()
    logging.info(f"Loaded sample of natural embeddings of shape: {natural_embeddings.shape}")

    # Names
    metrics_output_file = os.path.join(output_dir, f"embedding_metrics_{suffix}.json")
    logging.info(f"Embedding metrics will be saved to: {metrics_output_file}")
    viz_output_dir = os.path.join(output_dir, f"_viz_{suffix}")
    os.makedirs(viz_output_dir, exist_ok=True)
    logging.info(f"Embedding visualizations will be saved to: {viz_output_dir}")

    # Calculate metrics
    logging.info("Calculating metrics...")
    # -- Cosine distances --
    # -- 1) Inter-sample (artificial embeddings X natural embeddings)
    cos_dist_metrics_art_nat = evaluate_cosine_distance(artificial_embeddings, natural_embeddings, os.path.join(output_dir, f"cos_dist_{suffix}.txt"))
    avg_min_cos_dist = cos_dist_metrics_art_nat["Avg. min cos distance"]
    logging.info(f"Avg. minimal pairwise cosine distance (inter-sample): {avg_min_cos_dist}")
    # -- 2) Intra-sample nat (natural embeddings X natural embeddings)
    cos_dist_metrics_nat_nat = evaluate_cosine_distance(natural_embeddings, natural_embeddings, os.path.join(output_dir, f"cos_dist_{suffix}_nat.txt"))
    
    # -- 3) Intra-sample nat (artificial embeddings X artificial embeddings)
    cos_dist_metrics_art_art = evaluate_cosine_distance(artificial_embeddings, artificial_embeddings, os.path.join(output_dir, f"cos_dist_{suffix}_art.txt"))
    
    # Summary of mean / var 
    summary_nat = _summarize_embeddings_(natural_embeddings)
    logging.info(f"Summary of natural sample: {summary_nat}")
    summary_art = _summarize_embeddings_(artificial_embeddings)
    logging.info(f"Summary of artificial sample: {summary_art}")

    w_dist = evaluate_wasserstein_distance(artificial_embeddings, natural_embeddings)
    logging.info(f"Wasserstein distance (art, nat): {w_dist}")
    inv_w_dist = evaluate_wasserstein_distance(natural_embeddings, artificial_embeddings)
    logging.info(f"Wasserstein distance (nat, art): {inv_w_dist}")


    sinkhorn_dist = evaluate_sinkhorn_distance(artificial_embeddings, natural_embeddings)
    logging.info(f"Sinkhorn distance (art, nat): {sinkhorn_dist}")
    inv_sinkhorn_dist = evaluate_sinkhorn_distance(artificial_embeddings, natural_embeddings)
    logging.info(f"Sinkhorn distance (nat, art): {inv_sinkhorn_dist}")
   

    kl_div = evaluate_KL_divergence_KDE(artificial_embeddings, natural_embeddings)
    logging.info(f"KL divergence (art, nat): {kl_div}")
    inv_kl_div = evaluate_KL_divergence_KDE(natural_embeddings, artificial_embeddings)
    logging.info(f"KL divergence (nat, art): {inv_kl_div}")


    # Save final result
    with open(metrics_output_file, "w") as f:
        json.dump({
            "Path A": artificial_embeddings_path,
            "Path N": natural_embeddings_path,
            "Summary": {"A": summary_art, "N": summary_nat},
            "Pairwise cosine distance (intra-sample, art)": cos_dist_metrics_art_art,
            "Pairwise cosine distance (intra-sample, nat)": cos_dist_metrics_nat_nat,
            "Pairwise cosine distance (inter-sample)": cos_dist_metrics_art_nat,
            "Wasserstein distance (art, nat)": float(w_dist),
            "Wasserstein distance (nat, art)": float(inv_w_dist),
            "Sinkhorn distance (art, nat)": float(sinkhorn_dist),
            "Sinkhorn distance (nat, art)": float(inv_sinkhorn_dist),
            "Kullback-Leibler divergence KDE (art, nat)": float(kl_div),
            "Kullback-Leibler divergence KDE (nat, art)": float(inv_kl_div),
        }, f, indent=2)

    
    # Boxplot for each of the 128 dimensions
    plt.figure(figsize=(24, 6))
    plt.boxplot(artificial_embeddings, vert=True, patch_artist=True)
    # plt.title("Distribution of GST Embedding Dimensions - Synth")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(viz_output_dir, "synthetic.png"))
    logging.info(f"""Saved image for embedding dimensions (synthetic) to {os.path.join(viz_output_dir, "synthetic.png")}""")

    # Boxplot for each of the 128 dimensions
    plt.figure(figsize=(24, 6))
    plt.boxplot(natural_embeddings, vert=True, patch_artist=True)
    # plt.title("Distribution of GST Embedding Dimensions - Natural")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(viz_output_dir, "natural.png"))
    logging.info(f"""Saved image for embedding dimensions (natural) to {os.path.join(viz_output_dir, "natural.png")}""")


def evaluate_wer(artificial_audio_path: str, natural_audio_path: str, output_dir: str, suffix: str):
    """
    Evaluates transcript-based metrics. 
    Args:
        artificial_audio_path (str): Path to folder containing audios generated from artificial embeddings, and their transcript.txt
        natural_audio_path (str): Path to folder containing natural audios OR audios generated from natural embeddings, and their transcript.txt
        output_dir (str): Directory to save evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    # ASR Using pre-trained Whisper model + WER
    # synthetic_transcript_path = os.path.join(output_dir, f"transcripts_{suffix}.txt")
    synthetic_transcript_path = os.path.join(output_dir, f"transcripts.txt")
    # wer_path = os.path.join(output_dir, f"wer_{suffix}.txt")
    wer_path = os.path.join(output_dir, f"wer.txt")

    synthetic_transcripts, scores = calculate_wer(artificial_audio_path, natural_audio_path, synthetic_transcript_path, wer_path)

    logging.info(f"ASR results for synthetic audios were saved to: {synthetic_transcript_path}")
    logging.info(f"WER results were saved to: {wer_path}")
    logging.info(f"WER summary: {scores.summary}")

    summary_path = os.path.join(output_dir, f"wer_summary_{suffix}.txt")
    with open(summary_path, "w") as f:
        if scores != None:
            f.write(json.dumps(scores.summary, indent=2))
        else:
            f.write("WER could not be calculated!")

def evaluate_audios(artificial_audio_path: str, natural_audio_path: str, output_dir: str, suffix: str):
    """
    Evaluates wave-based metrics. 
    Args:
        artificial_audio_path (str): Path to folder containing audios generated from artificial embeddings, and their transcript.txt
        natural_audio_path (str): Path to folder containing natural audios OR audios generated from natural embeddings, and their transcript.txt
        output_dir (str): Directory to save evaluation results.
    """
    
    pesq_path = os.path.join(output_dir, f"pesq{suffix}.txt")
    pesq_scores, pesq_summary = calculate_pesq(artificial_audio_path, natural_audio_path, pesq_path)

    summary_path = os.path.join(output_dir, f"audio_summary{suffix}.txt")
    with open(summary_path, "w") as f:
        f.write(json.dumps(pesq_summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate artificial embeddings against natural embeddings."
    )

    parser.add_argument(
        "--artificial-embeddings",
        "-a",
        required=True,
        help="Path to the file containing artificial embeddings."
    )

    parser.add_argument(
        "--natural-embeddings",
        "-n",
        required=True,
        help="Path to the file containing natural embeddings."
    )

    parser.add_argument(
        "--artificial-audios",
        required=False,
        help="Path to the folder containing artificial wavs, and their transcripts.txt."
    )

    parser.add_argument(
        "--natural-audios",
        required=False,
        help="Path to the folder containing natural wavs (for the same utterances as artificial wavs), and their transcripts.txt."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where evaluation results will be stored."
    )

    parser.add_argument(
        "--setting",
        required=True,
        help="Experiment setting"
    )

    parser.add_argument(
        "--eval-mode",
        "-m",
        required=False,
        default="vec",
        choices=["vec", "wav", "wer", "vec+wer", "wav+wer", "vec+wav", "vec+wav+wer"],
        help="Mode of evaluation: vec for embeddings only, wav for audio metrics like PESQ, wer for SER, WER etc, vec+wav for both (default: vec)"
    )

    parser.add_argument(
        "--suffix",
        required=False,
        help="Will be appended to names of output files before extension.",
        default=""
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity level (default: info)."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    suffix = f"_{args.suffix}"

    if "vec" in args.eval_mode:
        evaluate_embeddings(
            artificial_embeddings_path=args.artificial_embeddings,
            natural_embeddings_path=args.natural_embeddings,
            output_dir=os.path.join(args.output_dir, "metrics", args.setting),
            suffix=suffix
        )

    if "wav" in args.eval_mode:
        evaluate_audios(
            artificial_audio_path=args.artificial_audios,
            natural_audio_path=args.natural_audios,
            output_dir=os.path.join(args.output_dir, "metrics", args.setting),
            suffix=suffix
        )

    if "wer" in args.eval_mode:
        evaluate_wer(
            artificial_audio_path=args.artificial_audios,
            natural_audio_path=args.natural_audios,
            output_dir=os.path.join(args.output_dir, "res_transcripts", args.setting),
            suffix=suffix
        )


if __name__ == "__main__":
    main()
