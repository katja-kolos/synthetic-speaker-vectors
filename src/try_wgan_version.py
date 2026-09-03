import os
import sys
from pathlib import Path
import argparse
import torch
from tqdm.auto import tqdm
from types import SimpleNamespace

# Ensure CUDA errors are raised immediately
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Add necessary paths for imports
paths_to_add = [
    "../models",
    "../models/WGAN/training",
    "../",
    "../Voice-Privacy-Challenge-2024"
]

for p in paths_to_add:
    abs_path = os.path.abspath(p)
    if abs_path not in sys.path:
        sys.path.append(abs_path)

from data_utils.listen_to_voice import _save_audio, _save_mel_spectrogram_to_image


from train_gan_model import main as train_model
from sample_gan_model import _generate_artificial_embeddings
from anonymization.modules.sttts.tts.ims_tts import ImsTTS

hifigan_path = '../../workspace/models/tts/HiFiGAN_combined/best.pt'
fastspeech_path = '../../workspace/models/tts/FastSpeech2_Multi/prosody_cloning.pt'
embedding_path = '../../workspace/models/tts/Embedding/embedding_function.pt'
device = 'cuda:3'

# --------- HELPER FUNCTIONS ---------
# train wgan model
def run_training(args: SimpleNamespace, log_file: Path = None):
    """Run WGAN model training and log output."""
    if log_file:
        from contextlib import redirect_stdout
        with open(log_file, "w") as f:
            with redirect_stdout(f):
                train_model(args)
    else:
        train_model(args)

# sample vectors from trained model
def find_latest_checkpoint(model_dir: Path, suffix="_wgan") -> Path:
    files = list(model_dir.glob(f"*{suffix}"))
    if not files:
        # raise FileNotFoundError(f"No wgan checkpoint found in {model_dir}")
        print(f"No wgan checkpoint found in {model_dir}")
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file

def sample_vector(vectors: torch.Tensor) -> torch.Tensor:
    """Randomly sample one vector from wgan embeddings."""
    idx = torch.randint(0, vectors.size(0), (1,))
    vec = vectors[idx].squeeze(0)
    print(f"Sampled vector {idx.item()} of size {vec.shape[0]}. Mean: {vec.mean():.4f}")
    return vec

def generate_synthetic_embeddings(model_path: Path, output_dir: Path, n_vectors: int, gpu_id: int, normalize_data: bool=True):
    return _generate_artificial_embeddings(
        gan_model_path=str(model_path),
        file_output_dir=str(output_dir),
        n=n_vectors, # for unconditional
        # labels=labels, # for conditional
        inv_norm=normalize_data,
        gpu_id=gpu_id
    )

# synthesize audios with synthetic vectors
def load_transcripts(kaldi_text_path: str) -> dict:
    """Load Kaldi text file into a dict {utterance_id: transcript}."""
    transcripts = {}
    with open(kaldi_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text
    return transcripts

def synthesize_harvard_sentences(vectors: torch.Tensor, setting: str, do_save: str='wav'):
    transcripts_path = "../../workspace/HarvardSent/evaluation/text"
    transcripts = load_transcripts(transcripts_path)
    audio_output_dir = Path("../../workspace/HarvardSent/evaluation") / "wav" / setting
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    tts = ImsTTS(hifigan_path, fastspeech_path, device, embedding_path=embedding_path, output_sr=16000, lang='en')

    for utt_id, sentence in transcripts.items():
        speaker_vector = sample_vector(vectors)

        assert type(speaker_vector) == torch.Tensor, f"Speaker vector must be a torch.Tensor but we got: {type(speaker_vector)}"
        assert speaker_vector.shape[0] == 128, f"Expecting a valid GST embedding of size 128 but we got {speaker_vector.shape[0]}."
      
        wave, mel = tts.read_text(sentence, speaker_vector, 
                                  text_is_phones=False, duration=None, pitch=None, energy=None,
                                  start_silence=None, end_silence=None)

        if 'mel' in do_save:
            mel = mel.cpu().detach().numpy()
            mel_spectrogram_filename = os.path.join(audio_output_dir, f"{utt_id}.png")
            _save_mel_spectrogram_to_image(mel_spectrogram_filename, mel)

        if 'wav' in do_save:
            wav_filename = os.path.join(audio_output_dir, f"{utt_id}.wav")
            _save_audio(wav_filename, wave)


    return audio_output_dir

# evaluate
def create_eval_script(setting: str, vectors_file: Path, natural_data: Path, artificial_audios: Path):
    script_path = Path(f"eval_{setting}.sh")
    with open(script_path, "w") as f:
        f.write(f"""#!/bin/bash
echo "Evaluating embedding pool for WGAN-{setting} vectors..."
python ./run_evaluation.py \\
-a "{vectors_file}" \\
-n "{natural_data}/embeddings/reference_embeddings/utt-level/speaker_vectors.pt" \\
-o "/home/users1/kolosea/hiwi/thesis/workspace/metrics" \\
--artificial-audios "{artificial_audios}" \\
--natural-audios "/home/users1/kolosea/hiwi/thesis/workspace/HarvardSent/evaluation" \\
--setting "{setting}" \\
--eval-mode "vec+wer"
echo "Done."
""")
    return script_path

# --------- MAIN WORKFLOW ---------
def main():
    parser = argparse.ArgumentParser(
        description="Train wgan model, sample embeddings, synthesize audio, and create eval script."
    )

    # Mandatory argument
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="Version of the wgan model (e.g., v0.2)"
    )

    # Optional arguments with sensible defaults
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU ID to use")
    parser.add_argument("--n-vectors", type=int, default=4616, help="Number of synthetic vectors to sample")
    parser.add_argument("--data-path", type=str, default="../../workspace/selected_voice_reference_dataset/", help="Path to training embeddings")
    parser.add_argument("--model-dir", type=str, default="../../workspace/models", help="Base directory for model checkpoints")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to wgan training config (optional, auto-generated from setting if not specified)"
    )
    parser.add_argument(
        "--file-output-dir",
        type=str,
        default=None,
        help="Directory to save synthetic embeddings (optional, auto-generated from setting if not specified)"
    )

    args = parser.parse_args()

    # Resolve paths based on the provided setting
    MODEL_DIR = Path(args.model_dir) / f"wgan-{args.setting}"
    CONFIG_PATH = Path(args.config_path) if args.config_path else Path(f"../models/WGAN/configs/train_gan-{args.setting}.json")
    FILE_OUTPUT_DIR = Path(args.file_output_dir) if args.file_output_dir else Path(f"../../workspace/synthetic_speaker_embeddings/WGAN-{args.setting}")
    DATA_PATH = Path(args.data_path)

    # Ensure output directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Setting:", args.setting)
    print("Model dir:", MODEL_DIR)
    print("Config path:", CONFIG_PATH)
    print("Data path:", DATA_PATH)
    print("File output dir:", FILE_OUTPUT_DIR)
    print("GPU ID:", args.gpu_id)
    print("Number of vectors:", args.n_vectors)

    vectors_file = FILE_OUTPUT_DIR / "vectors_file.pt"

    if not Path(vectors_file).exists():
        # Vectors not available for this setting; (train and) sample
        model_checkpoint = find_latest_checkpoint(MODEL_DIR)
        if model_checkpoint is not None:
            print(f"Using checkpoint: {model_checkpoint}")
        else:
            print(f"No checkpoint available at {MODEL_DIR}; training...")

            # Training
            ns_args = SimpleNamespace(
                data_path=str(DATA_PATH),
                model_dir=str(MODEL_DIR),
                gpu_id=args.gpu_id,
                id=None,
                config=str(CONFIG_PATH)
            )
            log_file = MODEL_DIR / f"train_log_{args.setting}.txt"
            run_training(ns_args, log_file=log_file)

        # Sampling
        # parse json parameter file
        import json
        with open(Path(CONFIG_PATH), "r") as f:
            gan_parameters = json.load(f)
        model_checkpoint = find_latest_checkpoint(MODEL_DIR)
        vectors, unused_indices = generate_synthetic_embeddings(model_checkpoint, FILE_OUTPUT_DIR, args.n_vectors, args.gpu_id, gan_parameters['normalize_data'])

    else:
        # load sampled vectors
        vectors = torch.load(vectors_file, map_location="cpu")
        print(f"Loaded a {type(vectors)} of shape: {vectors.shape}")

    # Synthesize Harvard sentences
    setting_name = f"wgan-{args.setting}"
    artificial_audio_dir = synthesize_harvard_sentences(vectors, setting_name)

    # Create evaluation script
    eval_script_path = create_eval_script(setting_name, vectors_file, DATA_PATH, artificial_audio_dir)
    print(f"Evaluation script created at {eval_script_path}")

    # gender evaluation
    print(f"Evaluating gender distribution...")
    from evaluate_gender_distribution import process_model as classify_gender_in_synthetic_embeddings
    from evaluate_gender_distribution import classifier_model as gender_classifier
    model_version = setting_name
    embeddings_path = vectors_file
    classify_gender_in_synthetic_embeddings(embeddings_path, model_version, gender_classifier)
    print("...done.")

if __name__ == "__main__":
    main()
