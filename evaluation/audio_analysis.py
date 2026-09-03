import json
import logging
import os
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from scipy.io import wavfile
from pesq import pesq
from tqdm import tqdm

module_path = os.path.abspath("../Voice-Privacy-Challenge-2024")
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath("../Voice-Privacy-Challenge-2024/evaluation")
if module_path not in sys.path:
    sys.path.append(module_path)

from utility.asr.speechbrain_asr.inference import InferenceSpeechBrainASR

# Helpers

def compute_single_pesq(args):
    utt_id, ref_path, synth_path = args
    rate_ref, ref = wavfile.read(ref_path)
    rate_deg, deg = wavfile.read(synth_path)

    try:
        wb = pesq(rate_ref, ref, deg, 'wb')
        nb = pesq(rate_ref, ref, deg, 'nb')
        return utt_id, {"wb_pesq": wb, "nb_pesq": nb, "error": None}

    except Exception as e:
        return utt_id, {"wb_pesq": None, "nb_pesq": None, "error": str(e)}


def compute_pesq_parallel(utt_ids_and_paths_nat, utt_ids_and_paths_synth, n_workers=None):
    items = [
        (utt_id, utt_ids_and_paths_nat[utt_id], synth_path)
        for utt_id, synth_path in utt_ids_and_paths_synth.items()
    ]

    metrics = {}
    with Pool(processes=n_workers or cpu_count()) as pool:
        for utt_id, result in tqdm(
            pool.imap_unordered(compute_single_pesq, items),
            total=len(items),
            desc="Computing PESQ",
        ):
            metrics[utt_id] = result

    return metrics

def normalize_text(text):
    # Transcripts are returned by ASR model in all capitals, while the original transcripts are given in classic orthographic way.
    # Speechbrain's WER calculation is case sensitive. We postprocess the references and hypotheses ignoring these differences, to have an optimistic estimate.

    text = text.lower() 
    text = re.sub(r"[^\w\s]", "", text) # remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # normalize whitespace after removing punctuation
    # this still doesn't do anything about US vs UK spelling but let's not think about that for now
    return text



# Metrics

def calculate_wer(audio_folder: str, natural_audio_folder: str, out_file: str, wer_out_file: str):
    # Calculates WER & SER with speechbrain
    # If transcripts are not available, performs ASR
    # Returns:
    #  dict of transcripts: for each utterance id (filename with wav stripped), the corresponding text 
    #  wer summary
    
    model_path = os.path.abspath("/home/users1/kolosea/hiwi/thesis/workspace/models/asr-wav2vec2-librispeech")
    asr_hparams = os.path.abspath("/home/users1/kolosea/hiwi/thesis/workspace/models/asr-wav2vec2-librispeech/hyperparams.yaml")
    model_type = "EncoderASR"
    device = "cuda:3"

    asr = InferenceSpeechBrainASR(model_path, asr_hparams, model_type, device)

    if not os.path.isfile(out_file):
        logging.info("No transcript found. Calculating.")
        # transcribe if transcript is not available already
        audio_paths =  [f for f in os.listdir(audio_folder) if f.lower().endswith(".wav") and os.path.isfile(os.path.join(audio_folder, f))]
        # remove helper prefixes added for readability to recover intended file paths
        PREFIX_RE = re.compile(
            r'^(spk\d+_|resynth_|synth_random_|synth_wgan_|synth_diffusion_)'
        )

        audio_paths = {
            PREFIX_RE.sub('', os.path.basename(x)).removesuffix('.wav'):
                os.path.abspath(os.path.join(audio_folder, x))
            for x in audio_paths
        }

        logging.info("Example audio_paths:")
        for k,v in audio_paths.items():
            logging.info(f"{k}: {v}")
            break
        hyp = asr.transcribe_audios_old(audio_paths=audio_paths, out_file=Path(out_file).absolute())

    else:
        logging.info(f"Transcript available at {out_file}")
        hyp = {}
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, *words = line.strip().split()
                hyp[utt_id] = " ".join(words)

    ref = {}
    with open(os.path.join(natural_audio_folder, "text"), "r", encoding="utf-8") as f:
        for line in f:
            utt_id, *words = line.strip().split()
            ref[utt_id] = " ".join(words)

    # preprocess -- see also: https://github.com/SarinaMeyer/VoicePAT-private/blob/codeswitching/analysis/wer_analysis.py for removing non-alpha
    ref = {utt_id: normalize_text(t) for utt_id, t in ref.items()}
    hyp = {utt_id: normalize_text(t) for utt_id, t in hyp.items()}

    res = asr.compute_wer(
        ref_texts=ref, 
        hyp_texts=hyp,
        out_file=Path(wer_out_file).absolute())

    return hyp, res


def calculate_pesq(audio_folder: str, natural_audio_folder: str, out_file: str):
    #Inputs:
    # audio_folder: path to synthetic wav files
    # natural_audio_folder: path to kaldi with reference data (has wav.scp, audio paths are restored from it)

    synth_audio_paths = [f for f in os.listdir(audio_folder) if f.lower().endswith(".wav") and os.path.isfile(os.path.join(audio_folder, f))]
    utt_ids_and_paths_synth = {(os.path.basename(x)
                        .removeprefix("resynth_")
                        .removeprefix("synth_random_")
                        .removeprefix("synth_wgan_")
                        .removeprefix("synth_diffusion_")
                        .removeprefix("synth_wgan-no-inv-norm_")
                        .removeprefix("synth_wgan-IMS_")
                        .removesuffix(".wav")): 
                        os.path.abspath(os.path.join(audio_folder, x)) for x in synth_audio_paths}
    
    nat_audio_paths = [f for f in os.listdir(natural_audio_folder) if f.lower().endswith(".wav") and os.path.isfile(os.path.join(natural_audio_folder, f))]
    wav_scp = os.path.join(natural_audio_folder, "wav.scp")
    utt_ids_and_paths_nat = {}
    with open(wav_scp, 'r') as f:
        for line in f:
            utt_id, path = line.strip().split()
            utt_ids_and_paths_nat[utt_id] = os.path.abspath(path)
    
    
    pesq_metrics = compute_pesq_parallel(
        utt_ids_and_paths_nat,
        utt_ids_and_paths_synth,
        n_workers=8
    )
        
    with open(out_file, 'w') as f:
        f.write(json.dumps(pesq_metrics))
    
    pesq_metrics_df = pd.DataFrame(pesq_metrics).transpose()
    valid = pesq_metrics_df.dropna(subset=["wb_pesq", "nb_pesq"])

    logging.info(f"Went through {len(pesq_metrics_df)} utterances. Returning mean PESQ for {len(valid)} utterances.")
    
    pesq_summary = {
        "mean_wb_pesq": valid["wb_pesq"].mean(),
        "mean_nb_pesq": valid["nb_pesq"].mean(),
    }

    return pesq_metrics, pesq_summary
