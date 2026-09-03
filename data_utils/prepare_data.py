# Data for voice anonymization evaluation and (final) attribute control evaluation
# VPC (VCTK full, LibriTTS dev, test, IEMOCAP full)

# Make sure no data in train or validation is part of VPC dev or evaluation!

import importlib
import os
import sys
from collections import defaultdict

# Kaldi recipe
# https://kaldi-asr.org/doc/data_prep.html
def save_to_kaldi_format(df, working_dir="workspace/voice_reference_dataset"):
    """
        Input: df with the following columns:
            ['prefix', 'utterance_id', 'speaker_id', 'audio_path', 'transcript', 'gender', 'emotion']
        Creates a kaldi-format folder in the given directory.
    """
    kaldi_dir = os.path.join(working_dir, 'kaldi')
    os.makedirs(kaldi_dir, exist_ok=True)

    wav_scp = []
    utt2spk = []
    text = []
    utt2gender = []
    utt2emotion = []
    spk2gender_dict = {}

    for i, row in df.iterrows():
        # Make utterance-id unique, e.g. ravdess_utt001
        utt_id = f"{row['prefix']}_{row['utterance_id']}"
        spk_id = f"{row['prefix']}_{row['speaker_id']}"

        wav_scp.append(f"{utt_id} {row['audio_path']}\n")
        utt2spk.append(f"{utt_id} {spk_id}\n")
        text.append(f"{utt_id} {row['transcript']}\n")

        if "gender" in df.columns:
            utt2gender.append(f"{utt_id} {row['gender'][0].lower()}\n")  # m/f

         # spk2gender (only store once)
        if spk_id not in spk2gender_dict:
            spk2gender_dict[spk_id] = row['gender'][0].lower()  # m/f

        if "emotion" in df.columns:
            utt2emotion.append(f"{utt_id} {row['emotion']}\n")

    spk2gender = []
    for k, v in spk2gender_dict.items():
        spk2gender.append(f"{k} {v}\n")

    # Write files
    with open(os.path.join(kaldi_dir, "spk2gender"), "w", encoding="utf-8") as f:
        f.writelines(spk2gender)

    with open(os.path.join(kaldi_dir, "wav.scp"), "w", encoding="utf-8") as f:
        f.writelines(wav_scp)

    with open(os.path.join(kaldi_dir, "utt2spk"), "w", encoding="utf-8") as f:
        f.writelines(utt2spk)

    with open(os.path.join(kaldi_dir, "text"), "w", encoding="utf-8") as f:
        f.writelines(text)

    if utt2gender:
        with open(os.path.join(kaldi_dir, "utt2gender"), "w", encoding="utf-8") as f:
            f.writelines(utt2gender)

    if utt2emotion:
        with open(os.path.join(kaldi_dir, "utt2emotion"), "w", encoding="utf-8") as f:
            f.writelines(utt2emotion)

    # spk2utt
    spk2utt_map = defaultdict(list)
    for line in utt2spk:
        utt, spk = line.strip().split()
        spk2utt_map[spk].append(utt)

    with open(os.path.join(kaldi_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for spk, utts in spk2utt_map.items():
            f.write(f"{spk} {' '.join(utts)}\n")

    print(f"Kaldi data saved in: {kaldi_dir}")