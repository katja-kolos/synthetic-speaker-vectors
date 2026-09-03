import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import pandas as pd

module_path = '../'
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath("../Voice-Privacy-Challenge-2024/")
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath("../Voice-Privacy-Challenge-2024/anonymization/modules/sttts/speaker_embeddings/anonymization/utils")
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)
print(os.listdir(module_path))

from data_utils.load_data import load_df
from data_utils.prepare_data import save_to_kaldi_format

from extract_train_embeddings import extract_train_embeddings


def print_info(data):
    for df_name, df in data.items():
        print(f"{df_name} has {len(df)} utterances from {len(set(df['speaker_id'].values))} speakers. Emotions: {set(df['emotion'].values)}")

def corpora_to_pickle(pickle_filename="../../workspace/datasets-ravdess-esd-libritts.pkl"):
    # Loads ESD, RAVDESS and LibriTTS (full, not only train)
    # Saves to dict of corpora to pickle_filename
    logging.info(f"Loading corpora...")
    ravdess_df = load_df(corpus_name="RAVDESS")
    libritts_df = load_df(corpus_name="LibriTTS", load_transcripts=True) #loading transcripts take long -- many individual files
    esd_df = load_df(corpus_name="ESD")

    data = {
        "RAVDESS": ravdess_df,
        "ESD": esd_df,
        "LibriTTS": libritts_df,
    }

    print_info(data)

    with open(pickle_filename, "wb") as f:
        pickle.dump(data, f)

    logging.info(f"Saved dict data to {pickle_filename}.")

def prepare_merged_df(data, shuffle=True):
    dfs = []

    for df_name, df in data.items():
        df['prefix'] = df_name
        dfs.append(df[['utterance_id', 'speaker_id', 'transcript', 'audio_path', 'emotion', 'gender', 'prefix']])

    combined = pd.concat(dfs, ignore_index=True)
    if shuffle:
        return combined.sample(frac=1, random_state=42).reset_index(drop=True) #shuffles rows
    return combined

def merge_df_to_pickle(
        infile="../../workspace/datasets-ravdess-esd-libritts.pkl", 
        outfile="../../workspace/ravdess-esd-librittstrain_df.pkl",
        partition="TRAIN"):
    # Selects data from corpora (ESD, RAVDESS, LibriTTS)
    # Keeps only clean LibriTTS train
    # Merges several data sources into one dataframe
    # Pickles the merged dataframe and saves on disk
    pickle_filename = infile
    try:
        with open(pickle_filename, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Loaded dict data from {pickle_filename}.")
    except FileNotFoundError:
        logging.error(f"Could not locate {pickle_filename}!")
        corpora_to_pickle(pickle_filename)
        with open(pickle_filename, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Loaded dict data from {pickle_filename}.")

    if partition == "TRAIN":
        subsets=['train-clean-360', 'train-clean-100']
        libritts_df_train = data["LibriTTS"].query(f"subset.isin({subsets})")
        data = {
            "RAVDESS": data["RAVDESS"],
            "ESD": data["ESD"],
            "LibriTTS_train": libritts_df_train,
        }
    elif partition == "DEV":
        subsets=['dev-clean']
        libritts_df_dev = data["LibriTTS"].query(f"subset.isin({subsets})")
        data = {
            "LibriTTS_dev": libritts_df_dev,
        }
    print_info(data)
    logging.info("Bringing everything into one df...")
    all_df = prepare_merged_df(data)
    logging.info(f"Prepared df with {len(all_df)} rows.")
    with open(outfile, "wb") as f:
        pickle.dump(all_df, f)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess data partitions.")
    parser.add_argument(
        "--partition",
        type=str,
        default="TRAIN",
        choices=["TRAIN", "DEV", "TEST"],
        help="Specify which partition to preprocess (default: TRAIN)."
    )
    parser.add_argument(
        "--calc_embeddings",
        action="store_true",
        help="If set, calculate embeddings after preprocessing."
    )
    args = parser.parse_args()

    if args.partition == 'TRAIN':
        print("Preprocessing TRAIN partition...")
        INFILE = "../../workspace/datasets-ravdess-esd-libritts.pkl"
        MERGED_DF_PATH = "../../workspace/ravdess-esd-librittstrain_df.pkl"
        DATASET_PATH = "../../workspace/voice_reference_dataset"
        KALDI_DIR = os.path.join(DATASET_PATH, "kaldi")
        EMBEDDING_DIR = os.path.join(DATASET_PATH, "embeddings")
    elif args.partition == 'DEV':
        print("Preprocessing DEV partition...")
        INFILE = "../../workspace/datasets-ravdess-esd-libritts.pkl"
        MERGED_DF_PATH = "../../workspace/LibriTTS/evaluation/libritts_dev_df.pkl"
        DATASET_PATH = "../../workspace/LibriTTS/evaluation"
        KALDI_DIR = os.path.join(DATASET_PATH, "kaldi")
        EMBEDDING_DIR = os.path.join(DATASET_PATH, "embeddings")

    else:
        raise NotImplementedError

    try:
        with open(MERGED_DF_PATH, "rb") as f:
            all_df = pickle.load(f)
    except FileNotFoundError:
        merge_df_to_pickle(infile=INFILE, outfile=MERGED_DF_PATH, partition=args.partition)
        with open(MERGED_DF_PATH, "rb") as f:
            all_df = pickle.load(f)

    save_to_kaldi_format(all_df, working_dir=DATASET_PATH)
    logging.info(f"Loaded and saved {args.partition} data")

    if args.calc_embeddings:
        logging.info(f"Calculating embeddings...")
        gpu_id = 1
        emb_model_path = "../../workspace/models/tts/Embedding/embedding_function.pt" #2.0
        # emb_model_path = "../../workspace/models/embedding_function0.pt" #GST model? https://github.com/DigitalPhonetics/speaker-anonymization/releases/tag/v2.0
        # emb_model_path = "../../workspace/models/embedding_function2.5.pt" #https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_function.pt
        vec_type = "style-embed" # "ecapa"
        emb_level = "utt" # "spk" 

        extract_train_embeddings(KALDI_DIR, EMBEDDING_DIR, gpu_id, emb_model_path, vec_type, emb_level)
    
    print("Done!")


if __name__ == "__main__":
    main()