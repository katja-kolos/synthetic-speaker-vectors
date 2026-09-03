# Adapted from: 
# https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/anonymization/modules/sttts/speaker_embeddings/anonymization/
# Dataset structure used for training the unconditional WGAN model.

import os
from pathlib import Path

import numpy as np
import torch


class SpeakerEmbeddingsDataset(torch.utils.data.Dataset):

    def __init__(self, feature_path, device, mode='utterance', normalize_data: bool = True):
        super(SpeakerEmbeddingsDataset, self).__init__()

        self.device = device

        self.normalize_data = normalize_data
        feature_path = Path(feature_path)
        self.x, self.speakers = self._load_features(feature_path)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, index):
        embedding = self.x[index]
        if self.normalize_data:
            embedding = self.normalize_embedding(embedding)
        return embedding, torch.zeros([0])

    def normalize_embedding(self, vector):
        return torch.sub(vector, self.mean) / self.std

    def get_speaker(self, label):
        return self.class2spk[label]

    def get_embedding_dim(self):
        return self.x.shape[-1]

    def get_num_speaker(self):
        return len(torch.unique((self.y)))

    def set_labels(self, labels):
        self.y_old = self.y # copy the original labels in case they are needed later
        self.y = torch.full(size=(len(self),), fill_value=labels).to(self.device)

    def _load_features(self, feature_path):
        
        # directly pass speaker vectors file to work with speaker_vectors.pt only, use case doesn't care about labels
        # load it only and return: the vectors and empty zero labels
        if os.path.isfile(feature_path):
            vectors = torch.load(feature_path, map_location=self.device)
            if isinstance(vectors, list):
                vectors = torch.stack(vectors)

            self.mean = torch.mean(vectors)
            self.std = torch.std(vectors)
            return vectors, torch.zeros(vectors.size(0))

        # pass folder (containing speaker vectors and id2idx file) 
        # to load not only the vectors, but also the labels (in unconditional WGAN, speaker ids)
        else:
            embeddings_folder = feature_path / 'embeddings/reference_embeddings/utt-level'
            vectors = torch.load(embeddings_folder / 'speaker_vectors.pt', map_location=self.device)

            self.mean = torch.mean(vectors)
            self.std = torch.std(vectors)

            spk2idx = {}
            with open(embeddings_folder / f'id2idx', 'r') as f: #remove self.mode -- we always have id2idx
                for line in f:
                    split_line = line.strip().split()
                    if len(split_line) == 2:
                        spk2idx[split_line[0].strip()] = int(split_line[1])

            speakers, indices = zip(*spk2idx.items())

            if (embeddings_folder / 'utt2spk').exists():  # spk2idx contains utt_ids not speaker_ids
                utt2spk = {}
                with open(feature_path / 'utt2spk', 'r') as f:
                    for line in f:
                        split_line = line.strip().split()
                        if len(split_line) == 2:
                            utt2spk[split_line[0].strip()] = split_line[1].strip()

                speakers = [utt2spk[utt] for utt in speakers]

            return vectors[np.array(indices)], speakers

    def _reformat_features(self, features):
        if len(features.shape) == 2:
            return features.reshape(features.shape[0], 1, 1, features.shape[1])