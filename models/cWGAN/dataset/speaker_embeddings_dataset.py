import os
from pathlib import Path

import numpy as np
import torch


class SpeakerVectorGenderDataset2(torch.utils.data.Dataset):

    def __init__(self, feature_path: Path, device, normalize_data: bool = True):
        super(SpeakerVectorGenderDataset2, self).__init__()

        self.device = device

        self.normalize_data = normalize_data

        self.x, self.gender_labels = self._load_features(feature_path)
        
        self.x = self._reformat_features(self.x)
        self.y = self.gender_labels #just alias
        
    def _reformat_features(self, features):
        if len(features.shape) == 2:
            return features.reshape(features.shape[0], 1, 1, features.shape[1])
    
    def normalize_embedding(self, vector):
        return torch.sub(vector, self.mean) / self.std
    
    def get_embedding_dim(self):
        return self.x.shape[-1]
        
    def _load_features(self, feature_path: Path):
        # Load utterance-level speaker vectors & labels
        
        if os.path.isfile(feature_path):
            raise NotImplementedError("Please give full dataset folder (we'll find the kaldi folder there)")

        feature_path = Path(feature_path)
        # entire folder should be given as path, load not only the vectors, but also the labels
        embeddings_folder = feature_path / 'embeddings/reference_embeddings/utt-level'
        # my additional file (not standard kaldi) with metadata (gender, emotion)
        utt2gender_path = feature_path / 'kaldi' / 'utt2gender'
        
        # we first load the vectors...
        vectors = torch.load(embeddings_folder / 'speaker_vectors.pt', map_location=self.device)
        if isinstance(vectors, list):
                vectors = torch.stack(vectors)
        self.mean = torch.mean(vectors)
        self.std = torch.std(vectors)
        # ... then load the actual ids of the speakers 
        id2idx = {}
        with open(embeddings_folder / f'id2idx', 'r') as f: #load utterance id2idx
            for line in f:
                split_line = line.strip().split()
                if len(split_line) == 2:
                    id2idx[split_line[0].strip()] = int(split_line[1])

        ids, indices = zip(*id2idx.items())
        print(f"Loaded ids and indices, example: {ids[0]}, {indices[0]}")

        # ... finally, load gender labels by utterance
        if (utt2gender_path).exists():
            utt2gender = {}
            with open(utt2gender_path, 'r') as f:
                for line in f:
                    split_line = line.strip().split()
                    if len(split_line) == 2:
                        utt2gender[split_line[0].strip()] = split_line[1].strip()

            gender_labels = [utt2gender[utt] for utt in ids]
            map_of_labels = {'m': 0, 'f': 1} # same as in train_classifier.ipynb
            gender_labels = [map_of_labels[label] for label in gender_labels]

        return vectors[np.array(indices)], gender_labels
    
    def __getitem__(self, index):
        embedding = self.x[index]
        if self.normalize_data:
            embedding = self.normalize_embedding(embedding)
        speaker_gender_label = self.gender_labels[index]
        return embedding, speaker_gender_label
    
    def __len__(self): # different from IMS WGAN: number of utterances!
        return len(self.x)