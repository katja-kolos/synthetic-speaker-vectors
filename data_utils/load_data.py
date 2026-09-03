import logging
import pandas as pd
import os

# RAVDESS
def parse_ravdess_filename(filename):
    filename = os.path.basename(filename)
    parts = filename.split('-')
    return {
        "modality": int(parts[0]),
        "vocal_channel": int(parts[1]),
        "emotion": int(parts[2]),
        "intensity": int(parts[3]),
        "transcript": int(parts[4]), # "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "speaker_id": int(parts[6].split('.')[0]), # "actor": int(parts[6].split('.')[0]),
    }

def ravdess_assign_gender(actor_id):
    return 'M' if actor_id % 2 else 'F'
def ravdess_assign_modality(modality_id):
    modality_id -= 1
    mapping = ['full-AV', 'video-only', 'audio_only']
    return mapping[modality_id]
def ravdess_assign_vocal_channel(vocal_channel_id):
    vocal_channel_id -= 1
    mapping = ['speech', 'song']
    return mapping[vocal_channel_id]
def ravdess_assign_emotion(emotion_id):
    emotion_id -= 1
    mapping = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    return mapping[emotion_id]
def ravdess_assign_transcript(statement_id):
    statement_id -= 1
    mapping = ['Kids are talking by the door', 'Dogs are sitting by the door']
    return mapping[statement_id]


def parse_ravdess(root_dir: str) -> pd.DataFrame:
    """
    Parse RAVDESS dataset structure.
    
    Expected structure (RAVDESS):
        root_dir/
            Actor_01
                03-01-01-01-01-01-01.wav
                03-01-01-01-01-02-01.wav
                ...
    Returns:
        pd.DataFrame with columns:
        ['modality', 'vocal_channel', 'emotion', 'intensity', 'transcript', 'repetition', 'speaker_id', 'audio_path', 'gender']
    """
     # Filename identifiers 
    # - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    # - Vocal channel (01 = speech, 02 = song).
    # - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    # - Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    # - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    # - Repetition (01 = 1st repetition, 02 = 2nd repetition).
    # - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


    data = []
    for root, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                file_metadata = parse_ravdess_filename(path)
                file_metadata["audio_path"] = path
                utterance_id = os.path.basename(path.replace(".wav", ""))
                file_metadata["utterance_id"] = utterance_id
                data.append(file_metadata)
                
    df = pd.DataFrame(data)
    
    
    df['gender'] = df['speaker_id'].map(ravdess_assign_gender)
    df['emotion'] = df['emotion'].map(ravdess_assign_emotion)
    df['vocal_channel'] = df['vocal_channel'].map(ravdess_assign_vocal_channel)
    df['modality'] = df['modality'].map(ravdess_assign_modality)
    df['transcript'] = df['transcript'].map(ravdess_assign_transcript)
    return df

# Emotional Speech Dataset
# note: this map is based on my subjective perception of the speaker's gender (most clear on 'Happy' audios)
# some speakers sound "gender neutral" to me (e.g. 0014, 0012), while some have a clear gender (0015, 0016, 0017, 0018 -- female, 0020 -- male)
# TODO: contact dataset authors and ask for gender information, if available
esd_gender_map = {
    "0011": "M",
    "0012": "M",
    "0013": "M",
    "0014": "M",
    "0015": "F",
    "0016": "F",
    "0017": "F",
    "0018": "F",
    "0019": "F",
    "0020": "M",
} 
def parse_esd(root_dir: str) -> pd.DataFrame:
    """
    Parse ESD dataset structure.
    Expected structure:
        root_dir/
            0011 (speaker 0011; speakers 0001-0010: Mandarin; speakers 0011-0020: English)
                Angry
                    0011_000351.wav
                    0011_000352.wav
                    ...
                Happy
                    0011_000701.wav
                    ...
                Neutral
                Sad
                Surprise
                0011.txt (all transcripts of speaker 0011)
                fixed_unicode.txt (all transcripts pf speaker 0011, postprocessed at the IMS)
    Returns:
        pd.DataFrame with columns:
        ['emotion', 'transcript', 'speaker_id', 'audio_path', 'gender']
    """
    data = []
    for speaker in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
        logging.info(f"Loading speaker: {speaker}")
        speaker_id = int(speaker)
        if not (11 <= speaker_id <= 20):
            continue  # Skip non-English speakers

        # fetch the transcripts file
        transcript_file_name = "fixed_unicode.txt"
        # transcript_file_name = f"{speaker}.txt"
        transcripts = {}
        with open(os.path.join(speaker_path, transcript_file_name), "r") as f:
            lines = f.readlines()
            logging.info('Read transcript file')
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) == 3:
                utterance_id, transcript, emotion = parts[0], parts[1], parts[2]
                _, utt_id = utterance_id.split('_')
                utterance_id = f"{speaker}_{utt_id.strip()}"
                transcripts[(utterance_id, emotion.lower())] = transcript

        logging.debug(f"Collected {len(transcripts)} transcripts. Example:")
        for k, v in transcripts.items():
            logging.debug(f"{k}: {v}")
            break
            
        # iterate emotion subfolders
        for emotion in os.listdir(speaker_path):
            emotion_folder = os.path.join(speaker_path, emotion)
            if not os.path.isdir(emotion_folder):
                continue
            emotion = emotion.lower()

            for audio in os.listdir(emotion_folder):
                utterance_audio_path = os.path.join(emotion_folder, audio)
                if not os.path.isfile(utterance_audio_path):
                    continue
                if not utterance_audio_path.endswith(".wav"):
                    continue

                utterance_id = f"{audio}".replace(".wav", "")
                save_utterance_id = f"{emotion}_{utterance_id}"
                transcript = transcripts.get((utterance_id, emotion), None)
                data.append({
                    "utterance_id": save_utterance_id,
                    "emotion": emotion,
                    "transcript": transcript,
                    "speaker_id": speaker,
                    "audio_path": utterance_audio_path,
                    "gender": esd_gender_map.get(speaker, None)
                })
            
    return pd.DataFrame(data)


# LibriTTS
def parse_libritts(root_dir: str, transcript_version='.normalized', load_transcripts=False) -> pd.DataFrame:
    """
    Parse LibriTTS dataset structure.
    If load_transcripts is set to True, 
        the `transcript` field will contain utterance text for each utterance (.normalized or .original), based on the transcript version.
        This means _much_ longer loading, which only makes sense for ASR, but not speaker embedding task.
    Otherwise, the `transcript` field will contain utterance transcript path.
    
    Expected structure (LibriTTS-full, LibriTTS):
        root_dir/
            train-clean-100/
                19/
                  198/
                    19_198_000000_000001.wav
                    19_198_000000_000001.normalized.txt
                    19_198_000000_000001.original.txt
                    ...
    
    Returns:
        pd.DataFrame with columns:
        ['utterance_id', 'speaker_id', 'chapter_id', 'subset', 'audio_path', 'transcript']
    """

    data = []
    for subset in os.listdir(root_dir):
        subset_path = os.path.join(root_dir, subset)
        if not os.path.isdir(subset_path):
            continue
        
        for speaker in os.listdir(subset_path):
            speaker_path = os.path.join(subset_path, speaker)
            if not os.path.isdir(speaker_path):
                continue

            logging.info(f"Loading speaker: {speaker}")
            
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if not os.path.isdir(chapter_path):
                    continue
                
                for utterance_audio_path in os.listdir(chapter_path):
                    if utterance_audio_path.endswith(".wav"):
                        utterance_id = utterance_audio_path.replace(".wav", "")
                        transcript_path = os.path.join(chapter_path, f'{utterance_id}{transcript_version}.txt')
                        if load_transcripts: 
                            # this step takes very long
                            with open(transcript_path, 'r') as f:
                                transcript = f.read()
                        else:
                            transcript = transcript_path
                        data.append({
                            "utterance_id": utterance_id,
                            "speaker_id": int(speaker), #string
                            "chapter_id": chapter,
                            "subset": subset,
                            "audio_path": os.path.join(chapter_path, utterance_audio_path),
                            # "transcript_path": transcript_path,
                            "transcript": transcript
                        })
    return pd.DataFrame(data)


def load_librivox_speaker_metadata(meta_file: str) -> pd.DataFrame:
    """
    Load speaker metadata file (SPEAKERS.txt on the IMS server).
    
    Expected format:
        ID  | SEX | SUBSET | MINUTES | NAME
    Note: first 11 rows start with ; and contain text comments about the corpus (originally, LibriVox)
    
    Returns:
        pd.DataFrame with ['speaker_id', 'gender', 'subset', 'minutes', 'name']
    """
    rows = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                continue
            rows.append({
                "speaker_id": int(parts[0]), 
                "gender": parts[1],
                "subset": parts[2],
                "minutes": float(parts[3]),
                "name": parts[4]
            })
    return pd.DataFrame(rows)


# VCTK
def parse_vctk(root_dir: str) -> pd.DataFrame:
    # TODO
    df = pd.DataFrame()
    return df

def load_vctk_speaker_metadata(meta_file: str) -> pd.DataFrame:
    pass


# Unification
def load_df(corpus_name: str, parent_path: str ='../../data/', **kwargs) -> pd.DataFrame:
    """
        Input: corpus name (normally a folder in parent_path), parent_path
        Chooses appropriate data loader
        Returns: pandas df with mandatory fields: ['speaker_id', 'gender', 'emotion', 'audio_path', 'transcript'] and optionally some other fields
    """
    if corpus_name == 'LibriTTS':
        corpus_dir = os.path.join(parent_path, 'LibriTTS-full')
        audio_df = parse_libritts(corpus_dir, **kwargs)
        speaker_metadata_df = load_librivox_speaker_metadata(os.path.join(corpus_dir, 'SPEAKERS.txt'))
        df = audio_df.merge(speaker_metadata_df[['speaker_id', 'gender']], how='left', on=['speaker_id'])
        df['emotion'] = pd.NA

        assert all([col in df.columns for col in ['speaker_id', 'gender', 'emotion', 'audio_path', 'transcript']])
        return df

    elif corpus_name == 'RAVDESS':
        corpus_dir = os.path.join(parent_path, 'RAVDESS')
        df = parse_ravdess(root_dir=corpus_dir, **kwargs)

        assert all([col in df.columns for col in ['speaker_id', 'gender', 'emotion', 'audio_path', 'transcript']])
        return df
    
    elif corpus_name == 'VCTK':
        corpus_dir = os.path.join(parent_path, 'VCTK')
        audio_df = parse_vctk(root_dir=corpus_dir, **kwargs)
        speaker_metadata_df = load_vctk_speaker_metadata(meta_file=os.path.join(corpus_dir, 'speaker-info.txt'))
        df = audio_df.merge(speaker_metadata_df[['speaker_id', 'gender']], how='left', on=['speaker_id'])
        df['emotion'] = pd.NA

        assert all([col in df.columns for col in ['speaker_id', 'gender', 'emotion', 'audio_path', 'transcript']])
        return df

    elif corpus_name == 'ESD':
        corpus_dir = os.path.join(parent_path, 'ESD')
        df = parse_esd(root_dir=corpus_dir)
        assert all([col in df.columns for col in ['speaker_id', 'gender', 'emotion', 'audio_path', 'transcript']])
        return df
    else:
        raise NotImplementedError