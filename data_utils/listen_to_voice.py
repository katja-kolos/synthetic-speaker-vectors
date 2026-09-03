import datetime
import os
import sys
import uuid

import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.io.wavfile import write

module_path = os.path.abspath("../Voice-Privacy-Challenge-2024")
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath("../Voice-Privacy-Challenge-2024/anonymization")
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath("../Voice-Privacy-Challenge-2024/anonymization/modules/sttts/tts")
if module_path not in sys.path:
    sys.path.append(module_path)

from anonymization.modules.sttts.tts.ims_tts import ImsTTS


def _save_audio(path: str, waveform: np.ndarray, sample_rate: int = 16000):
    """
    Save a waveform numpy array as a .wav file.
    """
    # Convert to int16 for standard WAV
    scaled = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
    write(path, sample_rate, scaled)
    print(f"Saved audio to {path}")

def _save_mel_spectrogram_to_image(path: str, mel_spec_np: np.ndarray):
    '''Input: spectrogram (n_mels x time_frames) and saving path'''

    # Plot and save
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec_np, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.tight_layout()

    # Save the image
    plt.savefig(path)
    plt.close()


def listen_to_voice(speaker_vector: torch.Tensor, 
                    utterance: str = 'Artificial intelligence is the intelligence exhibited by machines and software.',
                    hifigan_path: str ='../../workspace/models/tts/HiFiGAN_combined/best.pt',
                    fastspeech_path: str ='../../workspace/models/tts/FastSpeech2_Multi/prosody_cloning.pt',
                    embedding_path: str = '../../workspace/models/tts/Embedding/embedding_function.pt',
                    device: str ='cuda:3',

                    do_save: str | None = None, #or: 'wav', 'mel', 'wav+mel'
                    prefix: str = '',
                    save_audio_dir: str = '../../workspace/audio_samples/',
                    save_mel_dir: str = '../../workspace/audio_samples/',
                    do_suffix_with_id: bool = True):
    """
        Function to preview a speaker embedding.
        Args:
            speaker_vector (torch.Tensor): valid GST embedding of size 128
            utterance (str, optional): sentence in English to pronounce. Defaults to a short sentence from Wikipedia.
        Returns:
            audio wave (numpy array), 16 kHz
        ----
        Note: models were downloaded from https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/tts.zip
        as advised in https://github.com/DigitalPhonetics/speaker-anonymization/tree/prosody_cloning?tab=readme-ov-file.
        2.0 is also the version of embedding_function we used to obtain our pool of GST vectors
        Note 2: the branch is called prosody cloning as it supports prosody cloning; we, however, are not doing it.
    """
    assert type(speaker_vector) == torch.Tensor, "Speaker vector must be a torch.Tensor."
    assert speaker_vector.shape[0] == 128, "Expecting a valid GST embedding of size 128."

    tts = ImsTTS(hifigan_path, fastspeech_path, device, embedding_path=embedding_path, output_sr=16000, lang='en')

    wave, mel = tts.read_text(utterance, speaker_vector, text_is_phones=False, duration=None, pitch=None, energy=None,
                  start_silence=None, end_silence=None)
    
    if do_save is not None:
        if do_suffix_with_id:
            # Generate timestamp + short uid
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            short_uid = str(uuid.uuid4())[:6]
            file_id = f"{prefix}{timestamp}_{short_uid}"
        else:
            file_id = f"{prefix}"

        if 'mel' in do_save:
            mel = mel.cpu().detach().numpy()
            mel_spectrogram_filename = os.path.join(save_mel_dir, f"{file_id}.png")
            _save_mel_spectrogram_to_image(mel_spectrogram_filename, mel)

        if 'wav' in do_save:
            wav_filename = os.path.join(save_audio_dir, f"{file_id}.wav")
            _save_audio(wav_filename, wave)

    return wave