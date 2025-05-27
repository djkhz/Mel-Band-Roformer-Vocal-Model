# Vocal Cleanup Pipeline: RoFormer + Denoiser + TTS Prep

import os
from pydub import AudioSegment
import librosa
import soundfile as sf
from audio_separator.separator import Separator

# === Step 1: Separate Vocals with RoFormer ===

def separate_vocals(input_path, output_dir, model_config, model_ckpt):
    separator = Separator(config_path=model_config, model_path=model_ckpt)
    separator.separate(input_path, output_dir)
    return os.path.join(output_dir, "vocals.wav")

# === Step 2: (Optional) Apply simple denoising ===
# For real use, replace this with DeepFilterNet or external tool

def simple_lowpass_denoise(input_wav, output_wav, cutoff=8000):
    y, sr = librosa.load(input_wav, sr=None)
    # Simple low-pass filtering
    y_fft = librosa.stft(y)
    freq = librosa.fft_frequencies(sr=sr)
    y_fft[freq > cutoff, :] = 0
    y_filtered = librosa.istft(y_fft)
    sf.write(output_wav, y_filtered, sr)
    return output_wav

# === Step 3: Convert to MMS-TTS Format ===

def convert_to_mms_tts(input_wav, output_wav):
    audio = AudioSegment.from_wav(input_wav)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)
    audio.export(output_wav, format="wav")
    return output_wav

# === Example Usage ===

INPUT_AUDIO = "input_audio/song.wav"
MODEL_CONFIG = "configs/config_vocals_mel_band_roformer.yaml"
MODEL_CKPT = "MelBandRoformer.ckpt"
OUTPUT_DIR = "output_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Vocal Separation
vocals_path = separate_vocals(INPUT_AUDIO, OUTPUT_DIR, MODEL_CONFIG, MODEL_CKPT)

# 2. (Optional) Denoise
vocals_cleaned_path = os.path.join(OUTPUT_DIR, "vocals_cleaned.wav")
simple_lowpass_denoise(vocals_path, vocals_cleaned_path)

# 3. Convert to MMS-TTS Format
final_tts_ready_path = os.path.join(OUTPUT_DIR, "vocals_mms_ready.wav")
convert_to_mms_tts(vocals_cleaned_path, final_tts_ready_path)

print("✅ MMS-TTS-ready file saved at:", final_tts_ready_path)
