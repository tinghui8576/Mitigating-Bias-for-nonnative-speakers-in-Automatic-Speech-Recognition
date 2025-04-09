from google.cloud import speech
import os
import librosa
import soundfile as sf
import io
import csv
import pandas as pd

output_csv = "ssa_google.csv"

# Create the file with headers if it doesn't exist
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "transcript"])  # header

# def transcribe_file(audio_file: str) -> speech.RecognizeResponse:
#     """Transcribe the given audio file.
#     Args:
#         audio_file (str): Path to the local audio file to be transcribed.
#             Example: "resources/audio.wav"
#     Returns:
#         cloud_speech.RecognizeResponse: The response containing the transcription results
#     """

#     # Load and resample to 16kHz
#     audio, sr = librosa.load(audio_file, sr=16000)

#     # Save to in-memory buffer as WAV
#     buffer = io.BytesIO()
#     sf.write(buffer, audio, 16000, format='WAV')
#     buffer.seek(0)

#     client = speech.SpeechClient()

#     # Load content from buffer
#     audio_content = buffer.read()

#     audio = speech.RecognitionAudio(content=audio_content)
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="en-US",
#     )

#     response = client.recognize(config=config, audio=audio)

#     # Each result is for a consecutive portion of the audio. Iterate through
#     # them to get the transcripts for the entire audio file.
#     sentences = [result.alternatives[0].transcript for result in response.results]
#     full_transcript = " ".join(sentences)
#     return full_transcript


def transcribe_audio_chunk(audio_chunk, sr):
    """Transcribe a chunk of audio using in-memory buffer."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_chunk, sr, format='WAV')
    buffer.seek(0)

    client = speech.SpeechClient()
    audio_content = buffer.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    return " ".join([r.alternatives[0].transcript for r in response.results])

def transcribe_file(audio_file: str, max_duration=60):
    """Handles long audio by chunking if needed."""
    audio, sr = librosa.load(audio_file, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    if total_duration <= max_duration:
        return transcribe_audio_chunk(audio, sr)
    else:
        print(f"Chunking {audio_file} into two parts")
        midpoint = int(len(audio) / 2)
        first_half = audio[:midpoint]
        second_half = audio[midpoint:]

        transcript1 = transcribe_audio_chunk(first_half, sr)
        transcript2 = transcribe_audio_chunk(second_half, sr)
        return transcript1 + " " + transcript2

path = 'Data/speech-accent-archive'
audio_dir = os.path.join(path, 'audio')
audio_files = os.listdir(audio_dir)
meta_saa = pd.read_csv(os.path.join(path, 'bio.csv'))
meta_saa['native_language'] = meta_saa['native_language'].apply(lambda x: x.split('\n')[0])
meta_saa['sex'] = meta_saa['sex'].replace('famale', 'female')
language_counts = meta_saa["native_language"].value_counts()

languages_to_keep = language_counts[language_counts >=  30].index

meta_saa = meta_saa[meta_saa['native_language'].isin(languages_to_keep)]

for _, row in meta_saa.iterrows():
    audio_file = f"{row['filename']}.wav"
    file_path = os.path.join(audio_dir, audio_file)
    print(file_path)
    if os.path.exists(file_path):
        print(f"Transcribing {file_path} for language: {row['native_language']}")
        try:
            transcript = transcribe_file(file_path)

            # Save the result immediately
            with open(output_csv, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([row["filename"], transcript])
            
            print(f"✅ Saved transcript for {row['filename']}")

        except Exception as e:
            print(f"⚠️ Error transcribing {file_path}: {e}")