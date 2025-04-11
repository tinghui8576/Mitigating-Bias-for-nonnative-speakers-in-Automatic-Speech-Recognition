import os
import csv

path = '/Users/tinghui/Downloads/cv-corpus-5.1-2020-06-22/en'
tsvs = ["train.tsv", "dev.tsv"]
audio_entries = []
text_entries = []
for tsv in tsvs:
    tsv_path = os.path.join(path, tsv)
    with open(tsv_path , "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            filename = row.get("path")  # get the mp3 filename
            if filename:
                file_path = os.path.join(path, "clips", filename)
                if os.path.exists(file_path):
                    utt_id = os.path.splitext(filename)[0]
                    transcript = row.get("sentence")
                    text_entries.append(f"{utt_id} {transcript}")
                    audio_entries.append(f"{utt_id} {os.path.abspath(file_path)}")
                    

with open("commonvoice/audio_paths", "w", encoding="utf-8") as f:
    f.write("\n".join(audio_entries))

with open("commonvoice/text", "w", encoding="utf-8") as f:
    f.write("\n".join(text_entries))   