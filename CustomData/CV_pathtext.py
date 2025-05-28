import os
import csv

path = '/Users/tinghui/Downloads/cv-corpus-5.1-2020-06-22/en'
tsvs = ["train.tsv", "dev.tsv"]
tsv_dict = {
    "train": ["train.tsv", "dev.tsv"],
    "test": ["test.tsv"]
}
for key, tsvs in tsv_dict.items():
    audio_entries = []
    text_entries = []
    accent_entries = []
    for tsv in tsvs:
        tsv_path = os.path.join(path, tsv)
        with open(tsv_path , "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                filename = row.get("path")  # get the mp3 filename
                transcript = row.get("sentence")
                accent = row.get("accent")
                if filename and transcript and accent and accent.strip() and accent.strip().lower() != "other":
                    file_path = os.path.join(path, "clips", filename)
                    if os.path.exists(file_path):
                        utt_id = os.path.splitext(filename)[0]
                        text_entries.append(f"{utt_id} {transcript}")
                        audio_entries.append(f"{utt_id} {os.path.abspath(file_path)}")
                        accent_entries.append(f"{utt_id} {accent}")
                        

    with open(f"commonvoice/{key}/audio_paths", "w", encoding="utf-8") as f:
        f.write("\n".join(audio_entries))
    with open(f"commonvoice/{key}/text", "w", encoding="utf-8") as f:
        f.write("\n".join(text_entries))  
    with open(f"commonvoice/{key}/accent", "w", encoding="utf-8") as f:
        f.write("\n".join(accent_entries))   

# tsvs = ["test.tsv"]

# audio_entries = []
# text_entries = []
# for tsv in tsvs:
#     tsv_path = os.path.join(path, tsv)
#     with open(tsv_path , "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f, delimiter="\t")
#         for row in reader:
#             filename = row.get("path")  # get the mp3 filename
#             transcript = row.get("sentence")
#             accent = row.get("accent")
#             if filename and transcript and accent and accent.strip() and accent.strip().lower() != "other":
#                 file_path = os.path.join(path, "clips", filename)
#                 if os.path.exists(file_path):
#                     utt_id = os.path.splitext(filename)[0]
#                     text_entries.append(f"{utt_id} {transcript} {accent}")
#                     audio_entries.append(f"{utt_id} {os.path.abspath(file_path)}")
                    

# with open("test/audio_paths", "w", encoding="utf-8") as f:
#     f.write("\n".join(audio_entries))

# with open("test/text", "w", encoding="utf-8") as f:
#     f.write("\n".join(text_entries))   
