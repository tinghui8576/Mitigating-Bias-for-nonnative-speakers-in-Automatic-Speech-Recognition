import os
path = 'l2arctic_release_v5.0'
audio_dirs = os.listdir(path)
audio_entries = []
text_entries = []


file_accent = {"ABA":"Arabic","SKA":"Arabic","YBAA":"Arabic","ZHAA":"Arabic","BWC":"Chinese",
                "BWC":"Chinese","LXC":"Chinese","NCC":"Chinese","TXHC":"Chinese",
                "ASI":"Hindi","RRBI":"Hindi","SVBI":"Hindi","TNI":"Hindi",
                "HJK":"Korean","HKK":"Korean","YDCK":"Korean","YKWK":"Korean",
                "EBVS":"Spanish","ERMS":"Spanish","MBMPS":"Spanish","NJS":"Spanish",
                "HQTV":"Vietnamese","PNV":"Vietnamese","THV":"Vietnamese","TLV":"Vietnamese"}
for dirpath, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith(".wav"):
            audio_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(dirpath, path)
            utt_id = "_".join([rel_path.split(os.sep)[0]] + [os.path.splitext(filename)[0]])
            transcript_path = os.path.join(path, rel_path.split(os.sep)[0], "transcript", os.path.splitext(filename)[0]+'.txt')
            if os.path.exists(transcript_path):
                accent = file_accent[dirpath.split('/')[1]]
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                    text_entries.append(f"{utt_id} {transcript} {accent}")
                    audio_entries.append(f"{utt_id} {os.path.abspath(audio_path)}")


with open("L2/audio_paths", "w", encoding="utf-8") as f:
    f.write("\n".join(audio_entries))

with open("L2/text", "w", encoding="utf-8") as f:
    f.write("\n".join(text_entries))              