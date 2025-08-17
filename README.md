# Mitigating Bias for Non-native speakers in Automatic Speech Recognition

The Repository is the code and analysis for MSc thesis *Mitigating Bias for Non-native speakers in Automatic Speech Recognition* at Technical University of Denmark.

Project Supervisor: 
- Sneha Das (sned@dtu.dk)
- Line Clemmensen (lkhc@dtu.dk)

## Abstract 
> Automatic Speech Recognition (ASR) systems are increasingly used in a wide range of real-world applications. However, these systems often fail to perform equitably across all users, exhibiting notable performance disparities based on demographic factors such as gender, race, and accent. In particular, non-native speakers tend to experience higher error rates, a problem which remains unsolved in current ASR research. These disparities, often referred to as bias, highlight the need for more inclusive and robust ASR systems.
>
> This thesis aims to narrow the gap by investigating and mitigating bias in ASR performance between native (L1) and non-native (L2) speaker groups on the Whisper-small model. We conduct a comprehensive evaluation using multiple metrics to quantify disparities. Results reveal notable differences between L1 and L2 speakers: 4.086 in Word Error Rate (WER), 6.845 in Word Information Lost (WIL), and 5.003 in Semantic Distance (SemDist). These disparities correlate with factors such as linguistic distance, age of language acquisition, and speech speed.
>
> To reduce these performance gaps, this thesis explores three fine-tuning and bias mitigation strategies: partial fine-tuning with a Learning without Forgetting (LwF) constraint and interpolation of bases adaptation at the final layer, low-rank adaptation (LoRA) with similar interpolation, and full model freezing with LEAstsquares Concept Erasure (LEACE) applied at the final layer. Although none of the methods show statistically significant improvements across all languages and metrics, both the adapter-based and LEACE approaches demonstrate potential in narrowing the L1–L2 performance gap without substantially degrading native speaker performance when compared to the baseline Whisper-small model and purely fine-tuned approaches.

## Code Structure
- `CustomData`: Prepocessing Data into vector format.
- `DatasetLoad`: Preprocess and load dataset for finetune.
- `Evaluate`: Optimization functions for finetune ASR.
- `Models`: Implementation different mitigation methods in Whisper model.
- `Refer`: Code from repo (https://github.com/tonywu71/distilling-and-forgetting-in-large-pre-trained-models) for different trainer.
- `plot`: Results Plots.
- `utils`: Some functions and parameters for use.
- `google.py`: 
- `Finetune.py`: Script for finetuning Whisper model.
- `Transcribe_eval.py`: Script for transcribing and evaluating performance.
- `ablation_transcribe.py`: Transcribe audio while skipping one layer from the start to last.
- `ablation_eval.py`: Evaluate Performance for transcripts from `ablation_transcribe.py`.

## Start
1. Clone and move to the project directory
```bash
    git clone https://github.com/tinghui8576/Mitigating-Bias-for-nonnative-speakers-in-Automatic-Speech-Recognition & cd Mitigating-Bias-for-nonnative-speakers-in-Automatic-Speech-Recognition
```
2. Install the required dependencies
```bash
    pip install -r requirements.txt
```

## Dataset
| **Dataset**  | **Access**                                                                                    |
|--------------|-----------------------------------------------------------------------------------------------|
| EdAcc        | https://huggingface.co/datasets/edinburghcstr/edacc                                           |
| L2-ARCTIC    | https://psi.engr.tamu.edu/l2-arctic-corpus/                                                   |
| CommonVoice  | https://commonvoice.mozilla.org/en/datasets                                                   |
| Speech Accent Archive  | https://accent.gmu.edu/                                                             |

### For CustomData
Some custom data (i.e. speech accent archive) need to be process into the Dataset format.
See SSA_pathtext.py for an example of how to preprocess the data into desired format. 

After `audio_paths` `text` `accent` files are created. Prepare the data ino arrow file using
```python
python data_prep.py --source_data_dir [PATH TO SOURCE DIR] --output_data_dir [PATH TO STORE ARROW FILE]
```

### For Finetuning 
