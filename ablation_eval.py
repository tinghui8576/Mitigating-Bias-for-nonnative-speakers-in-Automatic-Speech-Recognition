import os
import torch
import pandas as pd
from tqdm import tqdm
from utils.eval import metrics
from utils.constant import sentence


# Load reference sentences
reference_df = pd.read_csv("Data/speech-accent-archive/bio.csv")
reference_df.rename(columns={"filename": "id"}, inplace=True)
reference_df["accent"] = reference_df["native_language"].apply(lambda x: x.split("\n")[0])

# =============================
# EVALUATION FUNCTION
# =============================

def evaluate_predictions(prediction_csv, output_csv):
    # Load predictions
    preds_df = pd.read_csv(prediction_csv)
    # Merge with reference
    merged_df = pd.merge(preds_df, reference_df, on="id", how="left")

    results = []
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc=f"Evaluating {os.path.basename(prediction_csv)}"):
        pred_text = row["prediction"]

        # Skip empty prediction
        if not isinstance(pred_text, str) or len(pred_text.strip()) == 0:
            continue
        if len(pred_text) > 700:
            continue
        performance = metrics(pred_text, sentence)

        # Compute metrics
        results.append({
            "id": row["id"],
            "WER": performance['WER'],
            "CER": performance['CER'],
            "MER": performance['MER'],
            "WIL": performance['WIL'],
            "Ember": performance['Ember'],
            "SemDist": performance['SemDist'],
            "Accent": row["accent"]
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved evaluation results to {output_csv}")
    return results_df

# =============================
# MAIN ENTRY POINT
# =============================

if __name__ == "__main__":
    import glob

    # Change this pattern to match your files:
    prediction_files = sorted(glob.glob("results/ablation/leace/ablation*.csv"))

    all_eval_results = []
    for pred_csv in prediction_files:
        layer_name = os.path.splitext(os.path.basename(pred_csv))[0]
        print(layer_name)
        eval_csv = f"results/ablation/leace/eval_{layer_name}.csv"
        df = evaluate_predictions(pred_csv, eval_csv)
        # Optionally collect for summary
        df["Layer"] = layer_name
        all_eval_results.append(df)
