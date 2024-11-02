import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt


def load_labels(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', usecols=['ID', 'Label'])
    invalid_labels = df[~df['Label'].isin(['VIOLATION DETECTED', 'NO VIOLATION DETECTED'])]
    if not invalid_labels.empty:
        print("Invalid labels found:")
        print(invalid_labels)
    return df.set_index('ID')['Label']


def compute_metrics(pred_file, gold_file, content_type, method):
    pred_labels = load_labels(pred_file)
    gold_labels = load_labels(gold_file)
    common_ids = pred_labels.index.intersection(gold_labels.index)
    pred_labels = pred_labels.loc[common_ids]
    gold_labels = gold_labels.loc[common_ids]
    gold_binary = gold_labels.map({'SENSITIVE': 1, 'NOT SENSITIVE': 0})
    pred_binary = pred_labels.map({'VIOLATION DETECTED': 1, 'NO VIOLATION DETECTED': 0})

    accuracy = accuracy_score(gold_binary, pred_binary)
    precision = precision_score(gold_binary, pred_binary)
    recall = recall_score(gold_binary, pred_binary)
    f1 = f1_score(gold_binary, pred_binary)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    precision_vals, recall_vals, _ = precision_recall_curve(gold_binary, pred_binary)
    os.makedirs("precision_recall_curve", exist_ok=True)
    plot_filename = f"precision_recall_curve/{content_type}_{method}_precision_recall_curve.png"

    plt.figure()
    plt.plot(recall_vals, precision_vals, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({content_type} - {method})")
    plt.savefig(plot_filename)
    plt.close()


def run_metrics_for_content_type(content_type):
    print(f"\n\n***** {content_type.upper()} *****")
    base_pred_path = f"results/{content_type}/enron_mail_annotated_gpt4o_full_{{}}.tsv"
    gold_file = "enron_mail_3k_annotated_gpt35_full.tsv"

    for method in ["zero_shot", "FAISS", "BM25"]:
        pred_file = base_pred_path.format(method)
        print(f"\n{method.capitalize()}")
        compute_metrics(pred_file, gold_file, content_type, method)


# Example usage for different content types
run_metrics_for_content_type("abs")
run_metrics_for_content_type("intros")
run_metrics_for_content_type("abs_intros")
