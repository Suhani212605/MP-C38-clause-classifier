"""Evaluation: loss/accuracy/macro metrics, per-class report, confusion matrix, error examples."""
import json, argparse
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval(); preds, labels, tot = [], [], 0.0
    for ids, lengths, y in loader:
        logits = model.classify_logits(ids.to(device), lengths)
        tot += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        preds += logits.argmax(-1).cpu().tolist(); labels += y.tolist()
    return np.array(preds), np.array(labels), tot / len(labels)


def metrics(labels, preds):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(labels, preds), macro_precision=p, macro_recall=r, macro_f1=f)


def save_report(labels, preds, names, texts, out="report_assets", tag="test"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rep = classification_report(labels, preds, target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"{out}/{tag}_per_class.csv")
    cm = confusion_matrix(labels, preds); pd.DataFrame(cm, index=names, columns=names).to_csv(f"{out}/{tag}_confusion.csv")
    fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(f"{out}/confusion_matrix.png", dpi=150); plt.close(fig)
    wrong = [dict(text=t[:300], expected=names[l], predicted=names[p]) for t, l, p in zip(texts, labels, preds) if l != p]
    pd.DataFrame(wrong).to_csv(f"{out}/errors.csv", index=False)


def main():
    from inference import ClauseClassifier
    from utils import ClauseDataset, get_device
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="models/best.pt"); ap.add_argument("--data", default="data")
    a = ap.parse_args()
    clf = ClauseClassifier(a.ckpt); test = pd.read_csv(f"{a.data}/test.csv")
    dl = DataLoader(ClauseDataset(test.text, test.label, clf.max_length), batch_size=32)
    preds, labels, loss = run_eval(clf.model, dl, clf.device)
    m = metrics(labels, preds); m["test_loss"] = loss
    save_report(labels, preds, clf.class_names, test.text.tolist())
    prev = json.load(open("report_assets/results.json")) if __import__("os").path.exists("report_assets/results.json") else {}
    prev["after_test"] = m; json.dump(prev, open("report_assets/results.json", "w"), indent=2); print(m)


if __name__ == "__main__":
    main()
