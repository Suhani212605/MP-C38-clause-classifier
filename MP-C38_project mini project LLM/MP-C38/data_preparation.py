"""Prepare the balanced 10-class LEDGAR subset (LexGLUE) and split 70/15/15."""
import json, re, argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split


def normalise_key(text):
    """Key for duplicate detection: lowercase, letters only (ignores digits/punctuation/spacing)."""
    return re.sub(r"[^a-z]+", "", str(text).lower())


def build_balanced(df, n_classes=10, per_class=800, seed=42):
    """df has columns text,label_name. Returns (balanced_df, info dict). Pure pandas -> easy to test."""
    df = df.copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= 20]
    df["key"] = df["text"].map(normalise_key)
    n_before = len(df)
    df = df.drop_duplicates("key")                       # removes duplicates BEFORE splitting -> no leakage
    # a text that appears under two different labels is ambiguous -> already collapsed by drop_duplicates
    counts = df["label_name"].value_counts()
    top = list(counts.index[:n_classes])
    n_use = int(min(per_class, counts[top].min()))
    parts = [df[df.label_name == c].sample(n=n_use, random_state=seed) for c in top]
    bal = pd.concat(parts).reset_index(drop=True)
    label_map = {c: i for i, c in enumerate(sorted(top))}
    bal["label"] = bal["label_name"].map(label_map)
    info = dict(examples_before_dedup=n_before, examples_after_dedup=len(df),
                top_classes_counts={c: int(counts[c]) for c in top}, per_class_used=n_use,
                shortfall=n_use < per_class, label_map=label_map, seed=seed)
    return bal[["text", "label", "label_name"]], info


def split_data(bal, seed=42):
    train, rest = train_test_split(bal, test_size=0.30, stratify=bal["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_class", type=int, default=800)
    a = ap.parse_args()
    import datasets
    try:
        ds = datasets.load_dataset("coastalcph/lex_glue", "ledgar", trust_remote_code=True)
    except Exception:
        ds = datasets.load_dataset("coastalcph/lex_glue", "ledgar", revision="refs/convert/parquet")
    print(ds)                                            # inspect splits / fields
    names = ds["train"].features["label"].names
    pool = pd.concat([ds[s].to_pandas() for s in ds.keys()])   # pool all splits, we re-split ourselves
    pool["label_name"] = pool["label"].map(lambda i: names[i])
    bal, info = build_balanced(pool[["text", "label_name"]], per_class=a.per_class, seed=a.seed)
    train, val, test = split_data(bal, a.seed)
    os.makedirs(a.out, exist_ok=True)
    for n, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(f"{a.out}/{n}.csv", index=False)
    info.update(original_splits={s: len(ds[s]) for s in ds.keys()}, original_num_labels=len(names),
                split_sizes=dict(train=len(train), val=len(val), test=len(test)),
                source="https://huggingface.co/datasets/coastalcph/lex_glue (config: ledgar)",
                datasets_version=datasets.__version__,
                selection_rule="10 most frequent labels after pooling all splits and removing duplicates")
    json.dump(info, open(f"{a.out}/data_info.json", "w"), indent=2)
    json.dump(info["label_map"], open(f"{a.out}/label_map.json", "w"), indent=2)
    bal["label_name"].value_counts().to_csv(f"{a.out}/class_distribution.csv")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
