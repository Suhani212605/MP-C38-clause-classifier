"""Fine-tune GPT-2 124M with a new 10-class head. Saves best checkpoint (by validation macro-F1)."""
import json, time, argparse, os
import pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from gpt_model import build_classifier, count_params, GPT2_SMALL
from utils import ClauseDataset, set_seed, get_device
from evaluate import run_eval, metrics, save_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--unfreeze", type=int, default=2, help="number of last transformer blocks to unfreeze")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="models/best.pt")
    a = ap.parse_args()
    set_seed(a.seed); dev = get_device(); os.makedirs("models", exist_ok=True); os.makedirs("report_assets", exist_ok=True)
    label_map = json.load(open(f"{a.data}/label_map.json")); names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    tr, va, te = [pd.read_csv(f"{a.data}/{s}.csv") for s in ("train", "val", "test")]
    mk = lambda d, sh: DataLoader(ClauseDataset(d.text, d.label, a.max_len), batch_size=a.batch, shuffle=sh)
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(ClauseDataset(tr.text, tr.label, a.max_len), batch_size=a.batch, shuffle=True, generator=g)
    vl, sl = mk(va, False), mk(te, False)

    model = build_classifier(len(names), GPT2_SMALL, pretrained=True, unfreeze_blocks=a.unfreeze).to(dev)
    total, trainable = count_params(model); print(f"params total={total:,} trainable={trainable:,} device={dev}")
    res = dict(total_params=total, trainable_params=trainable, device=str(dev), seed=a.seed, settings=vars(a))

    # accuracy BEFORE fine-tuning (pretrained GPT-2 + untrained random head) on the test set
    p, y, l = run_eval(model, sl, dev); res["before_test"] = {**metrics(y, p), "test_loss": l}; print("before:", res["before_test"])

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=a.wd)
    hist, best, t0 = [], -1, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); run, n, correct = 0.0, 0, 0
        for ids, lengths, y in tl:
            ids, y = ids.to(dev), y.to(dev)
            logits = model.classify_logits(ids, lengths)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(y); n += len(y); correct += (logits.argmax(-1) == y).sum().item()
        vp, vy, vloss = run_eval(model, vl, dev); vm = metrics(vy, vp)
        hist.append(dict(epoch=ep, train_loss=run / n, train_acc=correct / n, val_loss=vloss, val_acc=vm["accuracy"], val_macro_f1=vm["macro_f1"]))
        print(hist[-1])
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            torch.save(dict(state_dict=model.state_dict(), cfg=GPT2_SMALL, class_names=names, label_map=label_map,
                            max_length=a.max_len, seed=a.seed), a.out)
    res["train_seconds"] = time.time() - t0; res["history"] = hist; res["best_val_macro_f1"] = best

    # reload best checkpoint and evaluate ONCE on the held-out test set
    ck = torch.load(a.out, map_location=dev); model.load_state_dict(ck["state_dict"])
    p, y, l = run_eval(model, sl, dev); res["after_test"] = {**metrics(y, p), "test_loss": l}; print("after:", res["after_test"])
    save_report(y, p, names, te.text.tolist())
    json.dump(res, open("report_assets/results.json", "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]; plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train_loss"] for h in hist], "o-", label="train"); plt.plot(ep, [h["val_loss"] for h in hist], "s-", label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.legend(); plt.title("Training and validation loss"); plt.tight_layout()
    plt.savefig("report_assets/loss_curve.png", dpi=150)
    tr["label_name"].value_counts().plot(kind="barh", figsize=(7, 4), title="Training-set class distribution"); plt.tight_layout()
    plt.savefig("report_assets/class_distribution.png", dpi=150)


if __name__ == "__main__":
    main()
