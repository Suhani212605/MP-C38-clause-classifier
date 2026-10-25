"""Gradio app: single-clause classification + contract upload. Loads models/best.pt (no retraining)."""
import os, tempfile
import gradio as gr
from inference import ClauseClassifier
from utils import extract_text, split_into_clauses, results_to_csv

CKPT = os.environ.get("CKPT", "models/best.pt")
clf = ClauseClassifier(CKPT) if os.path.exists(CKPT) else None
DISCLAIMER = ("**Academic NLP prototype - not legal advice.** Predictions may be wrong and must be reviewed by a "
              "qualified person. Confidence is the model's softmax probability, not legal certainty.")
EXAMPLES = ["This Agreement shall be governed by and construed in accordance with the laws of the State of New York.",
            "This Agreement may be executed in counterparts, each of which shall be deemed an original.",
            "If any provision of this Agreement is held invalid, the remaining provisions shall remain in full force."]


def need_model():
    if clf is None:
        raise gr.Error(f"Model checkpoint '{CKPT}' not found. Run train.py first.")


def predict_single(text):
    need_model()
    if not text or not text.strip():
        raise gr.Error("Please enter a clause first.")
    top = clf.predict_batch([text], top_k=3)[0]
    return top[0][0], f"{top[0][1]:.1%} (model confidence)", {k: v for k, v in top}


def predict_contract(file):
    need_model()
    if file is None:
        raise gr.Error("Please upload a PDF or TXT contract.")
    try:
        clauses = split_into_clauses(extract_text(file if isinstance(file, str) else file.name))
    except ValueError as e:
        raise gr.Error(str(e))
    if not clauses:
        raise gr.Error("No clauses could be extracted from this file.")
    preds = clf.predict_batch(clauses, top_k=1)
    rows = [dict(clause_no=i + 1, clause_text=c, predicted_type=p[0][0], confidence=round(p[0][1], 4))
            for i, (c, p) in enumerate(zip(clauses, preds))]
    path = os.path.join(tempfile.mkdtemp(), "clause_predictions.csv")
    open(path, "w", encoding="utf-8").write(results_to_csv(rows))
    table = [[r["clause_no"], r["clause_text"][:300], r["predicted_type"], f"{r['confidence']:.1%}"] for r in rows]
    return table, path


with gr.Blocks(title="Contract Clause Type Identifier") as demo:
    gr.Markdown("# Contract Clause Type Identifier\nGPT-2 (124M) fine-tuned on 10 LEDGAR clause types. " + DISCLAIMER)
    gr.Markdown("**Supported clause types:** " + (", ".join(clf.class_names) if clf else "(train the model first)"))
    with gr.Tab("Single clause"):
        box = gr.Textbox(lines=6, label="Contract clause")
        gr.Examples(EXAMPLES, box, label="Illustrative examples (predictions still come from the trained model)")
        btn = gr.Button("Predict", variant="primary")
        lab, conf, top3 = gr.Textbox(label="Predicted clause type"), gr.Textbox(label="Confidence"), gr.Label(label="Top 3")
        btn.click(predict_single, box, [lab, conf, top3])
    with gr.Tab("Upload contract"):
        up = gr.File(label="PDF or TXT", file_types=[".pdf", ".txt"])
        go = gr.Button("Classify all clauses", variant="primary")
        tbl = gr.Dataframe(headers=["#", "Clause", "Predicted type", "Confidence"], wrap=True)
        dl = gr.File(label="Download CSV")
        go.click(predict_contract, up, [tbl, dl])

if __name__ == "__main__":
    demo.launch()
