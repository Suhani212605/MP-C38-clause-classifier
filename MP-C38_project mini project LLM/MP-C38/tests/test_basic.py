import os, sys, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import split_into_clauses, results_to_csv, extract_text
from data_preparation import build_balanced, split_data

SAMPLE = """1. Governing Law
This Agreement shall be governed by the laws of India and the courts of Bengaluru shall have jurisdiction.

2. Notices
All notices under this Agreement shall be in writing and delivered by hand or registered post to the parties.

(a) Counterparts. This Agreement may be signed in counterparts, each of which is an original document."""


def test_segmentation_keeps_order():
    c = split_into_clauses(SAMPLE)
    assert len(c) == 3 and "governed" in c[0] and "notices" in c[1].lower() and "Counterparts" in c[2]


def test_segmentation_no_numbering():
    txt = "Para one is long enough to be a clause on its own, really.\n\nPara two is also long enough to stand alone as a clause."
    assert len(split_into_clauses(txt)) == 2


def test_extract_unsupported(tmp_path):
    f = tmp_path / "x.docx"; f.write_text("hi")
    with pytest.raises(ValueError): extract_text(str(f))


def test_csv_export():
    s = results_to_csv([dict(clause_no=1, clause_text="a, b", predicted_type="X")])
    assert s.splitlines()[0] == "clause_no,clause_text,predicted_type"


def _fake():
    w = lambda n: "".join(chr(97 + (n // 26 ** k) % 26) for k in range(3))   # letters only (digits are ignored by the dedupe key)
    rows = [(f"clause text {w(i)} of type {w(c * 7 + 1000)} with words {w(i * 31 + c)}", f"L{c}") for c in range(12) for i in range(60)]
    rows += [rows[0]]                                         # a duplicate
    return pd.DataFrame(rows, columns=["text", "label_name"])


def test_balanced_and_split_integrity():
    bal, info = build_balanced(_fake(), per_class=50, seed=1)
    assert bal.label.nunique() == 10 and (bal.label.value_counts() == 50).all()
    tr, va, te = split_data(bal, 1)
    assert not (set(tr.text) & set(te.text)) and not (set(tr.text) & set(va.text)) and not (set(va.text) & set(te.text))
    assert len(tr) == 350 and len(va) == len(te) == 75


def test_model_shapes_and_batch_prediction():
    torch = pytest.importorskip("torch"); pytest.importorskip("tiktoken")
    from gpt_model import GPTModel
    from utils import ClauseDataset, pad_batch, encode, PAD_ID
    cfg = dict(vocab_size=50257, context_length=64, emb_dim=32, n_heads=2, n_layers=2, drop_rate=0.0, qkv_bias=True)
    m = GPTModel(cfg, out_dim=10).eval()
    ds = ClauseDataset(["short clause", "a much longer clause " * 40], [0, 1], max_length=16)
    ids, lengths, y = ds[1]
    assert ids.shape[0] == 16 and lengths.item() == 16            # truncation
    assert ds[0][0][-1].item() == PAD_ID                           # padding
    b, l = pad_batch([encode("one two", 16), encode("three", 16), encode("four five six", 16)])
    assert m.classify_logits(b, l).shape == (3, 10)                # one score vector (10) per clause
