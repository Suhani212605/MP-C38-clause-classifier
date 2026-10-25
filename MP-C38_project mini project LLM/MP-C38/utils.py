"""Shared helpers: seed, device, tokenizer, Dataset, text extraction, clause splitting, CSV export."""
import io, re, random
import numpy as np
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset

PAD_ID = 50256                      # GPT-2 <|endoftext|> token, used for padding (course Ch 6)
TOKENIZER = tiktoken.get_encoding("gpt2")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(text, max_length):
    """Text -> token ids, truncated to max_length. Same function is used in training AND inference."""
    return TOKENIZER.encode(str(text), allowed_special=set())[:max_length]


class ClauseDataset(Dataset):
    """Course-style Dataset: tokenizes, truncates, pads every clause to the same length with PAD_ID.
    Also returns the true (unpadded) length so the model can read the last REAL token."""
    def __init__(self, texts, labels, max_length=256):
        self.encoded = [encode(t, max_length) or [PAD_ID] for t in texts]
        self.labels = list(labels)
        self.max_length = max(len(e) for e in self.encoded)
        self.max_length = min(self.max_length, max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        ids = self.encoded[i]
        padded = ids + [PAD_ID] * (self.max_length - len(ids))
        return torch.tensor(padded), torch.tensor(len(ids)), torch.tensor(self.labels[i])


def pad_batch(token_lists):
    """Pad a list of token-id lists to one tensor (used at inference time)."""
    lengths = [max(len(t), 1) for t in token_lists]
    width = max(lengths)
    ids = [(t or [PAD_ID]) + [PAD_ID] * (width - max(len(t), 1)) for t in token_lists]
    return torch.tensor(ids), torch.tensor(lengths)


# ---------- contract reading and clause splitting ----------
def extract_text(path):
    """Read text from a .txt or .pdf file. Raises ValueError with a readable message."""
    path = str(path)
    low = path.lower()
    try:
        if low.endswith(".txt"):
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")
        if low.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if not text.strip():
                raise ValueError("No selectable text found in the PDF (it may be a scanned image).")
            return text
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read the file: {e}")
    raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")


# a new clause usually starts with: 1. / 1.1 / (a) / Section 3 / ARTICLE II
_CLAUSE_START = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+\S|\(\w{1,4}\)\s+\S|(?:section|article|clause)\s+\w+|(?-i:[A-Z][A-Z \-]{3,40})\.?\s*$)",
    re.IGNORECASE)


def split_into_clauses(text, min_chars=40, max_chars=1500):
    """Split contract text into clauses (original order kept).
    Rules: numbered/lettered headings start a clause; otherwise blank lines separate paragraphs;
    very short pieces are merged with the next piece; very long pieces are cut at sentence ends."""
    text = text.replace("\r", "")
    # PDF extraction can break a line (even a blank line) in the middle of a sentence: join those pieces
    text = re.sub(r"(?<![.;:!?)\"\u201d])\n\s*\n(?=\S)", " ", text)
    # numbered clauses that ended up inside one line ("... effect. 5. Entire Agreement ...") start a new line
    text = re.sub(r"(?<=[.;:])[ \t]+(?=\d{1,2}(?:\.\d+)*\.\s+[A-Z])", "\n", text)
    lines = text.split("\n")
    blocks, cur = [], []
    for line in lines:
        if not line.strip():                       # blank line = paragraph boundary
            if cur: blocks.append(" ".join(cur)); cur = []
            continue
        if _CLAUSE_START.match(line) and cur:      # numbering / heading = new clause
            blocks.append(" ".join(cur)); cur = []
        cur.append(line.strip())
    if cur: blocks.append(" ".join(cur))
    blocks = [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]

    merged, carry = [], ""                        # merge tiny pieces (e.g. lone headings) forward
    for b in blocks:
        b = (carry + " " + b).strip() if carry else b
        if len(b) < min_chars: carry = b
        else: merged.append(b); carry = ""
    if carry: merged.append(carry) if not merged else merged.__setitem__(-1, merged[-1] + " " + carry)

    clauses = []
    for b in merged:                              # cut very long blocks at sentence boundaries
        if len(b) <= max_chars: clauses.append(b); continue
        piece = ""
        for s in re.split(r"(?<=[.;])\s+", b):
            if len(piece) + len(s) > max_chars and piece:
                clauses.append(piece.strip()); piece = ""
            piece += " " + s
        if piece.strip(): clauses.append(piece.strip())
    return clauses


def results_to_csv(rows):
    """rows: list of dicts -> CSV string (clause order preserved)."""
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()
