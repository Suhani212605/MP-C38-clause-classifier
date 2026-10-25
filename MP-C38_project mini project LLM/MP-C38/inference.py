"""Loads the saved checkpoint once and exposes clause_type(text) / batch prediction.
Uses the same tokenizer, truncation, padding and label map as training."""
import torch
from gpt_model import GPTModel
from utils import encode, pad_batch, get_device


class ClauseClassifier:
    def __init__(self, ckpt_path="models/best.pt"):
        self.device = get_device()
        ck = torch.load(ckpt_path, map_location=self.device)
        self.class_names, self.max_length = ck["class_names"], ck["max_length"]
        self.model = GPTModel(ck["cfg"], out_dim=len(self.class_names))
        self.model.load_state_dict(ck["state_dict"]); self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_batch(self, texts, top_k=3, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            ids, lengths = pad_batch([encode(t, self.max_length) for t in texts[i:i + batch_size]])
            probs = torch.softmax(self.model.classify_logits(ids.to(self.device), lengths), dim=-1).cpu()
            for p in probs:
                v, ix = p.topk(min(top_k, len(self.class_names)))
                out.append([(self.class_names[j], float(s)) for s, j in zip(v, ix)])
        return out                                  # per clause: [(label, prob), ...] best first

    def clause_type(self, text):
        """Main function: clause text -> predicted clause type."""
        return self.predict_batch([text])[0][0][0]
