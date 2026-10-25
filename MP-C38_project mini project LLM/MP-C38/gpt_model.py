"""GPT-2 architecture written from scratch (course Ch 3-4) + loading of pretrained GPT-2 124M weights (Ch 5)
+ new classification head (Ch 6). No Hugging Face model classes are used; the HF hub is only used to
download the weight file."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

GPT2_SMALL = dict(vocab_size=50257, context_length=1024, emb_dim=768, n_heads=12,
                  n_layers=12, drop_rate=0.1, qkv_bias=True)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]; self.h = cfg["n_heads"]
        self.W_query = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(d, d)
        self.drop = cfg["drop_rate"]

    def forward(self, x):
        b, t, d = x.shape
        split = lambda m: m(x).view(b, t, self.h, d // self.h).transpose(1, 2)
        q, k, v = split(self.W_query), split(self.W_key), split(self.W_value)
        # causal mask: each token only sees earlier tokens (so right-padding never leaks into real tokens)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.drop if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(b, t, d))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.att = MultiHeadAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        return x + self.drop(self.ff(self.norm2(x)))


class GPTModel(nn.Module):
    """GPT-2 backbone. out_dim=None -> language-model head (vocab). out_dim=10 -> classification head."""
    def __init__(self, cfg, out_dim=None):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], out_dim or cfg["vocab_size"], bias=out_dim is not None)

    def forward(self, ids):                       # ids: [B, T] -> logits [B, T, out_dim]
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.drop_emb(self.tok_emb(ids) + self.pos_emb(pos))
        return self.out_head(self.final_norm(self.trf_blocks(x)))

    def classify_logits(self, ids, lengths):
        """One vector of class scores per clause, read at the last REAL (non-pad) token."""
        logits = self(ids)
        idx = (lengths - 1).clamp(min=0).to(logits.device)
        return logits[torch.arange(ids.shape[0], device=logits.device), idx]


def load_gpt2_weights(model, repo_id="openai-community/gpt2"):
    """Download GPT-2 124M weights (safetensors) and copy them into OUR GPTModel."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = {k.replace("transformer.", ""): v for k, v in load_file(hf_hub_download(repo_id, "model.safetensors")).items()}
    d = model.cfg["emb_dim"]
    with torch.no_grad():
        model.tok_emb.weight.copy_(sd["wte.weight"]); model.pos_emb.weight.copy_(sd["wpe.weight"])
        for i, blk in enumerate(model.trf_blocks):
            p = f"h.{i}."
            w, b = sd[p + "attn.c_attn.weight"].T, sd[p + "attn.c_attn.bias"]   # HF stores Conv1D transposed
            for j, lin in enumerate([blk.att.W_query, blk.att.W_key, blk.att.W_value]):
                lin.weight.copy_(w[j * d:(j + 1) * d]); lin.bias.copy_(b[j * d:(j + 1) * d])
            blk.att.out_proj.weight.copy_(sd[p + "attn.c_proj.weight"].T); blk.att.out_proj.bias.copy_(sd[p + "attn.c_proj.bias"])
            blk.ff[0].weight.copy_(sd[p + "mlp.c_fc.weight"].T); blk.ff[0].bias.copy_(sd[p + "mlp.c_fc.bias"])
            blk.ff[2].weight.copy_(sd[p + "mlp.c_proj.weight"].T); blk.ff[2].bias.copy_(sd[p + "mlp.c_proj.bias"])
            blk.norm1.weight.copy_(sd[p + "ln_1.weight"]); blk.norm1.bias.copy_(sd[p + "ln_1.bias"])
            blk.norm2.weight.copy_(sd[p + "ln_2.weight"]); blk.norm2.bias.copy_(sd[p + "ln_2.bias"])
        model.final_norm.weight.copy_(sd["ln_f.weight"]); model.final_norm.bias.copy_(sd["ln_f.bias"])
    return model


def build_classifier(num_classes, cfg=GPT2_SMALL, pretrained=True, unfreeze_blocks=1):
    """Pretrained GPT-2 + NEW trainable head. Freeze everything, then unfreeze the last
    `unfreeze_blocks` transformer blocks, the final LayerNorm and the head (course Ch 6)."""
    model = GPTModel(cfg)
    if pretrained:
        load_gpt2_weights(model)
    model.out_head = nn.Linear(cfg["emb_dim"], num_classes)       # new head, random init
    for p in model.parameters():
        p.requires_grad = False
    for blk in list(model.trf_blocks)[len(model.trf_blocks) - unfreeze_blocks:]:
        for p in blk.parameters(): p.requires_grad = True
    for p in list(model.final_norm.parameters()) + list(model.out_head.parameters()):
        p.requires_grad = True
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total, sum(p.numel() for p in model.parameters() if p.requires_grad)
