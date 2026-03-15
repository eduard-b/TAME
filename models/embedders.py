from __future__ import annotations
import torch.nn as nn
import time
import torch
import numpy as np
from typing import Any, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------
# 1. Simple BN embedder (baseline)
# ------------------------------------------------------
class EmbedderBN(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------
# 2. Deeper BN embedder (3 BN blocks)
# ------------------------------------------------------
class EmbedderBNDeep(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------
# 3. Wide BN embedder (1024 → 512 → 256 → embed_dim)
# ------------------------------------------------------
class EmbedderBNWide(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------
# 4. Residual BN embedder
# ------------------------------------------------------
class BNResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x
        out = self.bn1(self.fc1(x))
        out = F.relu(out)
        out = self.bn2(self.fc2(out))
        out = out + identity
        return F.relu(out)

class EmbedderBNRes(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128, num_blocks=2):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden)
        self.bn_in = nn.BatchNorm1d(hidden)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(BNResBlock(hidden))
        self.blocks = nn.Sequential(*blocks)

        self.fc_out = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn_in(self.fc_in(x)))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x

# ------------------------------------------------------
# 5. BN Cascade Embedder (heavy BN stack)
# ------------------------------------------------------
class EmbedderBNCascade(nn.Module):
    def __init__(self, input_dim, hidden=512, embed_dim=128, depth=4):
        super().__init__()

        layers = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            prev = hidden

        layers.append(nn.Linear(hidden, embed_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ln(x))

# -----------------------------------------------
# 6. LN (2-layer MLP)
# -----------------------------------------------
class EmbedderLN(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# 8. LNDeep (3-layer)
# -----------------------------------------------
class EmbedderLNDeep(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# 9. LNWide (wider)
# -----------------------------------------------
class EmbedderLNWide(nn.Module):
    def __init__(self, input_dim, hidden=512, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, hidden),
            LNBlock(hidden),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------
# 10. LNRes (Residual MLP)
# -----------------------------------------------
class EmbedderLNRes(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.act = nn.ReLU()

        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)

        self.out = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.act(self.ln1(self.fc1(x)))
        h = h + self.act(self.ln2(self.fc2(h)))   # residual connection
        return self.out(h)

# -----------------------------------------------
# 11. LNCascade (maximal depth)
# -----------------------------------------------
class EmbedderLNCascade(nn.Module):
    def __init__(self, input_dim, hidden=256, embed_dim=128):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(6):  # 6 LN blocks
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            prev = hidden

        layers.append(nn.Linear(hidden, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------
# 12. LN ResMLP: deeper + wider
# ----------------------------

class _LNResBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * expansion
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h
# -----------------------------------------
# 13. NODE-style embedder: soft oblivious trees
# -----------------------------------------

def _leaf_bit_matrix(depth: int, device=None):
    """
    Returns bits matrix of shape (2^depth, depth) with entries in {0,1}
    where row i is the binary representation of i over 'depth' bits.
    """
    n_leaves = 2 ** depth
    ar = torch.arange(n_leaves, device=device).unsqueeze(1)  # (L,1)
    bits = (ar >> torch.arange(depth, device=device)) & 1    # (L,depth) little-endian
    return bits.float()


class ObliviousTreeEnsemble(nn.Module):
    """
    Differentiable oblivious trees (NODE-ish):
    - Each depth chooses a (soft) feature via softmax over input dims
    - Uses learnable thresholds and temperatures (alpha)
    - Computes leaf probabilities and mixes leaf values -> (B, tree_dim)
    """
    def __init__(
        self,
        input_dim: int,
        num_trees: int,
        depth: int,
        tree_dim: int,
        alpha_init: float = 5.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim

        # Feature selection logits: (T, D, input_dim)
        self.feature_logits = nn.Parameter(torch.zeros(num_trees, depth, input_dim))

        # Thresholds per tree+depth: (T, D)
        self.thresholds = nn.Parameter(torch.zeros(num_trees, depth))

        # Temperature / sharpness per tree+depth: (T, D), constrained positive via softplus
        self.alpha_unconstrained = nn.Parameter(torch.full((num_trees, depth), math.log(math.exp(alpha_init) - 1.0)))

        # Leaf values: (T, 2^D, tree_dim)
        self.leaf_values = nn.Parameter(torch.zeros(num_trees, 2 ** depth, tree_dim))

        # Small init helps stability
        nn.init.normal_(self.leaf_values, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (B, input_dim)
        returns: (B, tree_dim)
        """
        B, Din = x.shape
        assert Din == self.input_dim

        device = x.device
        bits = _leaf_bit_matrix(self.depth, device=device)  # (L, D)
        L = bits.shape[0]

        # Soft feature selection
        sel = F.softmax(self.feature_logits, dim=-1)  # (T, D, Din)

        # Selected feature value per tree+depth: (B, T, D)
        # einsum: (B,Din) x (T,D,Din) -> (B,T,D)
        x_sel = torch.einsum("bi,tdi->btd", x, sel)

        # Compute decision probs p in (0,1): (B,T,D)
        alpha = F.softplus(self.alpha_unconstrained) + 1e-6
        thr = self.thresholds
        p = torch.sigmoid((x_sel - thr.unsqueeze(0)) * alpha.unsqueeze(0))

        # Leaf probs for each tree: (B,T,L)
        # For leaf with bit=1 use p, else use (1-p)
        # Expand to (B,T,1,D) and (1,1,L,D)
        p_exp = p.unsqueeze(2)                 # (B,T,1,D)
        bits_exp = bits.view(1, 1, L, self.depth)
        probs = bits_exp * p_exp + (1.0 - bits_exp) * (1.0 - p_exp)  # (B,T,L,D)
        leaf_prob = probs.prod(dim=-1)         # (B,T,L)

        # Mix leaf values: (B,T,L) @ (T,L,tree_dim) -> (B,T,tree_dim)
        out = torch.einsum("btl,tld->btd", leaf_prob, self.leaf_values)

        # Sum over trees -> (B, tree_dim)
        return out.sum(dim=1)


class EmbedderNODE(nn.Module):
    """
    A NODE-style embedder built as stacked oblivious-tree ensembles with residuals.
    Produces (B, embed_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,      # not strictly needed; kept for compatibility with your factory
        embed_dim: int,
        num_layers: int = 2,
        num_trees: int = 64,
        depth: int = 6,
        tree_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if tree_dim is None:
            tree_dim = max(16, embed_dim // 2)

        self.in_proj = nn.Linear(input_dim, tree_dim)
        self.layers = nn.ModuleList([
            ObliviousTreeEnsemble(
                input_dim=tree_dim,
                num_trees=num_trees,
                depth=depth,
                tree_dim=tree_dim,
            )
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(tree_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(tree_dim, embed_dim)

    def forward(self, x):
        h = self.in_proj(x)  # (B, tree_dim)
        for layer in self.layers:
            # residual update (NODE-ish stacking)
            h = h + self.dropout(layer(self.ln(h)))
        return self.out_proj(self.ln(h))

class DCNv2CrossLayer(nn.Module):
    """
    DCNv2-style cross layer with low-rank factorization.

    x_{l+1} = x_l + x0 * (U (V x_l)) + b

    Shapes:
      x0, x_l: (B, d)
      V: (d, r) via Linear(d -> r)
      U: (r, d) via Linear(r -> d)
    """
    def __init__(self, d: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.v = nn.Linear(d, rank, bias=False)
        self.u = nn.Linear(rank, d, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, x):
        # (B, r) -> (B, d)
        cross = self.u(self.dropout(self.v(x)))
        return x + x0 * cross

class DCNv2Block(nn.Module):
    """
    Stack of DCNv2 cross layers.
    """
    def __init__(self, d: int, num_layers: int = 3, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DCNv2CrossLayer(d=d, rank=rank, dropout=dropout) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x0 = x
        h = x
        for layer in self.layers:
            h = layer(x0, h)
        return self.ln(h)

class EmbedderDCNv2MLP(nn.Module):
    """
    DCNv2 + MLP embedder:
      x -> in_proj -> DCNv2Block -> MLP -> LN -> out_proj(embed_dim)
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        embed_dim: int,
        cross_layers: int = 3,
        cross_rank: int = 16,
        mlp_depth: int = 2,
        mlp_expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden)

        self.cross = DCNv2Block(
            d=hidden,
            num_layers=cross_layers,
            rank=cross_rank,
            dropout=dropout,
        )

        mlp_inner = hidden * mlp_expansion
        mlp = []
        for _ in range(mlp_depth):
            mlp.append(nn.LayerNorm(hidden))
            mlp.append(nn.Linear(hidden, mlp_inner))
            mlp.append(nn.GELU())
            mlp.append(nn.Dropout(dropout))
            mlp.append(nn.Linear(mlp_inner, hidden))
        self.mlp = nn.Sequential(*mlp)

        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.cross(h)
        # residual MLP refinement
        h = h + self.mlp(h)
        h = self.out_ln(h)
        return self.out_proj(h)

def _round(x: float, multiple: int = 8, min_val: int = 8) -> int:
    v = int(round(x / multiple) * multiple)
    return max(v, min_val)

SIZE_LADDERS_LIGHT = {
    "ln_res_l": {
        "tiny":  dict(hidden_mul=0.5, depth=4,  expansion=2),
        "small": dict(hidden_mul=0.75, depth=6, expansion=3),
        "base":  dict(hidden_mul=1.0, depth=8, expansion=4),
        "large": dict(hidden_mul=1.5, depth=10, expansion=4),
        "xl":    dict(hidden_mul=2.0, depth=12, expansion=5),
    },
    "dcnv2_base": {
        "tiny":  dict(hidden_mul=0.5),
        "small": dict(hidden_mul=0.75),
        "base":  dict(hidden_mul=1.0),
        "large": dict(hidden_mul=1.5),
        "xl":    dict(hidden_mul=2.0),
    },
    "node": {
        "tiny":  dict(num_layers=1, num_trees=16, depth=4, tree_dim_mul=0.5),
        "small": dict(num_layers=1, num_trees=32, depth=5, tree_dim_mul=0.67),
        "base":  dict(num_layers=2, num_trees=64, depth=6, tree_dim_mul=1.0),
        "large": dict(num_layers=2, num_trees=96, depth=6, tree_dim_mul=1.33),
        "xl":    dict(num_layers=3, num_trees=128, depth=7, tree_dim_mul=2.0),
    },
}

SIZE_LADDERS = {
    "ln_res_l": {
        "tiny":  dict(hidden_mul=0.25, depth=2,  expansion=2),
        "small": dict(hidden_mul=0.5, depth=4, expansion=3),
        "base":  dict(hidden_mul=1.0, depth=8, expansion=4),
        "large": dict(hidden_mul=2, depth=16, expansion=4),
        "xl":    dict(hidden_mul=4.0, depth=32, expansion=5),
    },
    "dcnv2_base": {
        "tiny":  dict(hidden_mul=0.25),
        "small": dict(hidden_mul=0.5),
        "base":  dict(hidden_mul=1.0),
        "large": dict(hidden_mul=2.0),
        "xl":    dict(hidden_mul=4.0),
    },
    "node": {
        "tiny":  dict(num_layers=1, num_trees=16, depth=4, tree_dim_mul=0.25),
        "small": dict(num_layers=2, num_trees=32, depth=5, tree_dim_mul=0.5),
        "base":  dict(num_layers=4, num_trees=64, depth=6, tree_dim_mul=1.0),
        "large": dict(num_layers=8, num_trees=96, depth=6, tree_dim_mul=2.0),
        "xl":    dict(num_layers=8, num_trees=128, depth=7, tree_dim_mul=4.0),
    },
}

class EmbedderDCNv2Base(EmbedderDCNv2MLP):
    def __init__(self, input_dim: int, hidden: int, embed_dim: int):
        super().__init__(
            input_dim=input_dim,
            hidden=hidden,
            embed_dim=embed_dim,
            cross_layers=3,
            cross_rank=16,
            mlp_depth=2,
            mlp_expansion=2,
            dropout=0.0,
        )

class EmbedderAdapter(nn.Module):
    """
    Wraps a base embedder and maps its output to a common dim.
    """
    def __init__(self, base: nn.Module, base_out_dim: int, out_dim: int):
        super().__init__()
        self.base = base
        self.proj = nn.Identity() if base_out_dim == out_dim else nn.Linear(base_out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    @torch.no_grad()
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        z = self.base(x)
        z = self.proj(z)
        return self.ln(z)

class FusionEmbedder(nn.Module):
    """
    Runs several base embedders, concatenates their features, then compresses to out_dim.
    """
    def __init__(self, embedders: list[nn.Module], out_dims: list[int], out_dim: int):
        super().__init__()
        assert len(embedders) == len(out_dims)
        self.embedders = nn.ModuleList(embedders)
        self.out_dims = out_dims

        in_dim = sum(out_dims)
        self.compress = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    @torch.no_grad()
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feats = [e(x) for e in self.embedders]            # list[(B, d_i)]
        z = torch.cat(feats, dim=1)                      # (B, sum(d_i))
        return self.compress(z)                          # (B, out_dim)

class EmbedderLNResL(nn.Module):
    """
    Larger (but not XL) residual MLP:
    - increase depth a bit
    - keep expansion moderate
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        embed_dim: int,
        depth: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([_LNResBlock(hidden, expansion=expansion, dropout=dropout) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)
        return self.out_proj(h)

class EmbedderLNResSkip(nn.Module):
    """
    Residual MLP with extra interlayer skips:
    - standard resblocks
    - plus: accumulate a skip buffer and add it every `skip_every` blocks

    This often improves gradient flow and smooths optimization.
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        embed_dim: int,
        depth: int = 10,
        expansion: int = 4,
        dropout: float = 0.0,
        skip_every: int = 2,
        skip_scale: float = 0.5,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([_LNResBlock(hidden, expansion=expansion, dropout=dropout) for _ in range(depth)])
        self.skip_every = skip_every
        self.skip_scale = skip_scale
        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.in_proj(x)
        skip_buf = torch.zeros_like(h)

        for i, blk in enumerate(self.blocks, start=1):
            h = blk(h)
            skip_buf = skip_buf + h
            if (i % self.skip_every) == 0:
                h = h + self.skip_scale * skip_buf
                skip_buf = torch.zeros_like(h)

        h = self.out_ln(h)
        return self.out_proj(h)

class EmbedderLNResDense(nn.Module):
    """
    Dense-ish residual MLP:
    - Keep a small projection of each block output
    - Concatenate recent summaries and re-project back to hidden

    This increases interlayer connectivity without exploding compute.
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        embed_dim: int,
        depth: int = 10,
        expansion: int = 4,
        dropout: float = 0.0,
        summary_dim: int = 32,
        keep_last: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([_LNResBlock(hidden, expansion=expansion, dropout=dropout) for _ in range(depth)])

        self.summary_proj = nn.Linear(hidden, summary_dim)
        self.keep_last = keep_last
        self.mix_ln = nn.LayerNorm(hidden + summary_dim * keep_last)
        self.mix_proj = nn.Linear(hidden + summary_dim * keep_last, hidden)

        self.out_ln = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.in_proj(x)
        summaries = []

        for blk in self.blocks:
            h = blk(h)
            s = self.summary_proj(h)
            summaries.append(s)
            if len(summaries) > self.keep_last:
                summaries.pop(0)

            if len(summaries) == self.keep_last:
                cat = torch.cat([h] + summaries, dim=-1)
                h = self.mix_proj(self.mix_ln(cat)) + h  # residual mix

        h = self.out_ln(h)
        return self.out_proj(h)        
    
EMBEDDER_REGISTRY = {
    # ------------------
    # BatchNorm embedders
    # ------------------
    "bn": EmbedderBN,
    "bn_deep": EmbedderBNDeep,
    "bn_wide": EmbedderBNWide,
    "bn_res": EmbedderBNRes,
    "bn_cascade": EmbedderBNCascade,

    # ------------------
    # LayerNorm embedders
    # ------------------
    "ln": EmbedderLN,
    "ln_deep": EmbedderLNDeep,
    "ln_wide": EmbedderLNWide,
    "ln_res": EmbedderLNRes,
    "ln_cascade": EmbedderLNCascade,

    # ------------------
    # Node and xp embedders
    # ------------------
    "node": EmbedderNODE,

    # ------------------
    # dcnv2
    # ------------------
    "dcnv2_base": EmbedderDCNv2Base,

    #-------------------
    # residual
    #-------------------

    "ln_res_l":     EmbedderLNResL,
    "ln_res_skip":  EmbedderLNResSkip,
    "ln_res_dense": EmbedderLNResDense,
}

def build_embedder(name: str, **kwargs):
    if name not in EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder '{name}'. "
            f"Available embedders: {list(EMBEDDER_REGISTRY.keys())}"
        )
    return EMBEDDER_REGISTRY[name](**kwargs)

def sample_random_embedder(
    embedder_type: str,
    embedder_size: str,
    input_dim: int,
    hidden: int,
    embed_dim: int,
    device: str,
    *,
    overrides: dict | None = None,
):
    seed = int(time.time() * 1000) % 100000
    torch.manual_seed(seed)

    name = embedder_type.lower()
    size = embedder_size.lower()
    overrides = overrides or {}

    if name not in SIZE_LADDERS:
        raise ValueError(f"Unknown embedder_type={name}")
    if size not in SIZE_LADDERS[name]:
        raise ValueError(f"Unknown size={size} for {name}")

    ladder = dict(SIZE_LADDERS[name][size])
    ladder.update(overrides)

    # ---- normalize kwargs ----
    if name == "ln_res_l":
        h = _round(hidden * ladder["hidden_mul"])
        kwargs = dict(
            input_dim=input_dim,
            hidden=h,
            embed_dim=embed_dim,
            depth=ladder["depth"],
            expansion=ladder["expansion"],
            dropout=0.0,
        )

    elif name == "dcnv2_base":
        h = _round(hidden * ladder["hidden_mul"])
        kwargs = dict(
            input_dim=input_dim,
            hidden=h,
            embed_dim=embed_dim,
        )

    elif name == "node":
        tree_dim = _round((embed_dim // 2) * ladder["tree_dim_mul"], multiple=8, min_val=16)
        kwargs = dict(
            input_dim=input_dim,
            hidden=hidden,  # kept for signature compatibility
            embed_dim=embed_dim,
            num_layers=ladder["num_layers"],
            num_trees=ladder["num_trees"],
            depth=ladder["depth"],
            tree_dim=tree_dim,
            dropout=0.0,
        )

    else:
        kwargs = dict(
            input_dim=input_dim,
            hidden=hidden,
            embed_dim=embed_dim,
        )

    # ---- build via EXISTING registry ----
    net = build_embedder(name, **kwargs)

    # ---- freeze + eval ----
    for p in net.parameters():
        p.requires_grad_(False)

    net = net.to(device)
    net.eval()
    return net

def sample_random_embedder_from_pool(
    pool: list[str],
    input_dim: int,
    hidden: int,
    embed_dim_out: int,
    device: str,
):
    """
    Samples one embedder type from pool, builds it, freezes, returns it.
    Assumes your existing sample_random_embedder can build each type with embed_dim=embed_dim_out.
    """
    name = pool[np.random.randint(0, len(pool))]
    embedder_size='base'
    net = sample_random_embedder(name, embedder_size, input_dim, hidden, embed_dim_out, device)
    net.eval()
    return net, name

def build_fusion_embedder(
    pool: list[str],
    input_dim: int,
    hidden: int,
    per_dim: int,
    out_dim: int,
    device: str,
):
    """
    Builds all embedders in pool with output dim = per_dim, concatenates, compresses to out_dim.
    """
    embedders = []
    out_dims = []
    embedder_size='base'
    for name in pool:
        e = sample_random_embedder(name, embedder_size, input_dim, hidden, per_dim, device)
        e.eval()
        embedders.append(e)
        out_dims.append(per_dim)

    fusion = FusionEmbedder(embedders, out_dims, out_dim).to(device)
    fusion.eval()

    # Freeze everything (including compress layer) because embedders are supposed to be fixed feature maps.
    fusion.freeze()
    return fusion


