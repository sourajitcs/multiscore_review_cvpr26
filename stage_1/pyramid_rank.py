from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


_DTYPE_MAP = {
    "auto": None,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_levels(s: str) -> List[int]:
    lv = sorted({int(x) for x in s.split(",") if x.strip()})
    if not lv:
        raise ValueError("No valid --levels provided.")
    return lv

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)

def to_dim(x: np.ndarray, target: int) -> np.ndarray:
    """Truncate or zero-pad to target dim."""
    d = x.shape[1]
    if d == target:
        return x
    if d > target:
        return x[:, :target].copy()
    padw = target - d
    return np.concatenate([x, np.zeros((x.shape[0], padw), dtype=x.dtype)], axis=1)

def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)

def upper_bound_level(zq: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """
    Vectorized upper bound per pair at one level:
      U = <zq, zc> + sqrt((1 - ||zq||^2)(1 - ||zc||^2))
    zq: [1, d], zc: [M, d]  -> broadcast to [M]
    """
    dots = zc @ zq.T  # [M, 1]
    nq2 = float(np.sum(zq * zq))               # scalar
    nc2 = np.sum(zc * zc, axis=1)             # [M]
    tails = np.sqrt(np.maximum(0.0, 1.0 - nq2) * np.maximum(0.0, 1.0 - nc2))
    return (dots[:, 0] + tails).astype(np.float32)  # [M]


class HFEmbedder:
    """
    Batched text encoder using HF (default: Qwen/Qwen3-0.6B-Embedding).
    Mean-pools last_hidden_state with attention mask.
    """
    def __init__(self, model_id: str, device: str = None, torch_dtype: str = "auto"):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _DTYPE_MAP.get(torch_dtype.lower(), None)

        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device).eval()

        with torch.no_grad():
            t = self.tok(["_"], return_tensors="pt", padding=True, truncation=True).to(self.device)
            o = self.model(**t)
            self.dim = int(o.last_hidden_state.shape[-1])

    def encode(self, texts: List[str], batch_size: int = 128, max_length: int = 256) -> np.ndarray:
        vecs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (HF)", unit="batch"):
            chunk = texts[i:i+batch_size]
            with torch.no_grad():
                tok = self.tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                out = self.model(**tok)
                last = out.last_hidden_state                     # [B, T, D]
                mask = tok["attention_mask"].unsqueeze(-1)       # [B, T, 1]
                denom = torch.clamp(mask.sum(dim=1), min=1)
                emb = (last * mask).sum(dim=1) / denom           # mean pool -> [B, D]
                vecs.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(vecs, axis=0)


def pyramidrank_for_query(
    zq_full: np.ndarray,             # [Dmax]
    Zc_full: np.ndarray,             # [N, Dmax]
    levels: List[int],
    K: int,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Performs PyramidRank for ONE query over N shared candidates.

    Returns:
      topk_indices: [K] indices into candidates
      topk_scores:  [K] cosine scores at final level (exact dot at dim=levels[-1])
      stats: dict with per-level pruning stats
    """
    Dmax = Zc_full.shape[1]
    assert zq_full.shape[0] == Dmax

    # Precompute per-level query prefixes
    zq_levels = {d: zq_full[:d][None, :] for d in levels}  # [1, d]

    N = Zc_full.shape[0]
    survivors = np.arange(N, dtype=np.int64)

    stats = {
        "N0": int(N),
        "levels": levels,
        "survivors_per_level": [],
        "thresholds": [],
        "iters_per_level": [],
    }

    # Binary-search threshold τ at each level
    for d in levels:
        if survivors.size <= K:
            stats["survivors_per_level"].append(int(survivors.size))
            stats["thresholds"].append(None)
            stats["iters_per_level"].append(0)
            continue

        zq = zq_levels[d]                            # [1, d]
        zc = Zc_full[survivors, :d]                  # [|S|, d]

        # \tau bounds (cosine domain), can also use observed min/max
        tau_min, tau_max = -1.0, 1.0
        it = 0

        # Precompute all upper bounds once at this level (vectorized)
        U = upper_bound_level(zq, zc)                # [|S|]

        # Binary search on \tau to hit ≈ K survivors
        while (tau_max - tau_min) > epsilon:
            it += 1
            tau = 0.5 * (tau_min + tau_max)
            keep = (U >= tau)
            n_keep = int(keep.sum())

            # tighten if too many survivors; loosen if too few
            if n_keep >= K:
                tau_min = tau
            else:
                tau_max = tau

        # Final keep at \tau = tau_min (tightest feasible)
        tau_final = tau_min
        keep = (U >= tau_final)
        survivors = survivors[keep]

        stats["survivors_per_level"].append(int(survivors.size))
        stats["thresholds"].append(float(tau_final))
        stats["iters_per_level"].append(int(it))

        if survivors.size == 0:
            # No candidates survive; early exit
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), stats

    # Final selection at max level: exact cosine, then take top-K
    dL = levels[-1]
    zqL = zq_levels[dL]                  # [1, dL]
    ZcL = Zc_full[survivors, :dL]        # [S, dL]
    scores = cosine(ZcL, zqL)            # [S]
    if survivors.size > K:
        top_idx = np.argpartition(-scores, K - 1)[:K]
        # sort the selected K
        order = np.argsort(-scores[top_idx])
        selected = survivors[top_idx][order]
        selected_scores = scores[top_idx][order]
    else:
        order = np.argsort(-scores)
        selected = survivors[order]
        selected_scores = scores[order]

    return selected, selected_scores.astype(np.float32), stats

def run_pyramidrank(
    queries: List[str],
    candidates: List[str],
    levels: List[int],
    K: int,
    epsilon: float,
    encoder_id: str,
    embed_bs: int,
    max_len: int,
    dtype: str,
) -> Tuple[List[List[int]], List[List[float]], List[Dict[str, Any]]]:
    """
    Stage-1 for all queries; shared candidate bank.
    Returns:
      all_topk_ids:  list of length Q, each a list[int] of length ≤ K
      all_topk_scores: list of length Q, each list[float]
      all_stats: list of per-query stats dicts
    """
    Q = len(queries)
    N = len(candidates)
    assert Q > 0 and N > 0
    levels = sorted(levels)
    Dmax = levels[-1]

    # Encode once
    embedder = HFEmbedder(encoder_id, torch_dtype=dtype)
    with tqdm(total=2, desc="Encoding texts", unit="set"):
        E_q = embedder.encode(queries, batch_size=embed_bs, max_length=max_len)
        E_c = embedder.encode(candidates, batch_size=embed_bs, max_length=max_len)

    # Project to Dmax and L2-normalize at full dim
    E_q = l2_normalize(to_dim(E_q, Dmax))
    E_c = l2_normalize(to_dim(E_c, Dmax))

    all_topk_ids: List[List[int]] = []
    all_topk_scores: List[List[float]] = []
    all_stats: List[Dict[str, Any]] = []

    for qi in tqdm(range(Q), desc="PyramidRank per query", unit="q"):
        zq_full = E_q[qi]              # [Dmax]
        Zc_full = E_c                  # [N, Dmax]
        sel_ids, sel_scores, stats = pyramidrank_for_query(
            zq_full, Zc_full, levels, K, epsilon
        )
        all_topk_ids.append(sel_ids.tolist())
        all_topk_scores.append(sel_scores.tolist())
        all_stats.append({
            "query_index": qi,
            **stats
        })

    return all_topk_ids, all_topk_scores, all_stats

# ---------------------- CLI ---------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="PyramidRank Stage-1 (Qwen3 0.6B embeddings) for batched queries.")
    p.add_argument("--batch_size", type=int, required=True, help="Number of queries Q.")
    p.add_argument("--queries", nargs="+", required=True, help="Q query strings.")
    p.add_argument("--candidates", nargs="+", required=True, help="Shared candidate strings (size N).")
    p.add_argument("--K", type=int, required=True, help="Top-K to return per query.")
    p.add_argument("--epsilon", type=float, default=1e-3, help="Binary-search tolerance on τ.")
    p.add_argument("--levels", type=str, default="32,64,128,256,512,1024",
                   help="Comma-separated dims for MRL levels (ascending).")
    p.add_argument("--encoder", type=str, default="Qwen/Qwen3-0.6B-Embedding",
                   help="HF embedding model id.")
    p.add_argument("--embed_batch_size", type=int, default=128, help="Batch size for encoder.")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer truncation length.")
    p.add_argument("--dtype", type=str, default="auto", choices=list(_DTYPE_MAP.keys()),
                   help="Torch dtype for encoder.")
    p.add_argument("--save_json", type=str, default=None, help="Optional JSONL path for results.")
    p.add_argument("--save_stats", type=str, default=None, help="Optional JSONL path for pruning stats.")
    return p.parse_args()

def main():
    args = parse_args()

    if len(args.queries) != args.batch_size:
        print(f"[Error] --batch_size={args.batch_size} but got {len(args.queries)} queries.", file=sys.stderr)
        sys.exit(1)
    if args.K <= 0:
        print("[Error] --K must be > 0.", file=sys.stderr)
        sys.exit(1)

    levels = parse_levels(args.levels)

    topk_ids, topk_scores, stats = run_pyramidrank(
        queries=args.queries,
        candidates=args.candidates,
        levels=levels,
        K=args.K,
        epsilon=args.epsilon,
        encoder_id=args.encoder,
        embed_bs=args.embed_batch_size,
        max_len=args.max_length,
        dtype=args.dtype,
    )

    # print(f"[Info] levels = {levels} | K = {args.K} | epsilon = {args.epsilon}")
    # for i, (ids, sc) in enumerate(zip(topk_ids, topk_scores)):
    #     print(f"[Q{i}] topK_ids={ids}")
    #     print(f"[Q{i}] topK_scores={np.array(sc, dtype=np.float32)}")

    # Optional JSONL saves
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            for i, (ids, sc) in enumerate(zip(topk_ids, topk_scores)):
                f.write(json.dumps({
                    "query_index": i,
                    "topK_ids": ids,
                    "topK_scores": sc,
                }) + "\n")
        print(f"[Info] Saved results JSONL -> {args.save_json}")

    if args.save_stats:
        os.makedirs(os.path.dirname(args.save_stats) or ".", exist_ok=True)
        with open(args.save_stats, "w", encoding="utf-8") as f:
            for s in stats:
                f.write(json.dumps(s) + "\n")
        print(f"[Info] Saved per-level stats JSONL -> {args.save_stats}")

if __name__ == "__main__":
    main()
