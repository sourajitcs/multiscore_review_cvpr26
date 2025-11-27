# 🧩 Multi-Score Retrieval Framework

This repository implements the **two-stage zero-shot multimodal retrieval pipeline**  
The system combines efficient **Stage-1 candidate filtering** with fine-grained **Stage-2 multi-score re-ranking**, integrating both **Bidirectional Chain-of-Thought (CoT) Embedding** and **Question-Answer (QA) Relevance** signals.

---

## 📂 Directory Overview

```
multiscore_review_cvpr26/
│
├── stage_1/
│ └── pyramid_rank.py
│
├── stage_2/
│ ├── bidirectional_cot_embedding_score.py
│ └── qa_relevance_score.py
│
└── README.md ← (this file)
```

---

## ⚙️ Stage-1: Efficient Filtering via PyramidRank

**PyramidRank** performs hierarchical multi-resolution filtering based on **Matryoshka Representation Learning (MRL)**.  
It progressively increases embedding dimensionality (32→1024 D), computing an **upper-bound similarity** at each level and pruning away candidates whose bound falls below a binary-searched threshold τ until only `K` remain.

```bash
python stage_1/pyramid_rank.py \
  --batch_size $Q \
  --queries  $QUERIES \
  --candidates $CANDS \
  --K 50 \
  --epsilon 0.02 \
  --levels 32,64,128,256,512,1024 \
  --encoder Qwen/Qwen3-0.6B-Embedding \
  --embed_batch_size 128 \
  --max_length 256 \
  --dtype bf16 \
  --save_json results.jsonl \
  --save_stats stats.jsonl
```

## 🧠 Stage-2: Bidirectional CoT Embedding Score

**This step extracts contextual embeddings from both directions — candidate → query and query → candidate — using a chain-of-thought (CoT) prompting scheme that introduces a special <emb> token.
The embeddings preceding this token are used as alignment features, and their cosine similarity forms the Dual CoT Embedding Score.**


```bash
python stage_2/bidirectional_cot_embedding_score.py \
  --batch_size 2 \
  --queries "a person is swimming in some white water rapids." \
            "young men discuss and demonstrate a video game." \
  --videos /path/to/video1.mp4 /path/to/video2.mp4 \
  --example_video /path/to/any_small_video.mp4

```

## 🤖 Stage-2: QA Relevance Score

**This component evaluates semantic correctness by generating a bank of discriminative Yes/No QA pairs from each query and assessing each candidate’s ability to answer them correctly.
The mean accuracy across generated pairs forms the QA Relevance Score (S₍QA₎ ∈ [0, 1]).**


```bash
python stage_2/qa_relevance_score.py \
  --batch_size 2 \
  --queries "a person is swimming in some white water rapids." \
            "two people playing tennis indoors." \
  --videos /path/to/vid1.mp4 /path/to/vid2.mp4 \
  --num_qas 5 \
  --num_frames 12 \
  --save_scores qa_scores.npy

```


### 🧮 Conceptual Overview

| Stage | Purpose | Core Mechanism | Output |
|-------|----------|----------------|---------|
| **Stage 1 – PyramidRank** | Efficient coarse filtering | Hierarchical MRL embedding + binary search pruning | Top-K candidate set |
| **Stage 2a – Bidirectional CoT Score** | Cross-directional alignment | `<emb>` hidden state similarity | CoT similarity vector |
| **Stage 2b – QA Relevance Score** | Semantic verification | MLLM-based Yes/No QA accuracy | QA relevance vector |





