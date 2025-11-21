# 🧩 Multi-Score Retrieval Framework (CVPR 2026 Review Submission)

This repository implements the **two-stage zero-shot multimodal retrieval pipeline** used in our CVPR 2026 submission.  
The system combines efficient **Stage-1 candidate filtering** with fine-grained **Stage-2 multi-score re-ranking**, integrating both **Bidirectional Chain-of-Thought (CoT) Embedding** and **Question-Answer (QA) Relevance** signals.

---

## 📂 Directory Overview















# multiscore_review_cvpr26


Stage 1: Computing Ranking with PyramidRank

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



Stage 2: Computing Re-ranking with  Biderectional CoT Embedding Score

python stage_2/bidirectional_cot_embedding_score.py \
  --batch_size 2 \
  --queries "a person is swimming in some white water rapids." "young men discuss and demonstrate a video game." \
  --videos /path/to/video1.mp4 /path/to/video2.mp4 \
  --example_video /path/to/any_small_video.mp4


Stage 2: Computing Re-ranking with  QA Relevance Score

python stage_2/qa_relevance_score.py \
  --batch_size 2 \
  --queries "a person is swimming in some white water rapids." "two people playing tennis indoors." \
  --videos /path/to/vid1.mp4 /path/to/vid2.mp4 \
  --num_qas 5 \
  --num_frames 12 \
  --save_scores qa_scores.npy



