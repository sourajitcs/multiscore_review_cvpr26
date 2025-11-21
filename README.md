# multiscore_review_cvpr26


Stage 1: Computing Ranking with PyramidRank

python pyramid_rank.py \
  --batch_size 128 \
  --queries $(python - <<'PY'
for i in range(128): print(f'"query text {i}"', end=' ')
PY
) \
  --candidates $(python - <<'PY'
for i in range(128): print(f'"candidate text {i}"', end=' ')
PY
) \
  --levels 32,64,128,256,512,1024 \
  --save_bounds per_level_bounds.npy


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



