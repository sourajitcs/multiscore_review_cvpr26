# multiscore_review_cvpr26

python stage_2/bidirectional_cot_embedding_score.py \
  --batch_size 2 \
  --queries "a person is swimming in some white water rapids." "young men discuss and demonstrate a video game." \
  --videos /path/to/video1.mp4 /path/to/video2.mp4 \
  --example_video /path/to/any_small_video.mp4
