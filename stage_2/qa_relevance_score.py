import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)


def _uniform_indices(n_total: int, num: int) -> List[int]:
    if n_total <= 0:
        return []
    if num <= 1:
        return [0]
    return np.linspace(0, n_total - 1, num=num).astype(int).tolist()

def load_video_frames_decord(path: str, num_frames: int) -> List[Image.Image]:
    import decord  
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path)
    idxs = _uniform_indices(len(vr), num_frames)
    return [Image.fromarray(vr[i].asnumpy()) for i in idxs]

def load_video_frames_opencv(path: str, num_frames: int) -> List[Image.Image]:
    import cv2  
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[Image.Image] = []

    if n_total <= 0:
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not raw:
            raise RuntimeError(f"No frames decoded from video: {path}")
        idxs = _uniform_indices(len(raw), num_frames)
        frames = [raw[i] for i in idxs]
    else:
        idxs = _uniform_indices(n_total, num_frames)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                ok2, frame2 = cap.read()
                if not ok2:
                    continue
                frame = frame2
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

    if not frames:
        raise RuntimeError(f"Failed to sample frames for: {path}")
    return frames

def load_video_frames(path: str, num_frames: int = 12) -> List[Image.Image]:
    last_err = None
    try:
        return load_video_frames_decord(path, num_frames)
    except Exception as e:
        last_err = e
    try:
        return load_video_frames_opencv(path, num_frames)
    except Exception as e2:
        raise RuntimeError(
            f"Could not load video frames with decord ({last_err}) or OpenCV ({e2}) for {path}"
        )


def load_model_and_processor(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    return tok, processor, model


def build_qa_generation_message(query_text: str, M: int) -> List[dict]:
    """
    Uses only text (no video). Asks the model to output exactly M pairs:
    Q1: ... ? A: Yes/No
    ...
    """
    instructions = (
        "You are given a multimodal query: <q>\n"
        "Generate " + str(M) + " Yes/No questions that capture essential semantics from the above query.\n"
        "For each question, also provide its correct Yes/No answer.\n"
        "Format strictly as:\n"
        "Q1: <question>?  A: Yes/No\n"
        "Q2: <question>?  A: Yes/No\n"
        "...\n"
        "Output only the list of Q/A pairs."
    )
    content = [
        {"type": "text", "text": instructions.replace("<q>", query_text)}
    ]
    return [{"role": "user", "content": content}]

def build_qa_answer_message(question_text: str, video_frames: List[Image.Image]) -> List[dict]:
    """
    Asks the model a single Yes/No question conditioned on the candidate video frames.
    Forces concise Yes/No output.
    """
    prompt = (
        "Answer the following question using ONLY the video content. "
        "Reply with a single word: 'Yes' or 'No'.\n"
        f"Question: {question_text}"
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "video", "video": video_frames}, 
    ]
    return [{"role": "user", "content": content}]

_QA_LINE = re.compile(
    r"^\s*Q\s*(\d+)\s*:\s*(.+?)\s*A\s*:\s*(Yes|No)\s*$",
    flags=re.IGNORECASE,
)

def parse_qa_pairs(text: str, M: int) -> List[Tuple[str, str]]:
    """
    Parse lines like:
      Q1: Is there a person? A: Yes
    Returns list of (question, answer) of length up to M.
    """
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        m = _QA_LINE.match(line.strip())
        if not m:
            continue
        qtxt = m.group(2).strip()
        ans = m.group(3).strip().capitalize()
        if ans not in ("Yes", "No"):
            continue
        pairs.append((qtxt, ans))
        if len(pairs) >= M:
            break
    return pairs

def normalize_yesno(text: str) -> Optional[str]:
    """
    Extract a 'Yes' or 'No' from model output.
    """
    low = text.strip().lower()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    if "yes" in low and "no" not in low:
        return "Yes"
    if "no" in low and "yes" not in low:
        return "No"
    return None


def generate_text_from_messages(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    messages: List[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """
    Builds chat template, tokenizes, and generates text-only output string.
    """
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=None, videos=None, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0 else None,
            use_cache=True,
        )
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return out

def generate_yesno_from_video(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    question_text: str,
    video_frames: List[Image.Image],
    max_new_tokens: int = 4,
) -> str:
    """
    Generates a single-word Yes/No answer conditioned on video frames.
    """
    messages = build_qa_answer_message(question_text, video_frames)
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=None, videos=[[video_frames]], return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    ans = normalize_yesno(out) or "No" 
    return ans


def parse_args():
    p = argparse.ArgumentParser(description="Batch QA Relevance Score (Stage-2) with Qwen2.5-VL-3B-Instruct.")
    p.add_argument("--batch_size", type=int, required=True, help="Batch size m.")
    p.add_argument("--queries", nargs="+", required=True, help="List of m query texts.")
    p.add_argument("--videos", nargs="+", required=True, help="List of m candidate video paths.")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=12, help="Frames per video to sample.")
    p.add_argument("--num_qas", type=int, default=5, help="Number of Yes/No QA pairs to generate per query.")
    p.add_argument("--save_scores", type=str, default=None, help="Optional path to save npy of QA scores.")
    return p.parse_args()

def main():
    args = parse_args()

    if len(args.queries) != args.batch_size or len(args.videos) != args.batch_size:
        print(
            f"[Error] --batch_size={args.batch_size} but got {len(args.queries)} queries "
            f"and {len(args.videos)} videos.",
            file=sys.stderr,
        )
        sys.exit(1)

    tok, processor, model = load_model_and_processor(args.model_id)
    device = model.device
    model.eval()

    all_frames: List[List[Image.Image]] = []
    for vpath in tqdm(args.videos, desc="Loading videos", unit="vid"):
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video not found: {vpath}")
        frames = load_video_frames(vpath, num_frames=args.num_frames)
        all_frames.append(frames)

    scores = []

    for i in tqdm(range(args.batch_size), desc="Scoring batch", unit="item"):
        qtxt = args.queries[i]
        vframes = all_frames[i]

        # Generate M QA pairs from the query
        gen_msgs = build_qa_generation_message(qtxt, args.num_qas)
        gen_txt = generate_text_from_messages(processor, model, gen_msgs, max_new_tokens=256, temperature=0.0)
        qa_pairs = parse_qa_pairs(gen_txt, args.num_qas)

        if len(qa_pairs) == 0:
            qa_pairs = [("Is the described event present?", "Yes")]

        # Ask each question on candidate videos and compute qa relevance score
        correct = 0
        total = 0
        for (qq, ans_gt) in tqdm(qa_pairs, leave=False, desc=f"QA@{i}", unit="q"):
            pred = generate_yesno_from_video(processor, model, qq, vframes, max_new_tokens=4)
            total += 1
            if pred == ans_gt:
                correct += 1

        sqa = (correct / total) if total > 0 else 0.0
        scores.append(sqa)

    scores_np = np.asarray(scores, dtype=np.float32)
    print(f"[Info] QA scores per item (len={len(scores_np)}):\n{scores_np}")

    if args.save_scores:
        os.makedirs(os.path.dirname(args.save_scores) or ".", exist_ok=True)
        np.save(args.save_scores, scores_np)
        print(f"[Info] Saved scores -> {args.save_scores}")

if __name__ == "__main__":
    main()
