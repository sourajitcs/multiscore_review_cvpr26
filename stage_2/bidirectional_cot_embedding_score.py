import argparse
import os
import sys
from typing import List, Tuple, Optional

import torch
import numpy as np
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
    idx = np.linspace(0, n_total - 1, num=num).astype(int).tolist()
    return idx

def load_video_frames_decord(path: str, num_frames: int) -> List[Image.Image]:
    import decord  
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(path)
    idxs = _uniform_indices(len(vr), num_frames)
    frames = [Image.fromarray(vr[i].asnumpy()) for i in idxs]
    return frames

def load_video_frames_opencv(path: str, num_frames: int) -> List[Image.Image]:
    import cv2  
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        return [raw[i] for i in idxs]

    idxs = _uniform_indices(n_total, num_frames)
    frames: List[Image.Image] = []
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

def build_messages_c2q(query_text: str, test_video_path: str, example_video_path: Optional[str]) -> dict:
    ex_vid = example_video_path or test_video_path
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a helpful multimodal retrieval agent. "
                    "Given a database candidate and a query, you can summarize "
                    "the alignment between the candidate and the query based on "
                    "the provided examples and explanations."
                ),
            },
            {"type": "text", "text": "Example #1: Here is an example of a candidate and a query: "},
            {"type": "text", "text": "Candidate:"},
            {"type": "video", "video": ex_vid},
            {"type": "text", "text": "Query: bill murray is being interviewed by david letterman while talking about bill's past roles."},
            {
                "type": "text",
                "text": (
                    "Explanation: "
                    "The candidate indicates the following- "
                    "bill murray is seating on the david letterman show as a guest, "
                    "bill murray is wearing a black shirt and a red shorts, "
                    "david letterman is interviewing bill murray, "
                    "david letterman is wearing glasses, white shirt, a grey suit and a colorful tie, "
                    "there is a mug and a microphone on the table, "
                    "bill murray and david letterman both are seemingly enjoying the conversation. "
                    "The query indicates the following- "
                    "david letterman is interviewing bill murray, "
                    "the interview is about bill murray's past roles. "
                    "All or most contents of the candidate is found on the query. "
                    "Therefore, summary of alignment between this candidate and the query in one word is: high."
                ),
            },
            {"type": "text", "text": "Example #2: Here is an example of a candidate and a query: "},
            {"type": "text", "text": "Candidate:"},
            {"type": "video", "video": ex_vid},
            {"type": "text", "text": "Query: young men discuss and demonstrate a video game."},
            {
                "type": "text",
                "text": (
                    "Explanation: "
                    "The candidate indicates the following- "
                    "a pizza is being made, pizza toppings are being spread, "
                    "pizza toppings include: cheese, sliced onlins, sliced green bell pepper, sliced tomatoes, "
                    "the pizza is on top of a black colored baking tray. "
                    "The query indicates the following- "
                    "there is a young men, there is a men, there is a video game, "
                    "the men is playing a video game, the men is talking about a video game. "
                    "Allmost no contents of the candidate is found on the query. "
                    "Therefore, summary of alignment between this candidate and the query in one word is: low."
                ),
            },
            {"type": "text", "text": "Now, here is a test candidate and a test query: "},
            {"type": "text", "text": "Test Candidate:"},
            {"type": "video", "video": test_video_path},
            {"type": "text", "text": f"Test Query: {query_text}"},
            {
                "type": "text",
                "text": "Using knowledge from the above example summary of alignment between test canddate and test query in one word is: <emb>",
            },
        ],
    }

def build_messages_q2c(query_text: str, test_video_path: str, example_video_path: Optional[str]) -> dict:
    ex_vid = example_video_path or test_video_path
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a helpful multimodal retrieval agent. "
                    "Given a query and a database candidate, you can summarize "
                    "the alignment between the query and the candidate based on "
                    "the provided examples and explanations."
                ),
            },
            {"type": "text", "text": "Example #1: Here is an example of a query and a candidate: "},
            {"type": "text", "text": "Query: a man is singing and standing in the road."},
            {"type": "text", "text": "Candidate: "},
            {"type": "video", "video": ex_vid},
            {
                "type": "text",
                "text": (
                    "Explanation: "
                    "The query indicates the following- "
                    "there is a man, there is a singer, there is a road, "
                    "the man is singing, the man is standing, the man is on the road, "
                    "The candidate indicates the following- "
                    "there are some man, there are some singers, there is a restaurant, there are some glasses, there are billboards, "
                    "there is a table, there are some women, there are some cups and plates, there is a road, there are traffic signs, "
                    "there are tall downtown buildings, there is moving traffic, some man are standing, some man are singing, "
                    "some man are drinking, some man are making gestures while singing, some man are sitting, some man are waving hands, "
                    "some people are dancing, a man is walking through the road, some man are standing on the road, "
                    "some man are standing on the road and singing, some people are waving hands and singing, a man is walking on the road and singing. "
                    "All or most contents of the query is found on the candidate. "
                    "Therefore, summary of alignment between this query and the candidate in one word is: high."
                ),
            },
            {"type": "text", "text": "Example #2: Here is an example of a query and a candidate: "},
            {"type": "text", "text": "Query: woman talking to a man in an interview."},
            {"type": "text", "text": "Candidate: "},
            {"type": "video", "video": ex_vid},
            {
                "type": "text",
                "text": (
                    "Explanation: "
                    "The query indicates the following- "
                    "there is a man, there is a woman, there is a interview happening, "
                    "a conversation between a man and a woman, an interview between a man and a woman. "
                    "The candidate indicates the following- "
                    "a woman is singing, there are some people and a baby on the road, a woman is blowing a kiss to the baby, "
                    "there are cars in the street, a red car is pulled over by the police. "
                    "Allmost no contents of the query is found on the candidate. "
                    "Therefore, summary of alignment between this query and the candidate in one word is: low."
                ),
            },
            {"type": "text", "text": "Now, here is a test query and a test candidate: "},
            {"type": "text", "text": f"Test Query: {query_text}"},
            {"type": "text", "text": "Test Candidate: "},
            {"type": "video", "video": test_video_path},
            {
                "type": "text",
                "text": "Using knowledge from the above example summary of alignment between test query and test candidate in one word is: <emb>",
            },
        ],
    }

def build_processor_and_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.add_special_tokens({"additional_special_tokens": ["<emb>"]})
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    model.resize_token_embeddings(len(tok))
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    processor.tokenizer = tok
    return tok, processor, model

def pack_batch_for_processor(
    texts: List[str],
    videos_frames_batch: List[List[List[Image.Image]]],  # shape: B x (#vids in message) x (#frames per vid)
):
    """
    texts: list of chat-templated strings (len B)
    videos_frames_batch: for each item i, a list of N_video segments; each segment is the list of frames for that video.
    """
    inputs = processor(
        text=texts,
        images=None,
        videos=videos_frames_batch,
        padding=True,
        return_tensors="pt",
    )
    return inputs

def extract_emb_hidden_states(
    outputs, inputs, tok: AutoTokenizer
) -> torch.Tensor:
    """For each batch item, take hidden state at position (last <emb> - 1)."""
    emb_id = tok.convert_tokens_to_ids("<emb>")
    input_ids = inputs["input_ids"]            # [B, T]
    last_hid = outputs.hidden_states[-1]       # [B, T, H]
    B, T, H = last_hid.shape

    feats = []
    for b in range(B):
        where = (input_ids[b] == emb_id).nonzero(as_tuple=False).flatten()
        if where.numel() == 0:
            raise RuntimeError("No <emb> token found in item b=%d" % b)
        emb_pos = where[-1].item()
        prev_pos = max(0, emb_pos - 1)
        feats.append(last_hid[b, prev_pos, :].unsqueeze(0))
    return torch.cat(feats, dim=0)  # [B, H]

def cosine_sim_per_pair(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: [B, H], B: [B, H] -> cos sim per row, shape [B]
    """
    A_n = torch.nn.functional.normalize(A, dim=1)
    B_n = torch.nn.functional.normalize(B, dim=1)
    return (A_n * B_n).sum(dim=1)


def parse_args():
    p = argparse.ArgumentParser(description="Dual CoT batch feature extractor (C2Q & Q2C) with cosine similarity.")
    p.add_argument("--batch_size", type=int, required=True, help="Batch size m.")
    p.add_argument("--queries", nargs="+", required=True, help="List of m query texts.")
    p.add_argument("--videos", nargs="+", required=True, help="List of m candidate video paths.")
    p.add_argument("--example_video", type=str, default=None, help="Optional path used for the two example video slots.")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num_frames", type=int, default=12)
    p.add_argument("--save_prefix", type=str, default="dual_cot_out", help="Prefix for .pt/.npy outputs.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if len(args.queries) != args.batch_size or len(args.videos) != args.batch_size:
        print(
            f"[Error] --batch_size={args.batch_size} but got {len(args.queries)} queries "
            f"and {len(args.videos)} videos.",
            file=sys.stderr,
        )
        sys.exit(1)

    tok, processor, model = build_processor_and_model(args.model_id)
    device = model.device

    c2q_texts: List[str] = []
    c2q_videos_batch: List[List[List[Image.Image]]] = []

    q2c_texts: List[str] = []
    q2c_videos_batch: List[List[List[Image.Image]]] = []

    example_frames: Optional[List[Image.Image]] = None
    if args.example_video:
        example_frames = load_video_frames(args.example_video, num_frames=args.num_frames)

    for qtxt, vpath in tqdm(zip(args.queries, args.videos), total=args.batch_size, desc="Preparing prompts & videos", unit="item"):
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video not found: {vpath}")

        test_frames = load_video_frames(vpath, num_frames=args.num_frames)
        ex_frames_1 = example_frames if example_frames is not None else test_frames
        ex_frames_2 = example_frames if example_frames is not None else test_frames

        # ----- C2Q -----
        c2q_msg = build_messages_c2q(qtxt, vpath, args.example_video)
        c2q_text = processor.apply_chat_template([c2q_msg], tokenize=False, add_generation_prompt=True)
        c2q_texts.append(c2q_text)
        c2q_videos_batch.append([ex_frames_1, ex_frames_2, test_f_]()

        # ----- Q2C -----
        q2c_msg = build_messages_q2c(qtxt, vpath, args.example_video)
        q2c_text = processor.apply_chat_template([q2c_msg], tokenize=False, add_generation_prompt=True)
        q2c_texts.append(q2c_text)
        q2c_videos_batch.append([ex_frames_1, ex_frames_2, test_frames])

    with tqdm(total=2, desc="Tokenizing & encoding", unit="step") as pbar:
        c2q_inputs = processor(
            text=c2q_texts,
            images=None,
            videos=c2q_videos_batch,
            padding=True,
            return_tensors="pt",
        )
        for k, v in list(c2q_inputs.items()):
            if isinstance(v, torch.Tensor):
                c2q_inputs[k] = v.to(device)
        pbar.update(1)

        q2c_inputs = processor(
            text=q2c_texts,
            images=None,
            videos=q2c_videos_batch,
            padding=True,
            return_tensors="pt",
        )
        for k, v in list(q2c_inputs.items()):
            if isinstance(v, torch.Tensor):
                q2c_inputs[k] = v.to(device)
        pbar.update(1)

    model.eval()
    with torch.no_grad():
        with tqdm(total=2, desc="Model forward", unit="pass") as pbar:
            out_c2q = model(**c2q_inputs, output_hidden_states=True, use_cache=False)
            pbar.update(1)
            out_q2c = model(**q2c_inputs, output_hidden_states=True, use_cache=False)
            pbar.update(1)

    f_c2q = extract_emb_hidden_states(out_c2q, c2q_inputs, processor.tokenizer)  # [B, H]
    f_q2c = extract_emb_hidden_states(out_q2c, q2c_inputs, processor.tokenizer)  # [B, H]

    cos_sim = cosine_sim_per_pair(f_c2q, f_q2c)  # [B]

    B, H = f_c2q.shape
    print(f"[Info] C2Q feature tensor: {tuple(f_c2q.shape)}")
    print(f"[Info] Q2C feature tensor: {tuple(f_q2c.shape)}")
    print(f"[Info] Cosine similarity (C2Q vs Q2C), length={len(cos_sim)}:\n{cos_sim.detach().cpu().numpy()}")

    os.makedirs(os.path.dirname(args.save_prefix) or ".", exist_ok=True)
    torch.save(f_c2q.detach().cpu(), f"{args.save_prefix}_f_c2q.pt")
    torch.save(f_q2c.detach().cpu(), f"{args.save_prefix}_f_q2c.pt")
    np.save(f"{args.save_prefix}_cosine.npy", cos_sim.detach().cpu().numpy())
