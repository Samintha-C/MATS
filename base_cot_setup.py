import os
import json
import random
from pathlib import Path
from collections import Counter
import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

# Make HF listing/downloading less brittle
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster when available

REPO_ID = "uzaymacar/math-rollouts"
PREFIX = "deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution"

# Anchor tags we care about (lowercase to match dataset)
ANCHOR_TAGS = {"plan_generation", "self_checking", "uncertainty_management"}

def repo_rel(hf_name: str) -> str:
    """Convert 'datasets/<repo>/<path>' -> '<path>' for hf_hub_download()"""
    return hf_name.split(f"datasets/{REPO_ID}/", 1)[1]

def load_one_problem(hf_dir_entry) -> dict:
    """
    Returns a dict with: problem_dir, problem, gt_answer, level, type,
    base_cot, chunks_df (tidy per-chunk table), tag_counts.
    """
    pdir_rel = repo_rel(hf_dir_entry["name"])         # e.g., deepseek.../correct_base_solution/problem_1591
    pjson_rel = f"{pdir_rel}/problem.json"
    clabel_rel = f"{pdir_rel}/chunks_labeled.json"

    # Download only these two tiny files
    pjson_fp   = hf_hub_download(REPO_ID, filename=pjson_rel,  repo_type="dataset")
    clabel_fp  = hf_hub_download(REPO_ID, filename=clabel_rel, repo_type="dataset")

    prob = json.loads(Path(pjson_fp).read_text())
    chunks = json.loads(Path(clabel_fp).read_text())

    # Sort chunks by `chunk_idx` and build tidy table
    chunks_sorted = sorted(chunks, key=lambda c: c.get("chunk_idx", 1e9))
    rows = []
    tag_counter = Counter()
    for c in chunks_sorted:
        tags = [t.lower() for t in c.get("function_tags", [])]
        for t in tags:
            tag_counter[t] += 1
        rows.append({
            "problem_dir": Path(pdir_rel).name,
            "chunk_idx": c.get("chunk_idx"),
            "chunk": c.get("chunk", ""),
            "function_tags": tags,
        })
    chunks_df = pd.DataFrame(rows)

    # Reconstruct base CoT
    base_cot = "\n".join(chunks_df.sort_values("chunk_idx")["chunk"].tolist())

    return {
        "problem_dir": Path(pdir_rel).name,
        "problem": prob.get("problem", ""),
        "gt_answer": prob.get("gt_answer", ""),
        "level": prob.get("level", ""),
        "type": prob.get("type", ""),
        "base_cot": base_cot,
        "tag_counts": dict(tag_counter),
        "chunks_df": chunks_df,
    }

def download_sample_problems(num_problems=10, seed=13331237):
    """Download a sample of problems from the dataset"""
    random.seed(seed)
    
    # List just the problem_* dirs (no full snapshot)
    fs = HfFileSystem()
    items = fs.ls(f"datasets/{REPO_ID}/{PREFIX}", detail=True)
    problem_dirs = [it for it in items if it["type"] == "directory" and it["name"].rsplit("/", 1)[-1].startswith("problem_")]

    # sample problems
    sampled = random.sample(problem_dirs, k=min(num_problems, len(problem_dirs)))
    
    bundle = [load_one_problem(it) for it in sampled]
    
    # Create compact table about the sampled problems
    df_sample = pd.DataFrame([
        {k: v for k,v in b.items() if k not in ("chunks_df", "base_cot")}
        for b in bundle
    ])
    
    return bundle, df_sample

def extract_anchors(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Extract anchor chunks from the chunks dataframe"""
    return (
        chunks_df
        .assign(_has_anchor=lambda d: d["function_tags"].apply(lambda ts: bool(ANCHOR_TAGS & set(ts))))
        .loc[lambda d: d["_has_anchor"]]
        .drop(columns=["_has_anchor"])
        .copy()
    )

def render_cot_from_df(chunks_df: pd.DataFrame) -> str:
    """Reconstruct CoT from chunks dataframe"""
    return "\n".join(
        chunks_df.sort_values("chunk_idx")["chunk"].tolist()
    )

def replace_chunk(chunks_df: pd.DataFrame, chunk_idx: int, new_text: str) -> pd.DataFrame:
    """Replace a chunk in the dataframe with new text"""
    out = chunks_df.copy()
    out.loc[out["chunk_idx"] == chunk_idx, "chunk"] = new_text
    return out

def pick_anchor_rows(chunks_df: pd.DataFrame, max_pick=3) -> pd.DataFrame:
    """Pick anchor rows from chunks dataframe"""
    rows = []
    for _, r in chunks_df.iterrows():
        tags = set(t.lower() for t in (r.get("function_tags") or []))
        if ANCHOR_TAGS & tags:
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=chunks_df.columns)
    out = pd.DataFrame(rows).sort_values("chunk_idx").head(max_pick)
    return out

def cot_prefix_up_to(chunks_df: pd.DataFrame, target_idx: int) -> str:
    """Return CoT prefix through (and including) the chunk_idx."""
    return "\n".join(
        chunks_df[chunks_df["chunk_idx"] <= target_idx]
        .sort_values("chunk_idx")["chunk"].tolist()
    )
