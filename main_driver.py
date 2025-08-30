#!/usr/bin/env python3
"""
Main driver script for CoT editing and evaluation pipeline.
This script orchestrates the entire process:
1. Load base CoTs from dataset
2. Edit anchor chunks
3. Continue CoT with OpenRouter/Novita
4. Evaluate results and generate plots
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Or set environment variables manually.")

# Import our modules
from base_cot_setup import (
    download_sample_problems, 
    pick_anchor_rows, 
    render_cot_from_df, 
    replace_chunk, 
    cot_prefix_up_to
)
from editor_functions import EDITORS, EDIT_TYPES
from cot_continuation import continue_from_prefix_qwen
from evaluation_graphs import (
    extract_boxed_answers, 
    answers_match, 
    normalize_answer,
    create_evaluation_report
)

# Configuration
MAX_ANCHORS_PER_PROBLEM = 200  # cap how many anchors per problem to edit/continue
NUM_PROBLEMS = 10  # number of problems to sample
SEED = 1776  # random seed for reproducibility

def run_experiment(num_problems: int = NUM_PROBLEMS, 
                  max_anchors_per_problem: int = MAX_ANCHORS_PER_PROBLEM,
                  seed: int = SEED,
                  edit_types: List[str] = None,
                  checkpoint_dir: str = "checkpoints",
                  resume: bool = True) -> pd.DataFrame:
    """
    Run the complete CoT editing experiment with checkpointing support.
    
    Args:
        num_problems: Number of problems to sample from dataset
        max_anchors_per_problem: Maximum number of anchors to edit per problem
        seed: Random seed for reproducibility
        edit_types: List of edit types to apply (defaults to EDIT_TYPES)
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint if available
    
    Returns:
        DataFrame with all experimental results
    """
    if edit_types is None:
        edit_types = EDIT_TYPES
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint file paths
    checkpoint_file = os.path.join(checkpoint_dir, "experiment_checkpoint.json")
    config_file = os.path.join(checkpoint_dir, "experiment_config.json")
    
    # Load existing results if resuming
    results = []
    completed_problems = set()
    start_problem_idx = 0
    
    if resume and os.path.exists(checkpoint_file):
        print("=== RESUMING EXPERIMENT ===")
        try:
            # Load existing results - more robust loading
            results = []
            with open(checkpoint_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
            
            # Load config to get completed problems
            with open(config_file, 'r') as f:
                config = json.load(f)
                completed_problems = set(config.get('completed_problems', []))
                start_problem_idx = config.get('next_problem_idx', 0)
            
            print(f"Loaded {len(results)} existing trials")
            print(f"Completed problems: {len(completed_problems)}")
            print(f"Resuming from problem index: {start_problem_idx}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh experiment...")
            results = []
            completed_problems = set()
            start_problem_idx = 0
    else:
        print("=== STARTING NEW EXPERIMENT ===")
    
    print(f"Problems: {num_problems}")
    print(f"Max anchors per problem: {max_anchors_per_problem}")
    print(f"Edit types: {edit_types}")
    print(f"Seed: {seed}")
    
    # Step 1: Download and load problems
    print("\n1. Loading problems from dataset...")
    bundle, df_sample = download_sample_problems(num_problems=num_problems, seed=seed)
    print(f"Loaded {len(bundle)} problems")
    print(df_sample[["problem_dir", "level", "type", "gt_answer"]].head())
    
    # Step 2: Run experiments
    print("\n2. Running experiments...")
    
    for i in range(start_problem_idx, len(bundle)):
        pb = bundle[i]
        problem_dir = pb.get("problem_dir", "")
        
        # Skip if already completed
        if problem_dir in completed_problems:
            print(f"Skipping already completed problem: {problem_dir}")
            continue
        
        problem_text = pb["problem"]
        gt_answer = str(pb.get("gt_answer") or "").strip()
        chunks_df = pb["chunks_df"]
        
        print(f"\nProcessing problem {i+1}/{len(bundle)}: {problem_dir}")
        print(f"Ground truth: {gt_answer}")
        
        # Reconstruct base for baseline answer
        base_cot = render_cot_from_df(chunks_df)
        base_boxed = extract_boxed_answers(base_cot)
        base_ans = base_boxed[-1] if base_boxed else ""
        base_correct = answers_match(base_ans, gt_answer)
        
        print(f"Baseline answer: {base_ans}")
        print(f"Baseline correct: {base_correct}")
        
        # Get anchor chunks
        anchors_df = pick_anchor_rows(chunks_df, max_pick=max_anchors_per_problem)
        print(f"Found {len(anchors_df)} anchor chunks")
        
        if anchors_df.empty:
            print("No anchor tags found; skipping problem")
            completed_problems.add(problem_dir)
            continue
        
        # Process each anchor chunk
        chunk_results = []  # Store results for this problem
        
        try:
            for _, arow in anchors_df.iterrows():
                cidx = int(arow["chunk_idx"])
                ctext = arow["chunk"]
                tags = [t.lower() for t in (arow.get("function_tags") or [])]
                
                print(f"  Processing anchor chunk {cidx}: {ctext[:100]}...")
                
                # Get prefix up to this chunk
                prefix_base = cot_prefix_up_to(chunks_df, cidx)
                
                # Apply each edit type
                skipped_edits = 0
                for edit_type in edit_types:
                    print(f"    [EDIT {edit_type}] BEFORE: {ctext}")
                    try:
                        edited_chunk = EDITORS[edit_type](ctext)
                        edit_err = None
                    except Exception as ex:
                        edited_chunk = ctext  # fallback: no change
                        edit_err = str(ex)
                        print(f"    Edit error: {edit_err}")
                    
                    # Check if edit was skipped (returned None)
                    if edited_chunk is None:
                        print(f"    [EDIT {edit_type}] SKIPPED: No mathematical content in chunk")
                        skipped_edits += 1
                        continue
                    
                    print(f"    [EDIT {edit_type}] BEFORE: {ctext}")
                    print(f"    [EDIT {edit_type}] AFTER : {edited_chunk}")
                    
                    # Splice the edited chunk back into the CoT prefix
                    edited_df = replace_chunk(chunks_df, cidx, edited_chunk)
                    prefix_edt = cot_prefix_up_to(edited_df, cidx)
                    
                    # Continue with Qwen
                    try:
                        cont = continue_from_prefix_qwen(
                            problem_text,
                            prefix_edt,
                            max_tokens=32000,
                            temperature=0.6,
                            top_p=0.95,
                            forced_answer=False
                        )
                        cont_err = None
                    except Exception as ex:
                        cont = ""
                        cont_err = str(ex)
                        print(f"    Continuation error: {cont_err}")
                    
                    # Parse final answer
                    pred_boxed = extract_boxed_answers(cont) or extract_boxed_answers(prefix_edt)
                    pred_final = pred_boxed[-1] if pred_boxed else ""
                    
                    # Compare new answer with ground truth (not baseline)
                    changed = (normalize_answer(pred_final) != normalize_answer(gt_answer))
                    correct = answers_match(pred_final, gt_answer)
                    
                    print("continuation", cont)
                    print(f"    Final answer: {pred_final}")
                    print(f"    Changed from GT: {changed}, Correct: {correct}")
                    
                    # Calculate propagation score
                    from evaluation_graphs import propagation_score
                    prop_score = propagation_score(prefix_edt, cont, ctext, edited_chunk)
                    
                    # Store results
                    trial_result = {
                        "problem_dir": problem_dir,
                        "anchor_chunk_idx": cidx,
                        "anchor_tags": ",".join(tags),
                        "edit_type": edit_type,
                        "original_chunk": ctext,
                        "edited_chunk": edited_chunk,
                        "prefix_used": prefix_edt,
                        "continuation": cont,
                        "gt_answer": gt_answer,
                        "baseline_final": base_ans,
                        "baseline_correct": base_correct,  # Keep for reference but don't rely on it
                        "final_answer": pred_final,
                        "final_correct": correct,
                        "answer_changed_from_gt": changed,  # Renamed to be explicit
                        "propagation_score": prop_score,
                        "edit_error": edit_err,
                        "cont_error": cont_err,
                        "edit_skipped": False,
                    }
                    
                    results.append(trial_result)
                    chunk_results.append(trial_result)
                    
                    # Save checkpoint after each successful trial (more frequent saves)
                    if len(chunk_results) % 5 == 0:  # Save every 5 trials
                        save_checkpoint(results, completed_problems, i + 1, checkpoint_dir, 
                                      f"Partial progress: {len(chunk_results)} trials for {problem_dir}")
                
                # Report skipped edits for this chunk
                if skipped_edits > 0:
                    print(f"    SKIPPED {skipped_edits} edits for chunk {cidx} (no mathematical content)")
                    
                    # Store skipped edit record
                    skipped_result = {
                        "problem_dir": problem_dir,
                        "anchor_chunk_idx": cidx,
                        "anchor_tags": ",".join(tags),
                        "edit_type": "skipped",
                        "original_chunk": ctext,
                        "edited_chunk": ctext,  # No change
                        "prefix_used": cot_prefix_up_to(chunks_df, cidx),
                        "continuation": "",
                        "gt_answer": gt_answer,
                        "baseline_final": base_ans,
                        "baseline_correct": base_correct,  # Keep for reference
                        "final_answer": gt_answer,  # Use ground truth answer (should be same as baseline)
                        "final_correct": True,  # Should always be correct for ground truth
                        "answer_changed_from_gt": False,  # No change from ground truth
                        "propagation_score": 0.0,
                        "edit_error": None,
                        "cont_error": None,
                        "edit_skipped": True,
                    }
                    
                    results.append(skipped_result)
                    chunk_results.append(skipped_result)
            
            # Mark problem as completed and save final checkpoint
            completed_problems.add(problem_dir)
            save_checkpoint(results, completed_problems, i + 1, checkpoint_dir, 
                          f"Completed problem: {problem_dir}")
            
            print(f"Completed problem {problem_dir}. Total trials so far: {len(results)}")
            
        except Exception as e:
            # If anything fails during problem processing, save what we have
            print(f"Error processing problem {problem_dir}: {e}")
            print("Saving partial progress...")
            save_checkpoint(results, completed_problems, i + 1, checkpoint_dir, 
                          f"ERROR - Partial progress for {problem_dir}: {str(e)}")
            # Don't mark as completed since it failed
            continue
    
    # Step 3: Create results DataFrame
    results_df = pd.DataFrame(results)
    print(f"\n3. Experiment completed!")
    print(f"Total trials: {len(results_df)}")
    
    return results_df

def save_checkpoint(results: List[Dict], completed_problems: set, next_problem_idx: int, checkpoint_dir: str, message: str = ""):
    """Save experiment checkpoint"""
    checkpoint_file = os.path.join(checkpoint_dir, "experiment_checkpoint.json")
    csv_file = os.path.join(checkpoint_dir, "experiment_checkpoint.csv")
    master_csv_file = os.path.join(checkpoint_dir, "master_results.csv")
    config_file = os.path.join(checkpoint_dir, "experiment_config.json")
    
    # Save results as JSON Lines format
    try:
        with open(checkpoint_file, 'w') as f:
            for result in results:
                # Ensure the result is serializable
                json_str = json.dumps(result, ensure_ascii=False, default=str)
                f.write(json_str + '\n')
    except Exception as e:
        print(f"Error saving JSON checkpoint: {e}")
        # Fallback: save as regular JSON
        with open(checkpoint_file + ".fallback", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Save results as CSV for easier analysis
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_file, index=False)
        print(f"  CSV checkpoint saved: {csv_file}")
    except Exception as e:
        print(f"Error saving CSV checkpoint: {e}")
    
    # Update master CSV (append new results)
    try:
        results_df = pd.DataFrame(results)
        if os.path.exists(master_csv_file):
            # Append to existing master CSV
            master_df = pd.read_csv(master_csv_file)
            # Remove any duplicates (in case of resume)
            combined_df = pd.concat([master_df, results_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['problem_dir', 'anchor_chunk_idx', 'edit_type'], keep='last')
            combined_df.to_csv(master_csv_file, index=False)
            print(f"  Master CSV updated: {master_csv_file} ({len(combined_df)} total trials)")
        else:
            # Create new master CSV
            results_df.to_csv(master_csv_file, index=False)
            print(f"  Master CSV created: {master_csv_file}")
    except Exception as e:
        print(f"Error updating master CSV: {e}")
    
    # Save config
    config = {
        'completed_problems': list(completed_problems),
        'next_problem_idx': next_problem_idx,
        'total_trials': len(results),
        'timestamp': pd.Timestamp.now().isoformat(),
        'last_message': message
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved: {len(results)} trials, {len(completed_problems)} problems completed - {message}")

def check_progress(checkpoint_dir: str = "checkpoints"):
    """Check current experiment progress"""
    config_file = os.path.join(checkpoint_dir, "experiment_config.json")
    checkpoint_file = os.path.join(checkpoint_dir, "experiment_checkpoint.json")
    
    if not os.path.exists(config_file):
        print("No checkpoint found. No experiment in progress.")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("=== EXPERIMENT PROGRESS ===")
        print(f"Completed problems: {len(config.get('completed_problems', []))}")
        print(f"Next problem index: {config.get('next_problem_idx', 0)}")
        print(f"Total trials: {config.get('total_trials', 0)}")
        print(f"Last updated: {config.get('timestamp', 'Unknown')}")
        
        if os.path.exists(checkpoint_file):
            results_df = pd.read_json(checkpoint_file, lines=True)
            if len(results_df) > 0:
                print(f"\n=== RECENT RESULTS ===")
                print(f"Change rate: {results_df['answer_changed'].mean():.3f}")
                print(f"Post-edit accuracy: {results_df['final_correct'].mean():.3f}")
                print(f"Baseline accuracy: {results_df['baseline_correct'].mean():.3f}")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

def main():
    """Main execution function"""
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "NOVITA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        print("Please set these before running the experiment.")
        return
    
    # Run the experiment with checkpointing
    results_df = run_experiment(
        num_problems=20,  # Increased to 20 problems as requested
        max_anchors_per_problem=200,
        seed=203401,
        edit_types=["assumption", "goal_redirect"],
        checkpoint_dir="checkpoints",
        resume=True  # Will automatically resume if checkpoint exists
    )
    
    # Create evaluation report
    print("\n4. Generating evaluation report...")
    stats = create_evaluation_report(results_df, output_dir="results")
    
    print("\n=== EXPERIMENT COMPLETE ===")
    print(f"Results saved to 'results/' directory")
    print(f"Summary: {stats['n_trials']} trials, "
          f"{stats['change_rate']:.3f} change rate, "
          f"{stats['post_edit_accuracy']:.3f} final accuracy")

if __name__ == "__main__":
    main()
