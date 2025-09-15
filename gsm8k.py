import os
import re
import json
import random
import argparse
import unicodedata
from typing import List, Dict
from collections import Counter

import datasets
from transformers import set_seed
from vllm import LLM, SamplingParams

###############################################################################
#                               Format Prompts
###############################################################################

def format_prompt(question: str, prompt_template: str = None) -> str:
    """
    Example: a short chain-of-thought style prompt or any custom style
    you prefer for Alpaca Eval.
    """
    if prompt_template:
        return prompt_template.replace("{question}", question)
    return f"""
Instruction: {question}
"""

def format_prompt_noncot(question: str, prompt_template: str = None) -> str:
    """
    Example: a direct prompt (no chain-of-thought).
    """
    if prompt_template:
        return prompt_template.replace("{question}", question)
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

Instruction: {question}
"""

###############################################################################
#                            Generation Helper (vLLM)
###############################################################################

def vllm_generate(
    model: LLM,
    prompt: str,
    n_responses: int,
    max_length: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = False
) -> str:
    """
    A simple wrapper around vLLM's model.generate() call.
    Returns just the first generation's text (you can adapt if you want multiple).
    """
    # Build sampling params
    sampling_params = SamplingParams(
        n=n_responses, 
        temperature=temperature if do_sample else 0.0,
        max_tokens=max_length,
        top_p=top_p if do_sample else 1.0,
    )
    request_outputs = model.generate([prompt], sampling_params)
    
    # Grab the first RequestOutput (one request == one prompt)
    request_output = request_outputs[0]

    # request_output.outputs is a list of CompletionOutput objects
    completions = request_output.outputs

    if n_responses == 1:
        # Return the single completion's text
        return completions[0].text
    else:
        # Return a list of all completion texts
        return [comp.text for comp in completions]

###############################################################################
#                            Main Generation Logic
###############################################################################

def batch_generate_responses(args):
    """
    Load the Alpaca Eval dataset and generate vLLM completions for each example.
    """

    # 1. Load the Alpaca Eval dataset
    dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")
    eval_data = dataset["eval"]

    # If a maximum number of samples was requested, slice the data
    if args.max_samples:
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

    # 2. Build prompts
    if args.no_cot:
        # Non-chain-of-thought style prompts
        prompts = [format_prompt_noncot(item["instruction"], args.prompt_template) for item in eval_data]
    else:
        # Chain-of-thought style prompts
        prompts = [format_prompt(item["instruction"], args.prompt_template) for item in eval_data]

    # 3. Initialize the vLLM model
    print(f"Loading vLLM model {args.model_name}...")
    model = LLM(args.model_name, dtype="auto")

    # If you want reproducible sampling, set the seed
    set_seed(args.seed)
    random.seed(args.seed)

    print(f"Generating responses for {len(prompts)} samples...")
    results = []

    for i, prompt in enumerate(prompts):
        # Generate a single response for each prompt using vLLM
        completion = vllm_generate(
            model=model,
            prompt=prompt,
            n_responses=args.n_responses,  # you'll define n_responses in parse_args
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )

        # Store the completion
        results.append({
            "instruction": eval_data[i]["instruction"],
            "output": completion
        })

        # Simple progress log
        if (i + 1) % args.log_interval == 0:
            print(f"Processed {i+1} / {len(prompts)}...")

    # 4. Save results if requested
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)

    print("Done.")
    return results

###############################################################################
#                              Argument Parsing
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate responses for Alpaca Eval dataset using a vLLM model.'
    )
    parser.add_argument('--model_name', type=str, default="gpt2", 
                        help='vLLM model name or path (e.g. gpt2).')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to generate.')
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for sampling.')
    parser.add_argument('--do_sample', action='store_true', 
                        help='Use sampling. If false, temperature=0.0 is greedy.')
    parser.add_argument('--no_cot', action='store_true', 
                        help='Whether to use a no-CoT prompt style.')
    parser.add_argument('--top_p', type=float, default=1.0, 
                        help='Top-p for nucleus sampling.')
    parser.add_argument('--max_length', type=int, default=512, 
                        help='Max tokens for each generation.')
    parser.add_argument('--log_interval', type=int, default=10, 
                        help='Logging interval.')
    parser.add_argument('--prompt_template', type=str, default=None, 
                        help='Custom prompt template with {question} placeholder.')
    parser.add_argument('--save_results', type=str, default=None, 
                        help='Path to save results JSON.')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed.')
    # Add n_responses to control how many completions per prompt
    parser.add_argument('--n_responses', type=int, default=1, 
                        help='Number of responses to generate for each prompt.')
    return parser.parse_args()

###############################################################################
#                                  Main
###############################################################################

if __name__ == "__main__":
    args = parse_args()
    batch_generate_responses(args)
