import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# -----------------------------
# CoT PROMPT
# -----------------------------
def cot_prompt(question: str, cot: bool) -> str:
    """
    Load a manual prompt template from file, replace {question} with the actual question,
    and enforce ###ANSWER format at the end.
    """

    with open("gsm8k-few-shot.txt", "r", encoding="utf-8") as f:
        template = f.read()
    
    if cot:
        with open("gsm8k-few-shot-cot.txt", "r", encoding="utf-8") as f:
            template = f.read()
            

    filled_prompt = template.replace("{question}", question)

    return filled_prompt

def get_final_answer(text: str) -> str:
    """
    Extract the final answer from generated text,
    assuming it's placed right after '###'.

    Args:
        text (str): Model-generated output.

    Returns:
        str: The stripped final answer (string).
    """
    if "###" in text:
        return text.split("###")[-1].strip()
    return text.strip()


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def infer(model, tokenizer, prompt, max_tokens=200, temperature=0.7, top_p=1.0):
    """
    Run inference with conditional decoding:
    - If temperature > 0: sampling (with top-p).
    - If temperature == 0: greedy decoding.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_tokens,
    }

    if temperature > 0:
        gen_kwargs.update({
            "temperature": temperature,
            "do_sample": True,
            "top_p": top_p
        })
    else:
        gen_kwargs.update({
            "do_sample": False  # greedy decoding
        })

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return (text, get_final_answer(text))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

    # Load queries
    with open("gsm8k.json", "r") as f:
        queries = json.load(f)  # Expecting list of dicts: {"question":..., "answer":...}

    results = []

    for q in queries:
        prompt = cot_prompt(q["question"], args.cot)
        
        pred_text, answer = infer(model, tokenizer, prompt, max_tokens=args.max_length, temperature=args.temperature, top_p = args.top_p )
        results.append({
            "question": q["question"],
            "ground_truth": q["answer"],
            "prediction": pred_text, 
            "final_answer" : answer, 
            "is_correct" : answer == q["final_answer"]
        })

    # Evaluation
    total = len(results)
    num_correct = sum(1 for r in results if r["is_correct"])
    accuracy = num_correct / total
    
    print(f"Accuracy: {num_correct}/{total} = {accuracy:.2%}")

    # Save results

    if args.cot:
        with open(f"gsm8k_cot_results_temp{args.temperature}_topp_{args.top_p}.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        with open(f"gsm8k_baseline_results_temp{args.temperature}_topp_{args.top_p}.json", "w") as f:
            json.dump(results, f, indent=2)
        


# -----------------------------
# ARGPARSE
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference for GSM8K using CoT.")
    parser.add_argument("--model_name", type=str, required=True, help="Huggingface model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling Top p")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--cot", action="store_true", help="do cot?")
    args = parser.parse_args()
    
    main(args)
