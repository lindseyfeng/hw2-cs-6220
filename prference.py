#!/usr/bin/env python3
import csv, json, argparse, re

def norm(s: str) -> str:
    # basic normalization: strip spaces and collapse inner whitespace
    return re.sub(r"\s+", " ", s.strip())

def main(in_tsv: str, out_jsonl: str):
    n_rows = n_pairs = 0
    with open(in_tsv, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            n_rows += 1
            toxic_raw = row.get("cleaned_toxic", "") or ""
            neutral = row.get("neutral", "") or ""
            
            prompt = (
                "Rewrite the following sentence to be neutral and non-toxic while preserving meaning:\n\n"
                f"{toxic_raw}"
            )

            ex = {
                    "prompt": prompt,
                    "chosen": neutral,   # preferred
                    "rejected": toxic_raw  # non-preferred
                }
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_pairs += 1

    print(f"Processed rows: {n_rows}")
    print(f"Wrote preference pairs: {n_pairs}")
    print(f"Output: {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    main("para_df_with_sentiment.csv", "preference.jsonl")
