#!/usr/bin/env python3
import csv, json, argparse, re

def norm(s: str) -> str:
    # basic normalization: strip spaces and collapse inner whitespace
    return re.sub(r"\s+", " ", s.strip())

def main(in_tsv: str, out_jsonl: str):
    n_rows = n_pairs = 0
    with open(in_tsv, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter="\t")
        # Expect columns: toxic, neutral1, neutral2, neutral3
        for row in reader:
            n_rows += 1
            toxic_raw = row.get("toxic", "") or ""
            toxic = norm(toxic_raw)
            if not toxic:
                continue

            # collect available neutrals
            neutrals = []
            for k in ("neutral1", "neutral2", "neutral3"):
                val = norm(row.get(k, "") or "")
                if val:
                    neutrals.append(val)

            # dedup neutrals and drop those identical to toxic
            seen = set()
            clean_neutrals = []
            for n in neutrals:
                if n and n != toxic and n not in seen:
                    seen.add(n)
                    clean_neutrals.append(n)

            if not clean_neutrals:
                continue

            # constant instruction as the prompt (works well for detox tasks)
            # If you prefer, you can set prompt = toxic (but instruction prompt is more standard for DPO).
            prompt = (
                "Rewrite the following sentence to be neutral and non-toxic while preserving meaning:\n\n"
                f"{toxic}"
            )

            for ntext in clean_neutrals:
                ex = {
                    "prompt": prompt,
                    "chosen": ntext,   # preferred
                    "rejected": toxic  # non-preferred
                }
                f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_pairs += 1

    print(f"Processed rows: {n_rows}")
    print(f"Wrote preference pairs: {n_pairs}")
    print(f"Output: {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    main("paradetox.tsv", "preference.jsonl")
