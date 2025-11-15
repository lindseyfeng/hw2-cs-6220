import csv, json, argparse, re

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def main(in_csv: str, out_jsonl: str):
    n_rows = n_pairs = 0
    with open(in_csv, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)  # comma-delimited by default

        for row in reader:
            n_rows += 1

            tox = row.get("toxic", "")
            neu = row.get("neutral", "")
            cleaned = row.get("cleaned_toxic", "")
            sent = row.get("sentiment", "")

            # Optionally normalize whitespace
            tox = norm(tox)
            neu = norm(neu)
            cleaned = norm(cleaned)

            # Skip if key fields are missing
            if not (tox and neu and cleaned):
                continue

            prompt = (
                "Rewrite the following sentence to be neutral and non-toxic while preserving its original meaning:\n\n"
                f"{cleaned}"
            )

            ex = {
                "prompt": prompt,
                "chosen": neu,       # neutral rewrite
                "rejected": cleaned  # original toxic (cleaned)
            }

            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_pairs += 1

    print(f"Processed rows: {n_rows}")
    print(f"Wrote preference pairs: {n_pairs}")
    print(f"Output: {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="para_df_with_sentiment.csv")
    ap.add_argument("--out_jsonl", default="preference.jsonl")
    args = ap.parse_args()

    main(args.in_csv, args.out_jsonl)
