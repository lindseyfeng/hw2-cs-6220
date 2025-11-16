import csv, json, argparse, re

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def main(in_csv: str, out_jsonl: str, use_sentiment: bool = False):
    n_rows = n_pairs = 0
    with open(in_csv, "r", encoding="utf-8") as f_in, open(out_jsonl, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)  # comma-delimited by default

        for row in reader:
            n_rows += 1

            tox = row.get("toxic", "")
            neu = row.get("neutral", "")
            cleaned = row.get("cleaned_toxic", "")
            sent = row.get("sentiment", "")

            # Normalize whitespace
            tox = norm(tox)
            neu = norm(neu)
            cleaned = norm(cleaned)
            sent = norm(sent)

            # Skip if key fields are missing
            if not (tox and neu and cleaned):
                continue

            # Determine which sentiment string to preserve
            if use_sentiment and sent:
                sentiment_label = sent.lower()
            else:
                sentiment_label = "negative"

            # Optional prefix line mentioning sentiment
            sentiment_str = f"Sentiment: {sentiment_label}\n" if use_sentiment and sent else ""

            text = cleaned  # toxic input text

            prompt = (
                f"Rewrite the following toxic text in a neutral style, detoxify it while preserving its "
                f"{sentiment_label} sentiment:\n"
                f"{sentiment_str}{text}\nDetoxified:"
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
    ap.add_argument(
        "--use-sentiment",
        action="store_true",
        help="If set, preserve each row's sentiment label (positive/negative/etc.) instead of defaulting to negative."
    )
    args = ap.parse_args()

    main(args.in_csv, args.out_jsonl, args.use_sentiment)
