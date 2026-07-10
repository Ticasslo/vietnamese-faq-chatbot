import json
import re
import unicodedata
import pandas as pd
from pathlib import Path

#Cấu hình
INPUT_CSV = "FAQ_HCMUTE_paraphrases.csv"
INPUT_JSONL = "FAQ_HCMUTE_paraphrases.jsonl"
OUTPUT_CSV = "FAQ_HCMUTE_paraphrases_preprocessed.csv"
OUTPUT_JSONL = "FAQ_HCMUTE_paraphrases_preprocessed.jsonl"

# Pattern để phát hiện và loại bỏ emoji
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def sanitize_unicode(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def remove_emoji(text: str) -> str:
    if not text:
        return ""
    return EMOJI_PATTERN.sub("", text)


def preprocess(text: str, remove_emojis: bool = True) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = sanitize_unicode(text)
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    if remove_emojis:
        text = remove_emoji(text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # 1. Xử lý CSV
    if Path(INPUT_CSV).exists():
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
        df["sentence1"] = df["sentence1"].fillna("").astype(str).apply(preprocess)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Saved {len(df)} rows -> {OUTPUT_CSV}")
    else:
        print(f"File not found: {INPUT_CSV}")

    # 2. Xử lý JSONL
    if Path(INPUT_JSONL).exists():
        rows = []
        with open(INPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                obj["sentence1"] = preprocess(obj.get("sentence1", ""))
                rows.append(obj)
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for obj in rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} rows -> {OUTPUT_JSONL}")
    else:
        print(f"File not found: {INPUT_JSONL}")

    # Mẫu so sánh (ghi file để tránh lỗi encoding console)
    if Path(INPUT_CSV).exists() and Path(OUTPUT_CSV).exists():
        orig = pd.read_csv(INPUT_CSV)
        new_df = pd.read_csv(OUTPUT_CSV)
        with open("preprocess_paraphrases_sample.txt", "w", encoding="utf-8") as f:
            f.write("--- Sample (3 rows) ---\n")
            for i in range(min(3, len(orig))):
                f.write(f"  Original s1:  {str(orig.iloc[i]['sentence1'])[:60]}...\n")
                f.write(f"  Preprocessed: {str(new_df.iloc[i]['sentence1'])[:60]}...\n\n")
        print("Sample written to preprocess_paraphrases_sample.txt")


if __name__ == "__main__":
    main()
