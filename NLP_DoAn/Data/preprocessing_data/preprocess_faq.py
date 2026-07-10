
import re
import unicodedata
import pandas as pd
from pathlib import Path

#Cấu hình 
INPUT_CSV = "FAQ_HCMUTE.csv"  
OUTPUT_CSV = "FAQ_HCMUTE_preprocessed.csv"  

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Mặt cười, biểu cảm
    "\U0001F300-\U0001F5FF"  # Biểu tượng, hình vẽ
    "\U0001F680-\U0001F6FF"  # Phương tiện, bản đồ
    "\U0001F1E0-\U0001F1FF"  # Cờ các nước
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
    #Chuẩn hóa cơ bản câu hỏi
    if not text or not isinstance(text, str):
        return ""
    text = sanitize_unicode(text)           # Bỏ ký tự lỗi
    text = unicodedata.normalize("NFC", text)  # Chuẩn hóa Unicode
    text = text.strip()                     # Xóa khoảng trắng đầu/cuối
    text = re.sub(r"\s+", " ", text)        # Gộp nhiều space thành 1
    text = text.lower()                     # Chữ thường
    if remove_emojis:
        text = remove_emoji(text)
        text = re.sub(r"\s+", " ", text).strip()  # Dọn khoảng trắng thừa sau khi bỏ emoji
    return text


def main():
    path = Path(INPUT_CSV)
    if not path.exists():
        print(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(path)
    print(f"Read {len(df)} rows from {INPUT_CSV}")

    df["question"] = df["question"].fillna("").astype(str).apply(preprocess)

    out_path = Path(OUTPUT_CSV)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved to {OUTPUT_CSV}")

    orig = pd.read_csv(path)  #Đọc lại file gốc để so sánh
    with open("preprocess_sample.txt", "w", encoding="utf-8") as f:
        f.write("--- Mẫu so sánh (5 dòng đầu) ---\n")
        for i in range(min(5, len(df))):
            f.write(f"\n[ID {df.iloc[i]['faq_id']}]\n")
            f.write(f"  Gốc:       {str(orig.iloc[i]['question'])[:70]}...\n")
            f.write(f"  Chuẩn hóa:  {str(df.iloc[i]['question'])[:70]}...\n")
    print("Sample written to preprocess_sample.txt")  # Xem mẫu trước/sau chuẩn hóa


if __name__ == "__main__":
    main()
