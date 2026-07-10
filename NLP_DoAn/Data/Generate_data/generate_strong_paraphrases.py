
import json
import re
import unicodedata
import time
import requests
import pandas as pd
from pathlib import Path

EMOJI_PATTERN = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+",
    flags=re.UNICODE,
)


def preprocess(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    text = EMOJI_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

#Cấu hình
INPUT_CSV = "FAQ_HCMUTE_preprocessed.csv"
OUTPUT_CSV = "FAQ_HCMUTE_strong_paraphrases.csv"
OUTPUT_JSONL = "FAQ_HCMUTE_strong_paraphrases.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:9b"

PARAPHRASES_PER_QUESTION = 4  
PARAPHRASE_REQUEST_N = 6  
MAX_RETRIES = 3
MAX_ROWS = None  
DELAY_SEC = 0.3

STRONG_PARAPHRASE_PROMPT = """Bạn là chuyên gia tạo dữ liệu paraphrase tiếng Việt chất lượng cao.

Nhiệm vụ: Viết {n_request} câu hỏi TIẾNG VIỆT có CÙNG Ý NGHĨA với câu gốc, nhưng phải diễn đạt KHÁC BIỆT tối đa.

YÊU CẦU BẮT BUỘC:
1. Giữ nguyên ý nghĩa, KHÔNG thêm hoặc suy diễn thông tin mới.
2. Diễn đạt lại bằng:
   - Từ đồng nghĩa, cách nói khác
   - Biến đổi cấu trúc câu (chủ động ↔ bị động, hỏi trực tiếp ↔ gián tiếp, đảo vị trí thành phần)
   - Có thể thay đổi cách hỏi (ai/cái gì/như thế nào/ra sao…)
3. HẠN CHẾ tối đa việc lặp lại từ khóa chính từ câu gốc (càng khác từ càng tốt).
4. Mỗi câu phải tự nhiên như người Việt nói thật, KHÔNG gượng ép.
5. Tránh lặp lại giữa các câu sinh ra (đa dạng tối đa).
6. Không giải thích, không thêm ghi chú.

ĐỊNH DẠNG OUTPUT:
- Chỉ trả về danh sách các câu
- Mỗi câu 1 dòng
- Đánh số: 1. 2. 3. ...

Câu gốc: {question}

Các cách hỏi tương đương:"""


def sanitize_unicode(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def is_valid_vietnamese_paraphrase(text: str) -> bool:
    if not text or len(text) < 8:
        return False
    text = text.strip()
    if re.search(r"xin lỗi|sorry|không rõ", text, re.I) and len(text) < 40:
        return False
    if re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", text):
        return False
    foreign = [r"\b(what|why|how|could|can you)\b", r"\b(warum|wieso)\b"]
    for pat in foreign:
        if re.search(pat, text, re.I):
            return False
    return True


def llm_strong_paraphrase(question: str, n: int = 3) -> list[str]:
    for attempt in range(MAX_RETRIES):
        try:
            prompt = STRONG_PARAPHRASE_PROMPT.format(
                question=question, n_request=PARAPHRASE_REQUEST_N
            )
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.8, "num_predict": 256},
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            result = sanitize_unicode(result)

            lines = []
            for line in result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line).strip()
                if line and is_valid_vietnamese_paraphrase(line) and line not in lines:
                    lines.append(line)

            if lines:
                return lines[:n]
            if attempt < MAX_RETRIES - 1:
                print(f"    [Lần {attempt+1}] Retry...")
                time.sleep(1)
        except Exception as e:
            print(f"    [LLM lỗi: {e}]")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return []
    return []


def main():
    path = Path(INPUT_CSV)
    if not path.exists():
        print(f"Không tìm thấy {INPUT_CSV}")
        return

    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["question", "faq_id"])
    # Mỗi `faq_id` là duy nhất trong `FAQ_HCMUTE_preprocessed.csv`
    unique = df.drop_duplicates(subset=["faq_id"])
    if MAX_ROWS:
        unique = unique.head(MAX_ROWS)
        print(f"Thử {MAX_ROWS} câu đầu.")
    print(f"Tổng {len(unique)} FAQ cần sinh paraphrase mạnh.")

    rows = []
    for idx, r in enumerate(unique.itertuples(index=False)):
        sentence2 = str(r.question).strip()
        faq_id = r.faq_id
        print(f"[{idx+1}/{len(unique)}] faq_id={faq_id}: {sentence2[:50]}...")
        paraphrases = llm_strong_paraphrase(sentence2, n=PARAPHRASES_PER_QUESTION)
        if not paraphrases:
            paraphrases = [sentence2]
        for p in paraphrases:
            rows.append({"sentence1": preprocess(p), "sentence2": sentence2, "faq_id": faq_id})
        time.sleep(DELAY_SEC)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nĐã lưu {len(out_df)} dòng → {OUTPUT_CSV}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, r in out_df.iterrows():
            f.write(json.dumps({"sentence1": r["sentence1"], "sentence2": r["sentence2"], "faq_id": r["faq_id"]}, ensure_ascii=False) + "\n")
    print(f"Đã lưu → {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
