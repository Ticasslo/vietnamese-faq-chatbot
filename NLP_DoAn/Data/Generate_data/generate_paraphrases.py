import json
import re
import csv
import time
import requests
import pandas as pd

#Cấu hình
DATA_PATH = "FAQ_HCMUTE_preprocessed.csv" #data gốc đã preprocess
OUTPUT_JSONL = "FAQ_HCMUTE_paraphrases.jsonl"    
OUTPUT_CSV = "FAQ_HCMUTE_paraphrases.csv"         
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:9b" 
PARAPHRASES_PER_QUESTION = 5 
PARAPHRASE_REQUEST_N = 6  
MAX_RETRIES = 3  
MAX_FAQS = 50  
DELAY_SEC = 0.2  

PARAPHRASE_PROMPT = """Bạn là trợ lý sinh dữ liệu. Nhiệm vụ: viết {n_request} câu hỏi KHÁC NHAU bằng TIẾNG VIỆT, cùng ý nghĩa với câu dưới đây.

BẮT BUỘC:
- Chỉ viết bằng tiếng Việt (không tiếng Anh, Đức, Trung, v.v.)
- Không trộn từ nước ngoài vào câu (trừ thuật ngữ chuyên ngành đã phổ biến)
- Mỗi câu là cách hỏi mà sinh viên Việt Nam thực tế có thể dùng
- Giữ nguyên ý nghĩa, đa dạng: formal/informal, dài/ngắn
- Mỗi câu trên 1 dòng, đánh số 1. 2. 3. ...

Câu gốc: {question}

Các cách hỏi tương đương (chỉ tiếng Việt):"""


def sanitize_unicode(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def is_valid_vietnamese_paraphrase(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    text = text.strip()
    #Loại câu xin lỗi / không liên quan
    if re.search(r"xin lỗi|sorry|không rõ|không hiểu", text, re.I) and len(text) < 40:
        return False
    # Loại có ký tự CJK (Trung/Nhật/Hàn)
    if re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", text):
        return False
    # Loại có từ tiếng Anh/Đức phổ biến
    foreign_patterns = [
        r"\b(what|why|how|could|can you|please explain|tell me)\b",
        r"\b(warum|wieso|welche|come si|wat bedekt)\b",
        r"\b(avoid|egers:|ullmann)\b",
        r"\b(already|unlike|ready|study)\b",
        r"cóporter|bidden|mengapa|대\s*học",
        r"\d+iency",  # rõ6iency
    ]
    for pat in foreign_patterns:
        if re.search(pat, text, re.I):
            return False
    return True

def llm_paraphrase(question: str, n: int = 3) -> list[str]:
    for attempt in range(MAX_RETRIES):
        try:
            prompt = PARAPHRASE_PROMPT.format(question=question, n_request=PARAPHRASE_REQUEST_N)
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.7, "num_predict": 256},
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            result = sanitize_unicode(result)

            #Parse và kiểm tra tiếng Việt
            lines = []
            for line in result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line).strip()
                if line and is_valid_vietnamese_paraphrase(line):
                    lines.append(line)

            if lines:
                return lines[:n]
            #Không có câu hợp lệ thì yêu cầu lại
            if attempt < MAX_RETRIES - 1:
                print(f"    [Lần {attempt+1}] LLM trả khác tiếng Việt → yêu cầu lại...")
                time.sleep(1)  # Đợi chút trước khi retry
        except Exception as e:
            print(f"    [LLM lỗi: {e}]")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return []
    return []


def main():
    print(f"Đang load {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    questions = df["question"].tolist()
    faq_ids = df["faq_id"].tolist()

    if MAX_FAQS:
        questions = questions[:MAX_FAQS]
        faq_ids = faq_ids[:MAX_FAQS]
        print(f"Chế độ thử: chỉ xử lý {MAX_FAQS} FAQ đầu.")
        print(f"Check {OUTPUT_JSONL} xong, nếu ổn thì đổi MAX_FAQS=None để chạy full.")
    else:
        print(f"Xử lý toàn bộ {len(questions)} FAQ.")

    pairs = []  #(sentence1, sentence2)
    for i, (faq_id, q) in enumerate(zip(faq_ids, questions)):
        print(f"[{i+1}/{len(questions)}] faq_id={faq_id}: {q[:50]}...")
        paraphrases = llm_paraphrase(q, n=PARAPHRASES_PER_QUESTION)
        if not paraphrases:
            print(f"[!] Không có paraphrase hợp lệ → dùng câu gốc.")
            paraphrases = [q]  # Fallback: ít nhất 1 cặp (gốc, gốc)
        for p in paraphrases:
            pairs.append({"sentence1": p, "sentence2": q, "faq_id": faq_id})
        time.sleep(DELAY_SEC)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps({"sentence1": row["sentence1"], "sentence2": row["sentence2"]}, ensure_ascii=False) + "\n")

    # 2. CSV — dự phòng
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sentence1", "sentence2", "faq_id"])
        w.writeheader()
        w.writerows(pairs)

    print(f"\nĐã lưu {len(pairs)} cặp (sentence1, sentence2) → {OUTPUT_JSONL}, {OUTPUT_CSV}")
    print("Định dạng: sentence1 = paraphrase (query), sentence2 = FAQ gốc (positive).")


if __name__ == "__main__":
    main()
