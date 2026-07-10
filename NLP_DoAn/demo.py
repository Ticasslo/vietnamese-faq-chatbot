import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import chromadb
import torch
from pyvi import ViTokenizer

#Cấu hình
DATA_PATH = "FAQ_HCMUTE_preprocessed.csv"  
CHROMA_PATH = "./faq_chroma_phobert_finetuned"
MODEL_PATH = "./phobert_faq_retrieval"
THRESHOLD = 0.6  # Score >= 0.6: dùng RAG; < 0.6: out-of-scope
TOP_K = 5  

USE_LLM_NORMALIZE = True  

# Demo modes:
# 1) Chỉ Retrieval: score < THRESHOLD => xin lỗi không thể trả lời (không gọi LLM)
# 2) RAG: Retrieval top-k + gọi Ollama `qwen3.5:9b` sinh câu trả lời từ context
# 3) Hybrid: Retrieval top-1; nếu score < THRESHOLD => gọi mô hình sinh LoRA (không cần top-k)
DEMO_MODE = 1
OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Local generation (Qwen2.5 + LoRA)
USE_LOCAL_QWEN_GENERATE = True
LOCAL_QWEN_LORA_DIR = "./qwen25_7b_instruct_lora_best"
LOCAL_QWEN_BASE_MODEL = "unsloth/qwen2.5-7b-instruct-bnb-4bit"
LOCAL_QWEN_MAX_NEW_TOKENS = 512
LOCAL_QWEN_DO_SAMPLE = True
LOCAL_QWEN_TEMPERATURE = 0.7
LOCAL_QWEN_TOP_P = 0.9

OUT_OF_SCOPE_MSG = "Câu hỏi không liên quan hoặc không nằm trong phạm vi trả lời của tôi. Vui lòng hỏi về thông tin Trường Đại học Sư phạm Kỹ thuật TP.HCM."
CANNOT_ANSWER_MSG = "Xin lỗi, tôi không thể trả lời câu hỏi này."

LLM_PROMPT_NORMALIZE = """Bạn là trợ lý chuẩn hóa câu hỏi tiếng Việt. Nhiệm vụ: viết lại câu hỏi cho đúng chính tả, ngắn gọn, dễ tra cứu FAQ.
Quy tắc:
- Sửa lỗi chính tả
- Bỏ từ dư thừa (vậy, ạ, nha, cho em/mình hỏi, dạ, vâng...)
- Chuyển câu nói thành câu hỏi rõ ràng, dễ hiểu
- Giữ nguyên ý nghĩa, không thêm thông tin không có trong câu gốc

Chỉ trả về ĐÚNG MỘT dòng là câu hỏi đã sửa, không giải thích.

Câu hỏi gốc: {question}
Câu hỏi chuẩn:"""

LLM_PROMPT_RAG = """Bạn là trợ lý FAQ của Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE). Dựa CHỈ vào thông tin dưới đây để trả lời câu hỏi.

[THÔNG TIN FAQ]
{context}
[/THÔNG TIN]

Câu hỏi: {question}

Yêu cầu:
- Trả lời ngắn gọn, chính xác, bằng TIẾNG VIỆT
- Chỉ dùng thông tin trong [THÔNG TIN FAQ], không bịa đặt
- Suy luận từ ngữ cảnh
- Nếu thông tin không đủ để trả lời, nói rõ

Trả lời:"""


def sanitize_unicode(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def word_segment(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return ViTokenizer.tokenize(text)


def normalize_question(text: str) -> str:
    #Chuẩn hóa câu hỏi: strip, collapse spaces, unicode NFC, lowercase.
    if not text or not isinstance(text, str):
        return ""
    text = sanitize_unicode(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    return text


def _get_encoder_input(texts):
    lst = texts if isinstance(texts, list) else [texts]
    return [word_segment(normalize_question(t)) for t in lst]


class FinetunedEncoder:

    def __init__(self, model_path: str = MODEL_PATH, device: str = None):
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path, device=self.device)

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        preprocessed = _get_encoder_input(texts)
        return self.model.encode(preprocessed, normalize_embeddings=normalize_embeddings, batch_size=batch_size)


def _ollama_generate(prompt: str, max_tokens: int = 512) -> str:
    try:
        import requests
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.2, "num_predict": max_tokens},
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        return sanitize_unicode(result)
    except Exception as e:
        raise RuntimeError(f"Ollama lỗi: {e}") from e



_local_qwen_model = None
_local_qwen_tokenizer = None


def load_local_qwen_lora():
    global _local_qwen_model, _local_qwen_tokenizer
    if _local_qwen_model is not None and _local_qwen_tokenizer is not None:
        return

    if not Path(LOCAL_QWEN_LORA_DIR).exists():
        raise FileNotFoundError(f"Không tìm thấy LoRA dir: {LOCAL_QWEN_LORA_DIR}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "Thiếu dependency để load local LoRA. Cài thêm `pip install peft transformers`."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [Local LoRA] Đang load tokenizer từ {LOCAL_QWEN_LORA_DIR}...")
    _local_qwen_tokenizer = AutoTokenizer.from_pretrained(LOCAL_QWEN_LORA_DIR, use_fast=False)

    # Base model
    print(f"  [Local LoRA] Đang load base model: {LOCAL_QWEN_BASE_MODEL} ({device})...")
    base_model = None
    load_err = None

    try:
        from transformers import BitsAndBytesConfig  # type: ignore

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_QWEN_BASE_MODEL,
            device_map="auto",
            quantization_config=bnb_config,
        )
    except Exception as e:
        load_err = e

    if base_model is None:
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_QWEN_BASE_MODEL,
                device_map="auto",
                load_in_4bit=True,
            )
            load_err = None
        except Exception as e:
            load_err = e

    if base_model is None:
        print(f"  [Local LoRA] [WARN] Không load được 4-bit ({load_err}). Thử load thường...")
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_QWEN_BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else None,
        )

    print("  [Local LoRA] Đang load LoRA adapter...")
    _local_qwen_model = PeftModel.from_pretrained(base_model, LOCAL_QWEN_LORA_DIR)
    _local_qwen_model.eval()


def _local_generate_from_prompt(
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = LOCAL_QWEN_TEMPERATURE,
    top_p: float = LOCAL_QWEN_TOP_P,
) -> str:
    if _local_qwen_model is None or _local_qwen_tokenizer is None:
        raise RuntimeError("Local Qwen LoRA chưa được load. Gọi load_local_qwen_lora() trước.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [{"role": "user", "content": prompt}]
    text_prompt = _local_qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _local_qwen_tokenizer(
        text_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=32768,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_ids = _local_qwen_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            use_cache=False,  # giảm rủi ro lỗi attention mask shape
            pad_token_id=_local_qwen_tokenizer.eos_token_id,
        )

    gen_tokens = gen_ids[0, input_len:]
    gen_text = _local_qwen_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return sanitize_unicode(gen_text)


def local_generate_answer(
    question: str,
    max_new_tokens: int = LOCAL_QWEN_MAX_NEW_TOKENS,
    do_sample: bool = LOCAL_QWEN_DO_SAMPLE,
    temperature: float = LOCAL_QWEN_TEMPERATURE,
    top_p: float = LOCAL_QWEN_TOP_P,
) -> str:

    load_local_qwen_lora()
    return _local_generate_from_prompt(
        question,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def llm_normalize(question: str) -> str:
    try:
        prompt = LLM_PROMPT_NORMALIZE.format(question=question)
        result = _ollama_generate(prompt, max_tokens=128)
        result = result.split("\n")[0].strip()
        result = re.sub(r"^(câu hỏi chuẩn\s*:?\s*|dạ\s*,?\s*|vâng\s*,?\s*|ok\s*,?\s*|[\d\.\-\*]+\s*)", "", result, flags=re.I).strip()
        return result if result else question
    except Exception as e:
        print(f"  [LLM lỗi: {e}] → dùng câu gốc (kiểm tra Ollama: ollama run {OLLAMA_MODEL})")
        return question


def llm_generate_answer(question: str, top_k_metadatas: list, use_local: bool = False) -> str:
    context_parts = []
    for i, meta in enumerate(top_k_metadatas, 1):
        q = meta.get("question", "")
        a = meta.get("answer", "")
        if q and a:
            context_parts.append(f"{i}. Q: {q}\n   A: {a}")
    context = "\n\n".join(context_parts) if context_parts else "(Không có thông tin)"
    prompt = LLM_PROMPT_RAG.format(context=context, question=question)
    try:
        if use_local:
            # Local generation
            result = _local_generate_from_prompt(
                prompt,
                max_new_tokens=LOCAL_QWEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        else:
            # Ollama generation
            result = _ollama_generate(prompt, max_tokens=512)

        result = result.strip()
        result = re.sub(r"^(trả lời\s*:?\s*|đáp án\s*:?\s*)", "", result, flags=re.I).strip()
        return result if result else top_k_metadatas[0].get("answer", "")
    except Exception as e:
        print(f"  [LLM RAG lỗi: {e}] → trả về answer top-1")
        return top_k_metadatas[0].get("answer", "") if top_k_metadatas else ""


def _load_faq_data():
    for path in [DATA_PATH, "FAQ_HCMUTE_preprocessed.csv", "Data.csv"]:
        if Path(path).exists():
            df = pd.read_csv(path, encoding="utf-8")
            q_col = "question" if "question" in df.columns else "Question"
            a_col = "answer" if "answer" in df.columns else "Answer"
            questions_norm = [normalize_question(q) for q in df[q_col]]
            questions_orig = df[q_col].str.strip().tolist()
            answers = df[a_col].astype(str).tolist()
            return questions_norm, questions_orig, answers, len(questions_norm)
    raise FileNotFoundError(f"Không tìm thấy {DATA_PATH} hoặc FAQ_HCMUTE_preprocessed.csv")


def build_index(model, client):
    print("Đang load FAQ...")
    questions_norm, questions_orig, answers, n_faq = _load_faq_data()

    print(f"Đang embed {n_faq} câu hỏi...")
    embeddings = model.encode(questions_norm, normalize_embeddings=True, batch_size=32)

    collection = client.get_or_create_collection("hcmute_faq", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"faq_{i}" for i in range(n_faq)],
        embeddings=embeddings.tolist(),
        metadatas=[{"answer": ans, "stt": str(i + 1), "question": q_orig} for i, (ans, q_orig) in enumerate(zip(answers, questions_orig))],
    )
    print("Đã build xong index.\n")
    return collection, n_faq


def search_vector(collection, model, n_faq, question, top_k=TOP_K):
    emb = model.encode([question], normalize_embeddings=True)
    results = collection.query(
        query_embeddings=emb.tolist(),
        n_results=min(top_k, n_faq),
        include=["metadatas", "distances"],
    )
    return results


def main():
    print("=" * 60)
    print("  Demo FAQ Retrieval (PhoBERT fine-tuned)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(MODEL_PATH).exists():
        print(f"\nKhông tìm thấy {MODEL_PATH}. Chạy train_phobert_faq.py trước.")
        return

    print(f"\nĐang load model fine-tuned từ {MODEL_PATH} trên {device}...")
    model = FinetunedEncoder(MODEL_PATH, device=device)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection("hcmute_faq")
        n_faq = collection.count()
        if n_faq == 0:
            collection, n_faq = build_index(model, client)
        else:
            print(f"Đã load sẵn index ({n_faq} FAQ).\n")
    except Exception:
        collection, n_faq = build_index(model, client)

    demo_mode = DEMO_MODE
    mode = []
    if USE_LLM_NORMALIZE:
        mode.append("LLM chuẩn hóa")
    if demo_mode == 1:
        mode.append("Chỉ Retrieval (score<threshold => xin lỗi)")
    elif demo_mode == 2:
        mode.append("RAG (Retrieval top-k + qwen3.5:9b sinh trả lời)")
    elif demo_mode == 3:
        mode.append("Hybrid (score<threshold => local LoRA sinh, không cần top-k)")
    elif demo_mode == 4:
        mode.append("Chỉ mô hình sinh (Qwen2.5 + LoRA), có sampling để test lặp)")
    else:
        raise ValueError(f"DEMO_MODE không hợp lệ: {demo_mode}. Chọn 1/2/3/4.")

    suffix = f" [{', '.join(mode)}]" if mode else ""
    print(f"Nhập câu hỏi để tìm (Enter trống để thoát){suffix}:\n")

    # Load local Qwen LoRA chỉ khi cần (mode 3/4).
    if demo_mode in (3, 4):
        try:
            load_local_qwen_lora()
        except Exception as e:
            print(f"  [Local LoRA] Không thể load ({e}). Mode 3 sẽ fallback: luôn trả OUT_OF_SCOPE_MSG.")

    while True:
        try:
            raw_question = input("Bạn hỏi: ").strip()
            if not raw_question:
                print("Thoát.")
                break

            question = raw_question
            if USE_LLM_NORMALIZE:
                print("  Đang chuẩn hóa bằng LLM...")
                question = llm_normalize(question)
                print(f"  → LLM chuẩn hóa: {question}")
            question = normalize_question(question)
            print(f"  Câu hỏi đã chuẩn hóa: {question}")

            # Mode 4: chỉ mô hình sinh (không retrieval/top-k)
            if demo_mode == 4:
                try:
                    print("  Đang sinh câu trả lời bằng Qwen LoRA (chỉ mô hình sinh)...")
                    answer = local_generate_answer(
                        question,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    print(f"\n>>> Trả lời: {answer}")
                except Exception:
                    print(f"\n>>> {OUT_OF_SCOPE_MSG}")
                print("\n" + "-" * 40)
                continue

            results = search_vector(collection, model, n_faq, question, top_k=TOP_K)
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            best_score = 1 - distances[0]

            print("\n--- Top 5 retrieval ---")
            for i, (dist, meta) in enumerate(zip(distances, metadatas), 1):
                score = 1 - dist
                status = "✓" if score >= THRESHOLD else "✗"
                print(f"  [{i}] Score: {score:.3f} {status} | {meta.get('question', 'N/A')[:60]}...")

            # Chọn hành vi theo demo_mode
            if demo_mode == 1:
                if best_score >= THRESHOLD:
                    answer = metadatas[0]["answer"]
                else:
                    answer = CANNOT_ANSWER_MSG
                print(f"\n>>> Trả lời: {answer}")

            elif demo_mode == 2:
                print("  Đang sinh câu trả lời bằng LLM (RAG với qwen3.5:9b)...")
                answer = llm_generate_answer(question, metadatas, use_local=False)
                print(f"\n>>> Trả lời: {answer}")

            elif demo_mode == 3:
                # Retrieval top-1; nếu dưới ngưỡng thì chuyển sang local LoRA sinh
                if best_score >= THRESHOLD:
                    answer = metadatas[0]["answer"]
                    print(f"\n>>> Trả lời: {answer}")
                else:
                    try:
                        print("  Score thấp: chuyển sang local LoRA (không cần top-k)...")
                        answer = _local_generate_from_prompt(
                            question,
                            max_new_tokens=LOCAL_QWEN_MAX_NEW_TOKENS,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        print(f"\n>>> Trả lời: {answer}")
                    except Exception as e:
                        print(f"  [Hybrid debug] Local LoRA lỗi: {e}")
                        print(f"\n>>> {OUT_OF_SCOPE_MSG}")

            print("\n" + "-" * 40)

        except KeyboardInterrupt:
            print("\nThoát.")
            break


if __name__ == "__main__":
    main()
