# Đồ án NLP — Hệ thống hỏi đáp FAQ HCMUTE


## Nội dung đồ án (các “bài” / giai đoạn)

| Giai đoạn | Mô tả |
|-----------|--------|
| **1. Chuẩn bị & tiền xử lý FAQ** | Làm sạch CSV gốc (Unicode, emoji, khoảng trắng…). |
| **2. Sinh paraphrase** | Tạo thêm câu hỏi tương đương (Ollama) để tăng dữ liệu huấn luyện retrieval. |
| **3. Fine-tune PhoBERT retrieval** | Huấn luyện contrastive trên cặp câu (paraphrase), đánh giá P@k so với TF-IDF và PhoBERT gốc. |
| **4. Fine-tune sinh (tùy chọn)** | Notebook `finetune_mauV2.ipynb` + dữ liệu `finetune_qwen257B.jsonl` cho Qwen2.5 (LoRA). |
| **5. Demo / Web** | CLI (`demo.py`) hoặc giao diện Flask (`app.py`) với nhiều chế độ: chỉ retrieval, RAG, hybrid, chỉ sinh. |

---

## Cấu trúc thư mục & file

```
NLP_DoAn/
├── README.md                    # Tài liệu dự án (file này)
├── requirements.txt             # Thư viện Python
│
├── app.py                       # Web Flask (port 8080) — giao diện chat FAQ
├── demo.py                      # Logic retrieval, Chroma, Ollama/Local Qwen, prompt
├── train_phobert_faq.py         # Fine-tune PhoBERT + đánh giá + xuất model
├── finetune_mauV2.ipynb         # Notebook fine-tune Qwen2.5 (LoRA)
├── finetune_qwen257B.jsonl      # Dữ liệu instruction (Q/A theo faq_id) cho bước trên
│
├── FAQ_HCMUTE_preprocessed.csv  # FAQ đã tiền xử lý (dùng build index / demo)
├── Data.csv                     # Tên thay thế có thể dùng (xem demo.py)
│
├── templates/index.html         # Giao diện Flask
├── static/style.css             # Style cho app Flask
├── frontend/                    # Bản frontend tĩnh (HTML/CSS/JS) — có thể mở trực tiếp hoặc tùy chỉnh
│
├── Data/
│   ├── Prepare/                 # CSV/JSONL gốc & paraphrase ban đầu
│   ├── preprocessing_data/      # preprocess_faq.py, preprocess_paraphrases.py
│   ├── Generate_data/           # generate_paraphrases.py, generate_strong_paraphrases.py
│   └── After_Processing_Paraphrase/   # CSV/JSONL sau xử lý (ví dụ 10502 mẫu)
│
├── phobert_faq_retrieval/       # Model SentenceTransformer sau train (OUTPUT của train_phobert_faq.py)
├── qwen25_7b_instruct_lora_best/      # Adapter LoRA Qwen (dùng mode Hybrid / chỉ sinh trong demo)
├── faq_chroma_phobert_finetuned/      # ChromaDB persistent (tạo khi chạy demo/app)
├── data/                        # Train/val/test split (tạo bởi train_phobert_faq.py)
├── eval_phobert/                # Kết quả đánh giá, biểu đồ (khi chạy train đầy đủ)
└── phobert_faq_retrieval/eval/  # File eval kèm theo model (nếu có)
```

**Lưu ý:** Một số đường dẫn dữ liệu được **cố định trong code** (ví dụ `train_phobert_faq.py` trỏ tới `FAQ_HCMUTE_paraphrases_10502.csv` ở thư mục gốc; `demo.py` tìm `FAQ_HCMUTE_preprocessed.csv` hoặc `Data.csv`). Khi đổi vị trí file, cần sửa biến cấu hình tương ứng.

---

## Yêu cầu môi trường

- **Python** 3.10+ (khuyến nghị)
- **GPU NVIDIA + CUDA** (khuyến nghị cho train và cho Qwen LoRA); CPU vẫn chạy được retrieval nhưng chậm hơn
- **RAM/VRAM** đủ cho Sentence Transformers; Qwen 7B 4-bit cần GPU có VRAM hợp lý
- **Ollama** (tùy chọn): chỉ cần nếu bật chuẩn hóa câu hỏi bằng LLM hoặc chế độ RAG với `qwen3.5:9b`

---

## Cài đặt

```bash
cd c:\NLP_DoAn
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Tùy chọn:** Cài `matplotlib` nếu muốn script train vẽ biểu đồ loss/kết quả (`train_phobert_faq.py` có nhánh vẽ khi có thư viện).

---

## Luồng xử lý dữ liệu (tóm tắt)

1. **`Data/preprocessing_data/preprocess_faq.py`**  
   Đọc `FAQ_HCMUTE.csv` (trong thư mục chạy script) → xuất `FAQ_HCMUTE_preprocessed.csv`.

2. **`Data/Generate_data/generate_paraphrases.py`** (cần Ollama)  
   Sinh paraphrase → `FAQ_HCMUTE_paraphrases.jsonl` / `.csv`.

3. Các bước gộp / làm sạch thêm có thể dùng `preprocess_paraphrases.py`, `generate_strong_paraphrases.py` và thư mục `After_Processing_Paraphrase` tùy pipeline bạn đã thiết lập.

4. **`train_phobert_faq.py`**  
   Đọc file paraphrase (mặc định `FAQ_HCMUTE_paraphrases_10502.csv` / `.jsonl` nếu đặt đúng tên ở thư mục gốc) → huấn luyện → **`./phobert_faq_retrieval`**.

5. **`demo.py` / `app.py`**  
   Load encoder → embed FAQ từ CSV → lưu/lấy collection Chroma **`hcmute_faq`** trong `./faq_chroma_phobert_finetuned`.

---

## Huấn luyện PhoBERT retrieval

Đảm bảo file dữ liệu trùng với biến trong `train_phobert_faq.py` (`DATA_CSV`, `DATA_JSONL`, `DATA_FAQ`, …). Sau đó:

```bash
python train_phobert_faq.py
```

Kết quả: thư mục **`phobert_faq_retrieval`**, log/đánh giá trong **`eval_phobert/`** và **`data/`** (split train/val/test).

---

## Fine-tune Qwen (LoRA)

- Mở và chạy **`finetune_mauV2.ipynb`** theo từng ô (cấu hình base model, LoRA, dữ liệu).
- Dữ liệu mẫu định dạng instruction: **`finetune_qwen257B.jsonl`** (`instruction`, `input`, `output`, `faq_id`).
- Model adapter sau huấn luyện thường lưu vào thư mục dạng **`qwen25_7b_instruct_lora_best/`** (đã có trong repo nếu bạn commit sẵn).

---

## Chạy demo

### 1) Giao diện web (Flask)

```bash
python app.py
```

- Trình duyệt: **`http://127.0.0.1:8080`** (hoặc máy khác trong mạng: `http://<IP-máy>:8080` vì server bind `0.0.0.0`).
- Cần có **`phobert_faq_retrieval/`** và file FAQ (**`FAQ_HCMUTE_preprocessed.csv`** hoặc **`Data.csv`**) để build/load index.

### 2) Dòng lệnh (`demo.py`)

```bash
python demo.py
```

Chế độ mặc định và ngưỡng nằm trong **`demo.py`** (`DEMO_MODE`, `THRESHOLD`, `TOP_K`, …).

---

## Các chế độ trả lời (`DEMO_MODE` / chọn trên web)

| Mode | Ý nghĩa |
|------|--------|
| **1** | Chỉ **retrieval**: điểm cosine ≥ ngưỡng → trả lời từ FAQ; thấp hơn → thông báo ngoài phạm vi. |
| **2** | **RAG**: lấy top-k FAQ làm ngữ cảnh, sinh câu trả lời qua **Ollama** (`qwen3.5:9b`). |
| **3** | **Hybrid**: nếu điểm thấp, thử **Qwen2.5 + LoRA cục bộ** (không bắt buộc top-k). |
| **4** | Chỉ **mô hình sinh** (Qwen LoRA), có sampling — phù hợp thử nghiệm. |

**Cấu hình liên quan trong `demo.py`:**

- `USE_LLM_NORMALIZE`: bật chuẩn hóa câu hỏi qua Ollama trước khi embed.
- `OLLAMA_URL`, `OLLAMA_MODEL`: phục vụ normalize và mode RAG.
- `LOCAL_QWEN_LORA_DIR`, `LOCAL_QWEN_BASE_MODEL`: cho mode 3/4.

Cần chạy Ollama (ví dụ `ollama run qwen3.5:9b`) khi dùng normalize hoặc mode 2.

---

## Gợi ý khắc phục sự cố

| Hiện tượng | Hướng xử lý |
|------------|-------------|
| `Không tìm thấy ./phobert_faq_retrieval` | Chạy `train_phobert_faq.py` hoặc copy model vào đúng đường dẫn. |
| Lỗi không đọc được CSV FAQ | Đặt `FAQ_HCMUTE_preprocessed.csv` hoặc `Data.csv` ở thư mục gốc project (hoặc sửa `DATA_PATH` trong `demo.py`). |
| RAG / normalize lỗi kết nối | Kiểm tra Ollama đang chạy và `OLLAMA_URL` đúng (`http://localhost:11434/api/generate`). |
| Mode 3/4 lỗi LoRA | Kiểm tra tồn tại `qwen25_7b_instruct_lora_best/`, đủ RAM/VRAM, và dependency `peft`, `transformers`, `bitsandbytes` (Windows có thể cần cài đặt bổ sung tùy máy). |

---

## Tác giả / môn học
23110280 - HUỲNH THANH NHÂN

23110300 - TRƯƠNG TẤN SANG

23110327 - HUỲNH NGỌC THẮNGTHẮNG
