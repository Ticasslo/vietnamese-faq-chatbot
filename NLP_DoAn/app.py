from pathlib import Path

from flask import Flask, render_template, request, jsonify

from demo import (
    FinetunedEncoder,
    CHROMA_PATH,
    MODEL_PATH,
    THRESHOLD,
    TOP_K,
    USE_LLM_NORMALIZE,
    OUT_OF_SCOPE_MSG,
    CANNOT_ANSWER_MSG,
    DEMO_MODE,
    local_generate_answer,
    llm_normalize,
    llm_generate_answer,
    normalize_question,
    search_vector,
    build_index,
)

import chromadb
import torch

app = Flask(__name__)

_model = None
_collection = None
_n_faq = 0


def init_app():
    global _model, _collection, _n_faq
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Không tìm thấy {MODEL_PATH}. Chạy train_phobert_faq.py trước.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = FinetunedEncoder(MODEL_PATH, device=device)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        _collection = client.get_collection("hcmute_faq")
        _n_faq = _collection.count()
        if _n_faq == 0:
            _collection, _n_faq = build_index(_model, client)
    except Exception:
        _collection, _n_faq = build_index(_model, client)
    return _n_faq


@app.route("/")
def index():
    return render_template(
        "index.html",
        n_faq=_n_faq,
        top_k=TOP_K,
        threshold=THRESHOLD,
        vector_model="PhoBERT fine-tuned",
        cosine_metric="Cosine similarity",
    )


@app.route("/ask", methods=["POST"])
def ask():
    if _model is None or _collection is None:
        return jsonify({"error": "Model chưa được load. Khởi động lại app."}), 500
    data = request.get_json() or {}
    raw = (data.get("question") or request.form.get("question") or "").strip()
    mode = data.get("mode", DEMO_MODE)
    try:
        mode = int(mode)
    except Exception:
        mode = DEMO_MODE
    if mode not in (1, 2, 3, 4):
        mode = DEMO_MODE
    if not raw:
        return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400

    question = raw
    if USE_LLM_NORMALIZE:
        try:
            question = llm_normalize(question)
        except Exception:
            pass
    question = normalize_question(question)

    # Mode 4: chỉ mô hình sinh (không retrieval/top-k)
    if mode == 4:
        try:
            answer = local_generate_answer(question, do_sample=True, temperature=0.7, top_p=0.9)
            out_of_scope = False
        except Exception:
            answer = OUT_OF_SCOPE_MSG
            out_of_scope = True
        return jsonify({
            "question_raw": raw,
            "question_normalized": question,
            "answer": answer,
            "out_of_scope": out_of_scope,
            "mode": mode,
            "best_score": None,
            "top_k": [],
        })

    results = search_vector(_collection, _model, _n_faq, question, top_k=TOP_K)
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    best_score = float(1 - distances[0])

    top_k_items = []
    for i, (dist, meta) in enumerate(zip(distances, metadatas), 1):
        top_k_items.append({
            "rank": i,
            "score": round(1 - dist, 3),
            "question": meta.get("question", ""),
            "answer": meta.get("answer", ""),
        })

    if mode == 1:
        if best_score >= THRESHOLD:
            answer = metadatas[0].get("answer", "")
            out_of_scope = False
        else:
            answer = CANNOT_ANSWER_MSG
            out_of_scope = True

    elif mode == 2:
        # Luôn RAG theo top-k
        try:
            answer = llm_generate_answer(question, metadatas, use_local=False)
        except Exception:
            answer = metadatas[0].get("answer", "")
        out_of_scope = False

    else:  # mode == 3
        if best_score >= THRESHOLD:
            answer = metadatas[0].get("answer", "")
            out_of_scope = False
        else:
            # fallback sang mô hình sinh LoRA, không cần context/top-k
            try:
                answer = local_generate_answer(question, do_sample=True, temperature=0.7, top_p=0.9)
                out_of_scope = False
            except Exception as e:
                answer = (
                    f"{OUT_OF_SCOPE_MSG}\n"
                    f"[Hybrid debug] Local LoRA lỗi: {str(e)}"
                )
                out_of_scope = True

    return jsonify({
        "question_raw": raw,
        "question_normalized": question,
        "answer": answer,
        "out_of_scope": out_of_scope,
        "mode": mode,
        "best_score": best_score,
        "top_k": top_k_items,
    })


@app.route("/chat", methods=["POST"])
def chat():
    if _model is None or _collection is None:
        return jsonify({"error": "Model chưa được load. Khởi động lại app."}), 500

    data = request.get_json() or {}
    raw = (data.get("question") or request.form.get("question") or "").strip()
    mode = data.get("mode", DEMO_MODE)
    try:
        mode = int(mode)
    except Exception:
        mode = DEMO_MODE
    if mode not in (1, 2, 3, 4):
        mode = DEMO_MODE

    if not raw:
        return jsonify({"error": "Vui lòng nhập câu hỏi."}), 400

    question = raw
    if USE_LLM_NORMALIZE:
        try:
            question = llm_normalize(question)
        except Exception:
            pass
    question = normalize_question(question)

    if mode == 4:
        try:
            answer = local_generate_answer(question, do_sample=True, temperature=0.7, top_p=0.9)
            out_of_scope = False
        except Exception:
            answer = OUT_OF_SCOPE_MSG
            out_of_scope = True
        return jsonify({
            "answer": answer,
            "mode": mode,
            "question_raw": raw,
            "question_normalized": question,
            "out_of_scope": out_of_scope,
        })

    results = search_vector(_collection, _model, _n_faq, question, top_k=TOP_K)
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    best_score = float(1 - distances[0])

    if mode == 1:
        # chỉ retrieval
        if best_score >= THRESHOLD:
            answer = metadatas[0].get("answer", "")
            out_of_scope = False
        else:
            answer = CANNOT_ANSWER_MSG
            out_of_scope = True

    elif mode == 2:
        try:
            answer = llm_generate_answer(question, metadatas, use_local=False)
        except Exception:
            answer = metadatas[0].get("answer", "")
        out_of_scope = False

    else:  # mode == 3
        # hybrid: dưới ngưỡng => Qwen LoRA sinh câu hỏi
        if best_score >= THRESHOLD:
            answer = metadatas[0].get("answer", "")
            out_of_scope = False
        else:
            try:
                answer = local_generate_answer(question, do_sample=True, temperature=0.7, top_p=0.9)
                out_of_scope = False
            except Exception as e:
                answer = f"{OUT_OF_SCOPE_MSG}\n[Hybrid debug] Local LoRA lỗi: {str(e)}"
                out_of_scope = True

    if "out_of_scope" not in locals():
        out_of_scope = False

    return jsonify({
        "answer": answer,
        "mode": mode,
        "question_raw": raw,
        "question_normalized": question,
        "out_of_scope": out_of_scope,
    })


def main():
    try:
        print("Đang load model và index...")
        n = init_app()
        print(f"Đã load {n} FAQ.")
    except FileNotFoundError as e:
        print(str(e))
        return
    # host='0.0.0.0' để truy cập từ bên ngoài (Lightning AI, etc.)
    print("Khởi động Flask: host=0.0.0.0 port=8080")
    app.run(host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    main()
