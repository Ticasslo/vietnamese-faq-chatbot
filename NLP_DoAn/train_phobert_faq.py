import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi import ViTokenizer
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from typing import Any

#Cấu hình
# Bộ paraphrase gộp (CSV/JSONL cùng nội dung: sentence1, sentence2, faq_id)
DATA_CSV = "Data/After_Processing_Paraphrase/FAQ_HCMUTE_paraphrases_10502.csv"
DATA_JSONL = "Data/After_Processing_Paraphrase/FAQ_HCMUTE_paraphrases_10502.jsonl"
DATA_FAQ = "FAQ_HCMUTE_preprocessed.csv"  
MODEL_NAME = "vinai/phobert-base-v2"
OUTPUT_DIR = "./phobert_faq_retrieval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cuda" else 16
EPOCHS = 2
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1  
RANDOM_SEED = 42
EVAL_K = [1, 3, 5]
LOGGING_STEPS = 25

#Thư mục xuất file
EVAL_DIR = Path("./eval_phobert")  # Thư mục chính đánh giá
EVAL_CHARTS_DIR = EVAL_DIR / "charts"  # Biểu đồ PNG
EVAL_RESULTS_DIR = EVAL_DIR / "results"  # JSON tổng hợp
EVAL_PER_SENTENCE_DIR = EVAL_DIR / "per_sentence"  # Chi tiết từng câu (test split)
DATA_DIR = Path("./data")  # Train/val/test CSV

EVAL_RESULTS_PATH = str(EVAL_RESULTS_DIR / "eval_phobert_results.json")
EVAL_CHART_PATH = str(EVAL_CHARTS_DIR / "eval_phobert_results.png")
EVAL_LOSS_CHART_PATH = str(EVAL_CHARTS_DIR / "eval_phobert_loss.png")
EVAL_ANALYSIS_PATH = str(EVAL_DIR / "eval_phobert_analysis.md")
EVAL_PER_SENTENCE_BASELINE = str(EVAL_PER_SENTENCE_DIR / "phobert_baseline.json")
EVAL_PER_SENTENCE_FINETUNED = str(EVAL_PER_SENTENCE_DIR / "phobert_finetuned.json")
EVAL_PER_SENTENCE_TFIDF = str(EVAL_PER_SENTENCE_DIR / "tfidf.json")
EVAL_RANK_MATRIX_PATH = str(EVAL_CHARTS_DIR / "eval_rank_matrix.png")
OUTPUT_TRAIN = str(DATA_DIR / "Data_paraphrases_train.csv")
OUTPUT_VAL = str(DATA_DIR / "Data_paraphrases_val.csv")
OUTPUT_TEST = str(DATA_DIR / "Data_paraphrases_test.csv")


class LossHistoryCallback(TrainerCallback):

    def __init__(self):
        self.train_losses: list[float] = []
        self.steps: list[int] = []
        self.eval_losses: list[float] = []
        self.eval_steps: list[int] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append(float(logs["loss"]))
            self.steps.append(state.global_step)
        if "eval_loss" in logs:
            self.eval_losses.append(float(logs["eval_loss"]))
            self.eval_steps.append(state.global_step)


def _plot_loss_history(callback: LossHistoryCallback):
    if not callback.train_losses and not callback.eval_losses:
        return
    # Lưu loss history ra JSON
    loss_data = {
        "steps": callback.steps,
        "train_loss": callback.train_losses,
        "eval_steps": callback.eval_steps,
        "eval_loss": callback.eval_losses,
    }
    loss_json_path = str(EVAL_RESULTS_DIR / "eval_loss_history.json")
    with open(loss_json_path, "w", encoding="utf-8") as f:
        json.dump(loss_data, f, indent=2)
    print(f"  Đã lưu loss history → {loss_json_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Lưu ý] pip install matplotlib để vẽ biểu đồ loss")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if callback.train_losses:
        ax.plot(callback.steps, callback.train_losses, label="Train loss", color="#0d6efd", linewidth=1.5)
    if callback.eval_losses:
        ax.plot(callback.eval_steps, callback.eval_losses, label="Eval loss", color="#198754", marker="o", markersize=4)
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Training Loss - PhoBERT FAQ Retrieval", fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=1.5)
    plt.savefig(EVAL_LOSS_CHART_PATH, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  Đã lưu biểu đồ loss → {EVAL_LOSS_CHART_PATH}")


def word_segment(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return ViTokenizer.tokenize(text)


def load_and_split_data():
    if Path(DATA_CSV).exists():
        df = pd.read_csv(DATA_CSV, encoding="utf-8")
        df = df.dropna(subset=["sentence1", "sentence2"])
        if "faq_id" in df.columns:
            df["group"] = df["faq_id"].astype(int)
        elif "stt" in df.columns:
            df["group"] = df["stt"].astype(int)
        else:
            df["group"] = df["sentence2"].astype(str)
    elif Path(DATA_JSONL).exists():
        rows = []
        with open(DATA_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "faq_id" in obj and obj["faq_id"] is not None:
                    grp = int(obj["faq_id"])
                elif "stt" in obj and obj["stt"] is not None:
                    grp = int(obj["stt"])
                else:
                    grp = obj["sentence2"]
                rec = {
                    "sentence1": obj["sentence1"],
                    "sentence2": obj["sentence2"],
                    "group": grp,
                }
                if isinstance(grp, int):
                    rec["faq_id"] = grp
                rows.append(rec)
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError(f"Không tìm thấy {DATA_CSV} hoặc {DATA_JSONL}")

    unique_groups = df["group"].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_groups)

    n = len(unique_groups)
    if n < 3:
        raise ValueError("Số nhóm FAQ quá ít để chia train/val/test. Cần tối thiểu 3 nhóm.")

    n_train = max(1, int(round(n * TRAIN_RATIO)))
    n_val = max(1, int(round(n * VAL_RATIO)))
    n_test = max(1, n - n_train - n_val)

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while n_train + n_val + n_test < n:
        n_train += 1

    grp_train = set(unique_groups[:n_train])
    grp_val = set(unique_groups[n_train:n_train + n_val])
    grp_test = set(unique_groups[n_train + n_val:n_train + n_val + n_test])

    def to_examples(grp_set):
        subset = df[df["group"].isin(grp_set)]
        return [
            InputExample(texts=[word_segment(str(r["sentence1"]).strip()), word_segment(str(r["sentence2"]).strip())])
            for _, r in subset.iterrows()
        ]

    train_examples = to_examples(grp_train)
    val_examples = to_examples(grp_val)
    test_examples = to_examples(grp_test)
    train_df = df[df["group"].isin(grp_train)].copy()
    val_df = df[df["group"].isin(grp_val)].copy()
    test_df = df[df["group"].isin(grp_test)].copy()

    return train_examples, val_examples, test_examples, train_df, val_df, test_df


def build_val_binary_examples(val_df: pd.DataFrame) -> list[InputExample]:
    if len(val_df) < 2:
        return []

    df = val_df.dropna(subset=["sentence1", "sentence2", "group"]).reset_index(drop=True)
    if len(df) < 2:
        return []

    groups = df["group"].to_numpy()
    s1_list = df["sentence1"].astype(str).str.strip().tolist()
    s2_list = df["sentence2"].astype(str).str.strip().tolist()

    rng = np.random.default_rng(RANDOM_SEED)
    binary_examples: list[InputExample] = []

    for i, (s1, s2, grp) in enumerate(zip(s1_list, s2_list, groups)):
        binary_examples.append(
            InputExample(texts=[word_segment(s1), word_segment(s2)], label=1.0)
        )

        diff_idx = np.where(groups != grp)[0]
        if len(diff_idx) == 0:
            continue
        j = int(rng.choice(diff_idx))
        s2_neg = s2_list[j]
        binary_examples.append(
            InputExample(texts=[word_segment(s1), word_segment(s2_neg)], label=0.0)
        )

    return binary_examples


def _assert_no_data_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    split_groups = {
        "train": set(train_df["group"].tolist()),
        "val": set(val_df["group"].tolist()),
        "test": set(test_df["group"].tolist()),
    }

    overlaps = [
        ("train", "val", split_groups["train"] & split_groups["val"]),
        ("train", "test", split_groups["train"] & split_groups["test"]),
        ("val", "test", split_groups["val"] & split_groups["test"]),
    ]
    bad = [(a, b, ov) for a, b, ov in overlaps if ov]
    if bad:
        msg = "; ".join([f"{a}-{b}: {len(ov)} groups" for a, b, ov in bad])
        raise ValueError(f"Data leakage theo group giữa các split: {msg}")

    def _pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
        return set(
            zip(
                df["sentence1"].astype(str).str.strip().tolist(),
                df["sentence2"].astype(str).str.strip().tolist(),
            )
        )

    train_pairs = _pair_set(train_df)
    val_pairs = _pair_set(val_df)
    test_pairs = _pair_set(test_df)
    dup_tv = len(train_pairs & val_pairs)
    dup_tt = len(train_pairs & test_pairs)
    dup_vt = len(val_pairs & test_pairs)
    if dup_tv or dup_tt or dup_vt:
        print(
            "  [Cảnh báo] Có exact duplicate pair giữa split: "
            f"train-val={dup_tv}, train-test={dup_tt}, val-test={dup_vt}"
        )
    else:
        print("  [Leak-check] Không có overlap group và không trùng exact pair giữa các split.")


def _build_st_dataset(df: pd.DataFrame) -> Any:
    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover - fallback runtime path
        raise RuntimeError("Thiếu package `datasets` để dùng SentenceTransformerTrainer.") from exc

    if len(df) == 0:
        return Dataset.from_dict({"sentence1": [], "sentence2": []})

    s1 = [word_segment(str(x).strip()) for x in df["sentence1"].tolist()]
    s2 = [word_segment(str(x).strip()) for x in df["sentence2"].tolist()]
    return Dataset.from_dict({"sentence1": s1, "sentence2": s2})


def _get_faq_path():
    for p in [DATA_FAQ]:
        if Path(p).exists():
            return p
    return None


def eval_retrieval_precision(model, test_df, faq_path: str, k_list: list[int], batch_size: int = 32):
    df_faq = pd.read_csv(faq_path, encoding="utf-8")
    q_col = "question" if "question" in df_faq.columns else "Question"
    a_col = "answer" if "answer" in df_faq.columns else "Answer"
    if q_col not in df_faq.columns or a_col not in df_faq.columns:
        raise ValueError(f"{faq_path} cần cột 'question' và 'answer'")
    faq_questions = set(df_faq[q_col].str.strip())

    q_seg = [word_segment(q) for q in df_faq[q_col].str.strip().tolist()]
    faq_emb = model.encode(q_seg, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    q_seg_test = [word_segment(q) for q in test_df["sentence1"].str.strip().tolist()]
    query_emb = model.encode(q_seg_test, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    sim = np.dot(query_emb, faq_emb.T)

    n_valid = int((test_df["sentence2"].str.strip().isin(faq_questions)).sum())
    if n_valid < len(test_df) * 0.9:
        print(f"  [Cảnh báo] Chỉ {n_valid}/{len(test_df)} test khớp FAQ.")

    n_faq = len(df_faq)
    results = {}
    for k in k_list:
        k_use = min(k, n_faq)
        correct = 0
        for i in range(len(test_df)):
            gt_question = test_df.iloc[i]["sentence2"].strip()
            if gt_question not in faq_questions:
                continue
            top_k_idx = np.argsort(sim[i])[::-1][:k_use]
            retrieved_questions = [df_faq.iloc[j][q_col].strip() for j in top_k_idx]
            if gt_question in retrieved_questions:
                correct += 1
        results[f"P@{k}"] = correct / n_valid if n_valid > 0 else 0.0
    return results


def eval_retrieval_precision_tfidf(test_df, faq_path: str, k_list: list[int]):
    df_faq = pd.read_csv(faq_path, encoding="utf-8")
    q_col = "question" if "question" in df_faq.columns else "Question"
    a_col = "answer" if "answer" in df_faq.columns else "Answer"
    if q_col not in df_faq.columns or a_col not in df_faq.columns:
        raise ValueError(f"{faq_path} cần cột 'question' và 'answer'")
    faq_questions = set(df_faq[q_col].str.strip())

    def tokenize_vi(text):
        return word_segment(str(text).strip()).split()

    faq_texts = [str(q).strip() for q in df_faq[q_col]]
    query_texts = test_df["sentence1"].str.strip().tolist()

    vectorizer = TfidfVectorizer(tokenizer=tokenize_vi, lowercase=False, token_pattern=None)
    faq_tfidf = vectorizer.fit_transform(faq_texts)
    query_tfidf = vectorizer.transform(query_texts)
    sim = cosine_similarity(query_tfidf, faq_tfidf)

    n_valid = int((test_df["sentence2"].str.strip().isin(faq_questions)).sum())
    n_faq = len(df_faq)
    results = {}
    for k in k_list:
        k_use = min(k, n_faq)
        correct = 0
        for i in range(len(test_df)):
            gt_question = test_df.iloc[i]["sentence2"].strip()
            if gt_question not in faq_questions:
                continue
            top_k_idx = np.argsort(sim[i])[::-1][:k_use]
            retrieved_questions = [df_faq.iloc[j][q_col].strip() for j in top_k_idx]
            if gt_question in retrieved_questions:
                correct += 1
        results[f"P@{k}"] = correct / n_valid if n_valid > 0 else 0.0
    return results


def eval_cosine_similarity(model, test_df, batch_size: int = 32) -> float:
    s1_list = [word_segment(q) for q in test_df["sentence1"].str.strip().tolist()]
    s2_list = [word_segment(q) for q in test_df["sentence2"].str.strip().tolist()]
    emb1 = model.encode(s1_list, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    emb2 = model.encode(s2_list, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    return float(np.mean(np.diag(np.dot(emb1, emb2.T))))


def eval_similarity_breakdown(model, test_df, faq_path: str, batch_size: int = 32) -> dict | None:
    df_faq = pd.read_csv(faq_path, encoding="utf-8")
    q_col = "question" if "question" in df_faq.columns else "Question"
    if q_col not in df_faq.columns:
        return None
    q2idx = {q.strip(): i for i, q in enumerate(df_faq[q_col].str.strip())}

    q_seg = [word_segment(q) for q in df_faq[q_col].str.strip().tolist()]
    faq_emb = model.encode(q_seg, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    q_seg_test = [word_segment(q) for q in test_df["sentence1"].str.strip().tolist()]
    query_emb = model.encode(q_seg_test, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    sim = np.dot(query_emb, faq_emb.T)

    n_faq = len(df_faq)
    sim_pos_list, sim_neg_list = [], []
    for i in range(len(test_df)):
        gt_q = test_df.iloc[i]["sentence2"].strip()
        idx = q2idx.get(gt_q)
        if idx is None:
            continue
        sim_pos_list.append(sim[i, idx])
        mask = np.ones(n_faq, dtype=bool)
        mask[idx] = False
        sim_neg_list.append(np.mean(sim[i, mask]))
    if not sim_pos_list:
        return None
    sim_pos = float(np.mean(sim_pos_list))
    sim_neg = float(np.mean(sim_neg_list))
    return {"sim_positive": sim_pos, "sim_negative": sim_neg, "margin": sim_pos - sim_neg}


def export_per_sentence_retrieval(
    model,
    test_df,
    faq_path: str,
    k_list: list[int],
    output_path: str,
    batch_size: int = 32,
) -> None:
    df_faq = pd.read_csv(faq_path, encoding="utf-8")
    q_col = "question" if "question" in df_faq.columns else "Question"
    a_col = "answer" if "answer" in df_faq.columns else "Answer"
    if q_col not in df_faq.columns or a_col not in df_faq.columns:
        raise ValueError(f"{faq_path} cần cột 'question' và 'answer'")
    q2answer = dict(zip(df_faq[q_col].str.strip(), df_faq[a_col]))

    q_seg = [word_segment(q) for q in df_faq[q_col].str.strip().tolist()]
    faq_emb = model.encode(q_seg, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    q_seg_test = [word_segment(q) for q in test_df["sentence1"].str.strip().tolist()]
    query_emb = model.encode(q_seg_test, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    sim = np.dot(query_emb, faq_emb.T)

    n_faq = len(df_faq)
    k_max = max(k_list)
    k_use = min(k_max, n_faq)

    rows = []
    for i in range(len(test_df)):
        query = test_df.iloc[i]["sentence1"].strip()
        gt_question = test_df.iloc[i]["sentence2"].strip()
        gt_answer = q2answer.get(gt_question)

        top_k_idx = np.argsort(sim[i])[::-1][:k_use]
        top_k_questions = [df_faq.iloc[j][q_col].strip() for j in top_k_idx]
        top_k_answers = [df_faq.iloc[j][a_col] for j in top_k_idx]
        top_k_sims = [float(sim[i, j]) for j in top_k_idx]

        prec_at_k = {}
        if gt_answer is not None:  
            for k in k_list:
                k_actual = min(k, len(top_k_questions))
                prec_at_k[k] = 1 if gt_question in top_k_questions[:k_actual] else 0
            correct_rank = next((r + 1 for r, q in enumerate(top_k_questions) if q == gt_question), -1)
        else:
            for k in k_list:
                prec_at_k[k] = -1  
            correct_rank = -1

        top_k = [
            {"rank": r + 1, "question": top_k_questions[r], "answer": top_k_answers[r], "sim": round(top_k_sims[r], 4)}
            for r in range(k_use)
        ]
        prec = {f"P@{k}": prec_at_k.get(k, -1) for k in k_list}
        rows.append({
            "query": query,
            "ground_truth": {"question": gt_question, "answer": gt_answer if gt_answer is not None else ""},
            "correct_rank": correct_rank,
            "top_k": top_k,
            "precision": prec,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  Đã xuất chi tiết từng câu → {output_path}")


def export_per_sentence_retrieval_tfidf(test_df, faq_path: str, k_list: list[int], output_path: str):
    df_faq = pd.read_csv(faq_path, encoding="utf-8")
    q_col = "question" if "question" in df_faq.columns else "Question"
    a_col = "answer" if "answer" in df_faq.columns else "Answer"
    if q_col not in df_faq.columns or a_col not in df_faq.columns:
        raise ValueError(f"{faq_path} cần cột 'question' và 'answer'")
    q2answer = dict(zip(df_faq[q_col].str.strip(), df_faq[a_col]))

    def tokenize_vi(text):
        return word_segment(str(text).strip()).split()

    faq_texts = [str(q).strip() for q in df_faq[q_col]]
    query_texts = test_df["sentence1"].str.strip().tolist()
    vectorizer = TfidfVectorizer(tokenizer=tokenize_vi, lowercase=False, token_pattern=None)
    faq_tfidf = vectorizer.fit_transform(faq_texts)
    query_tfidf = vectorizer.transform(query_texts)
    sim = cosine_similarity(query_tfidf, faq_tfidf)

    n_faq = len(df_faq)
    k_max = max(k_list)
    k_use = min(k_max, n_faq)

    rows = []
    for i in range(len(test_df)):
        query = test_df.iloc[i]["sentence1"].strip()
        gt_question = test_df.iloc[i]["sentence2"].strip()
        gt_answer = q2answer.get(gt_question)

        top_k_idx = np.argsort(sim[i])[::-1][:k_use]
        top_k_questions = [df_faq.iloc[j][q_col].strip() for j in top_k_idx]
        top_k_answers = [df_faq.iloc[j][a_col] for j in top_k_idx]
        top_k_sims = [float(sim[i, j]) for j in top_k_idx]

        prec_at_k = {}
        if gt_answer is not None:
            for k in k_list:
                k_actual = min(k, len(top_k_questions))
                prec_at_k[k] = 1 if gt_question in top_k_questions[:k_actual] else 0
            correct_rank = next((r + 1 for r, q in enumerate(top_k_questions) if q == gt_question), -1)
        else:
            for k in k_list:
                prec_at_k[k] = -1
            correct_rank = -1

        top_k = [
            {"rank": r + 1, "question": top_k_questions[r], "answer": top_k_answers[r], "sim": round(top_k_sims[r], 4)}
            for r in range(k_use)
        ]
        prec = {f"P@{k}": prec_at_k.get(k, -1) for k in k_list}
        rows.append({
            "query": query,
            "ground_truth": {"question": gt_question, "answer": gt_answer if gt_answer is not None else ""},
            "correct_rank": correct_rank,
            "top_k": top_k,
            "precision": prec,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  Đã xuất chi tiết TF-IDF → {output_path}")


def _save_eval_results(baseline: dict, finetuned: dict, baseline_cos: float, finetuned_cos: float,
                       baseline_breakdown: dict | None = None, finetuned_breakdown: dict | None = None,
                       tfidf: dict | None = None):
    data = {
        "tfidf": tfidf or {},
        "baseline_phobert": {**baseline, "cosine_similarity": baseline_cos, **(baseline_breakdown or {})},
        "finetuned": {**finetuned, "cosine_similarity": finetuned_cos, **(finetuned_breakdown or {})},
        "delta_phobert": {k: finetuned.get(k, 0) - baseline.get(k, 0) for k in set(baseline) | set(finetuned) if k != "cosine_similarity"},
        "delta_cosine": finetuned_cos - baseline_cos,
    }
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả → {EVAL_RESULTS_PATH}")


def _save_analysis_report(baseline: dict, finetuned: dict, baseline_cos: float, finetuned_cos: float,
                          baseline_breakdown: dict | None, finetuned_breakdown: dict | None,
                          tfidf: dict | None = None):
    """Xuất báo cáo phân tích (sim positive, sim negative, margin) ra markdown."""
    lines = [
        "# Phân tích kết quả huấn luyện PhoBERT cho FAQ Retrieval",
        "",
        "## 1. Tổng quan",
        "",
        "So sánh TF-IDF, PhoBERT gốc, Fine-tuned. Đánh giá: Precision@k.",
        "",
        "## 2. Kết quả đánh giá",
        "",
        "| Metric | TF-IDF | PhoBERT gốc | Fine-tuned | Δ (PhoBERT) |",
        "|--------|--------|-------------|------------|-------------|",
    ]
    t_vals = [tfidf.get(f"P@{k}", 0) if tfidf else 0 for k in EVAL_K]
    lines.append(
        f"| Cosine similarity | - | {baseline_cos:.4f} | {finetuned_cos:.4f} | {finetuned_cos - baseline_cos:+.4f} |"
    )
    for idx, k in enumerate(EVAL_K):
        key = f"P@{k}"
        b, f = baseline.get(key, 0), finetuned.get(key, 0)
        t = t_vals[idx] if tfidf else 0
        t_str = f"{t:.4f}" if tfidf else "-"
        lines.append(f"| {key} | {t_str} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")
    lines.append("")

    if baseline_breakdown and finetuned_breakdown:
        lines.extend([
            "## 3. Phân tích similarity (từ code)",
            "",
            "| Loại | PhoBERT gốc | Fine-tuned |",
            "|------|-------------|------------|",
            f"| sim(query, FAQ đúng) | {baseline_breakdown['sim_positive']:.4f} | {finetuned_breakdown['sim_positive']:.4f} |",
            f"| sim(query, FAQ sai) | {baseline_breakdown['sim_negative']:.4f} | {finetuned_breakdown['sim_negative']:.4f} |",
            f"| Margin | {baseline_breakdown['margin']:.4f} | {finetuned_breakdown['margin']:.4f} |",
            "",
            "Margin tăng → ranking tốt hơn → Precision@k tăng.",
            "",
        ])

    with open(EVAL_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Đã lưu báo cáo phân tích → {EVAL_ANALYSIS_PATH}")


def _plot_eval_comparison(baseline: dict, finetuned: dict, baseline_cos: float, finetuned_cos: float,
                          tfidf: dict | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Lưu ý] pip install matplotlib để vẽ biểu đồ")
        return

    metrics = ["Cosine similarity"] + [f"P@{k}" for k in EVAL_K]
    n_metrics = len(metrics)
    tfidf_vals = [0] + [tfidf.get(f"P@{k}", 0) for k in EVAL_K] if tfidf else None
    base_vals = [baseline_cos] + [baseline.get(f"P@{k}", 0) for k in EVAL_K]
    fine_vals = [finetuned_cos] + [finetuned.get(f"P@{k}", 0) for k in EVAL_K]

    x = np.arange(n_metrics)
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    bars_list = []
    methods = [
        (tfidf_vals, "TF-IDF", "#28a745"),
        (base_vals, "PhoBERT gốc", "#6c757d"),
        (fine_vals, "Fine-tuned", "#0d6efd"),
    ]
    idx = 0
    n_methods = sum(1 for v, _, _ in methods if v is not None)
    for vals, label, color in methods:
        if vals is not None:
            off = (idx - (n_methods - 1) / 2) * width
            b = ax.bar(x + off, vals, width, label=label, color=color)
            bars_list.extend(b)
            idx += 1

    ax.set_ylabel("Giá trị", fontsize=10)
    ax.set_title("So sánh TF-IDF / PhoBERT gốc / Fine-tuned", fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    for bar in bars_list:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
    plt.tight_layout(pad=1.5)
    plt.savefig(EVAL_CHART_PATH, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  Đã lưu biểu đồ → {EVAL_CHART_PATH}")


def _plot_rank_matrix():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    files = [
        (EVAL_PER_SENTENCE_TFIDF, "TF-IDF"),
        (EVAL_PER_SENTENCE_BASELINE, "PhoBERT gốc"),
        (EVAL_PER_SENTENCE_FINETUNED, "Fine-tuned"),
    ]
    rank_labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5", "Không tìm thấy"]
    data = []
    method_names = []

    for path, name in files:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        counts = [0] * 6
        for r in rows:
            rank = r.get("correct_rank", -1)
            if 1 <= rank <= 5:
                counts[rank - 1] += 1
            else:
                counts[5] += 1
        data.append(counts)
        method_names.append(name)

    if not data:
        return

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(rank_labels))
    width = 0.22
    colors = ["#28a745", "#6c757d", "#0d6efd"]
    for i, (row, name) in enumerate(zip(data, method_names)):
        offset = (i - len(data) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row, width, label=name, color=colors[i % len(colors)])
        for bar in bars:
            if bar.get_height() > 0:
                ax.annotate(f"{int(bar.get_height())}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)

    ax.set_ylabel("Số lượng query", fontsize=10)
    ax.set_title("Ma trận phân bố rank – Vị trí FAQ đúng trong top retrieval", fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(rank_labels, fontsize=9, rotation=20, ha="right")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, max(data.max() * 1.2, 1))
    plt.tight_layout(pad=1.5)
    plt.savefig(EVAL_RANK_MATRIX_PATH, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  Đã lưu ma trận rank → {EVAL_RANK_MATRIX_PATH}")


def _run_eval(model, test_df, faq_path):
    cos = eval_cosine_similarity(model, test_df)
    print(f"  Cosine similarity: {cos:.4f}")
    prec = eval_retrieval_precision(model, test_df, faq_path, EVAL_K) if faq_path else {}
    for k, v in prec.items():
        print(f"  {k}: {v:.4f}")
    breakdown = eval_similarity_breakdown(model, test_df, faq_path) if faq_path else None
    if breakdown:
        print(f"  sim(positive): {breakdown['sim_positive']:.4f} | sim(negative): {breakdown['sim_negative']:.4f} | margin: {breakdown['margin']:.4f}")
    return cos, prec, breakdown


def main():
    print("=" * 50)
    print("  Fine-tune PhoBERT cho FAQ Retrieval")
    print("=" * 50)

    # Tạo thư mục xuất file
    for d in [EVAL_DIR, EVAL_CHARTS_DIR, EVAL_RESULTS_DIR, EVAL_PER_SENTENCE_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n  Thư mục xuất: {EVAL_DIR}/, {DATA_DIR}/")

    print("\nĐang load dữ liệu...")
    train_examples, val_examples, test_examples, train_df, val_df, test_df = load_and_split_data()
    _assert_no_data_leakage(train_df, val_df, test_df)

    cols = ["sentence1", "sentence2"] + (["faq_id"] if "faq_id" in train_df.columns else (["stt"] if "stt" in train_df.columns else ["group"]))
    train_df[cols].to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8")
    test_df[cols].to_csv(OUTPUT_TEST, index=False, encoding="utf-8")
    if len(val_df) > 0:
        val_df[cols].to_csv(OUTPUT_VAL, index=False, encoding="utf-8")
    print(
        f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} cặp"
    )

    if len(train_examples) == 0:
        raise ValueError("Không có dữ liệu train.")

    evaluator = None
    eval_steps = 0
    if val_examples:
        val_binary_examples = build_val_binary_examples(val_df)
        if val_binary_examples:
            evaluator = BinaryClassificationEvaluator.from_input_examples(
                val_binary_examples,
                name="val_paraphrase_bin",
                show_progress_bar=False,
            )
        else:
            print("  [Lưu ý] Val quá nhỏ để tạo binary evaluator, bỏ qua eval trong train.")
            evaluator = None
        # Đánh giá thường xuyên hơn để theo dõi overfit.
        eval_steps = max(20, len(train_examples) // (BATCH_SIZE * 10)) if evaluator else 0

    faq_path = _get_faq_path()
    if not faq_path:
        print("\n[Lưu ý] Không tìm thấy data.csv → bỏ qua Precision@k")

    tfidf_results = None
    if len(test_df) > 0 and faq_path:
        print("\n--- TF-IDF (baseline so sánh) ---")
        tfidf_results = eval_retrieval_precision_tfidf(test_df, faq_path, EVAL_K)
        for k, v in tfidf_results.items():
            print(f"  {k}: {v:.4f}")
        export_per_sentence_retrieval_tfidf(test_df, faq_path, EVAL_K, EVAL_PER_SENTENCE_TFIDF)

    print(f"\nĐang load PhoBERT trên {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    baseline_cos, baseline, baseline_breakdown = None, None, None
    if len(test_df) > 0:
        print("\n--- Baseline (chưa fine-tune) ---")
        baseline_cos, baseline, baseline_breakdown = _run_eval(model, test_df, faq_path)
        if faq_path:
            export_per_sentence_retrieval(model, test_df, faq_path, EVAL_K, EVAL_PER_SENTENCE_BASELINE, BATCH_SIZE)

    print(f"\nTrain {EPOCHS} epochs, batch_size={BATCH_SIZE}...")
    loss_callback = LossHistoryCallback()
    print(f"  Eval mỗi ~{eval_steps} steps (dựa trên val set).")
    print(f"  Log train mỗi {LOGGING_STEPS} steps.")

    used_legacy_fit = False
    try:
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

        train_dataset = _build_st_dataset(train_df)
        eval_dataset = _build_st_dataset(val_df) if len(val_df) > 0 else None

        args = SentenceTransformerTrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=1e-5,
            logging_steps=LOGGING_STEPS,
            logging_strategy="steps",
            eval_strategy="steps" if evaluator and eval_dataset is not None else "no",
            eval_steps=eval_steps if evaluator and eval_dataset is not None else None,
            save_strategy="no",
            report_to=[],
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=evaluator,
            callbacks=[loss_callback],
        )
        trainer.train()
        trainer.save_model(OUTPUT_DIR)
    except Exception as exc:
        # Fallback để đảm bảo script vẫn chạy nếu môi trường chưa có trainer mới/deps liên quan.
        print(f"  [Lưu ý] Không dùng được SentenceTransformerTrainer ({exc}). Fallback về model.fit().")
        used_legacy_fit = True
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        best_eval_score = float("-inf")

        def _eval_callback(score: float, epoch: int, steps: int):
            nonlocal best_eval_score
            loss_callback.eval_losses.append(score)
            loss_callback.eval_steps.append(steps)
            is_finite = np.isfinite(score)
            is_best = is_finite and (score > best_eval_score)
            if is_best:
                best_eval_score = score
            print(
                f"  [Val] step={steps:>5} | epoch={epoch:.3f} | score={score:.4f}"
                f"{' | best' if is_best else ''}"
            )

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=EPOCHS,
            evaluation_steps=eval_steps if evaluator else 0,
            output_path=OUTPUT_DIR,
            show_progress_bar=True,
            callback=_eval_callback if evaluator else None,
        )

    if not used_legacy_fit and evaluator:
        print("  [Val] Dùng evaluator trong trainer; xem thêm log eval_* theo steps ở terminal.")
    _plot_loss_history(loss_callback)

    if len(test_df) > 0:
        print("\n--- Sau fine-tune ---")
        finetuned_cos, finetuned, finetuned_breakdown = _run_eval(model, test_df, faq_path)
        if faq_path:
            export_per_sentence_retrieval(model, test_df, faq_path, EVAL_K, EVAL_PER_SENTENCE_FINETUNED, BATCH_SIZE)

        if baseline_cos is not None or baseline:
            print("\n" + "=" * 88)
            print("  So sánh TF-IDF | PhoBERT gốc | Fine-tuned")
            print("=" * 88)
            print(
                f"  {'Metric':<18} {'TF-IDF':<10} {'PhoBERT gốc':<12} {'Fine-tune':<10} {'Δ':<8}"
            )
            print("-" * 88)
            if baseline_cos is not None:
                print(
                    f"  {'Cosine similarity':<18} {'-':<10} {baseline_cos:<12.4f} {finetuned_cos:<10.4f} {finetuned_cos - baseline_cos:+.4f}"
                )
            if baseline:
                for k in EVAL_K:
                    key = f"P@{k}"
                    t = tfidf_results.get(key, 0) if tfidf_results else 0
                    b, f = baseline.get(key, 0), finetuned.get(key, 0)
                    t_str = f"{t:.4f}" if tfidf_results else "-"
                    print(f"  {key:<18} {t_str:<10} {b:<12.4f} {f:<10.4f} {f - b:+.4f}")
            print("=" * 88)

        _save_eval_results(baseline or {}, finetuned, baseline_cos or 0.0, finetuned_cos,
                           baseline_breakdown, finetuned_breakdown, tfidf=tfidf_results)
        _save_analysis_report(baseline or {}, finetuned, baseline_cos or 0.0, finetuned_cos,
                             baseline_breakdown, finetuned_breakdown, tfidf=tfidf_results)
        _plot_eval_comparison(baseline or {}, finetuned, baseline_cos or 0.0, finetuned_cos,
                             tfidf=tfidf_results)
        _plot_rank_matrix()

    print(f"\nĐã lưu model → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
