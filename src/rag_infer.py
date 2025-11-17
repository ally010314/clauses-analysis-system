# rag_infer.py
import argparse, json, os
import faiss, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ==== 0) 경로들 ====
INDEX_PATH = "nlp_project/rag_index/faiss.index"
META_PATH  = "nlp_project/rag_index/meta.pkl"          # id -> {law_text, clauseField, ...}
SFT_DIR    = "nlp_project/models/llama31-8b-sft-fold2"              # 너가 저장한 SFT 체크포인트
EMB_MODEL  = "nlpai-lab/KURE-v1"                  # 제안서 지정 임베딩 모델

# ==== 1) 로드 ====
def load_index_and_meta():
    import pickle
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    # meta: list of dicts with keys: id, law_text, clauseField, file_name, ...
    return index, meta

def load_sft_model():
    import os
    from peft import PeftModel
    from transformers import BitsAndBytesConfig

    HF_TOKEN = os.environ.get("HF_TOKEN")  # 또는 문자열로 직접 넣어도 됨
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ADAPTER_DIR = "nlp_project/models/llama31-8b-sft-fold2"  # 너의 LoRA 체크포인트 경로

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 1) 토크나이저
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2) 베이스(4bit, SDPA)  ← 학습 설정과 일치시킴
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        attn_implementation="sdpa",
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    # 3) LoRA 어댑터 로드 (학습 당시 경로 체계와 동일하게 맞춰짐)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    # 4) 4bit에서는 merge 불가(권장 안됨). 그냥 adapter 붙인 채로 사용.
    # try:
    #     model = model.merge_and_unload()  # 4bit면 에러/의미없음
    # except Exception:
    #     pass

    return tok, model



def load_embedder():
    return SentenceTransformer(EMB_MODEL)  # KURE-v1

# ==== 2) SFT 한줄 추론 ====
@torch.no_grad()
def run_sft(tok, mdl, clause_text: str) -> str:
    system = (
        "당신은 약관의 공정성을 분석하는 법률 전문가입니다.\n"
        "문맥상 주체 (고객/ 사업자) 를 명확히 구분하세요.\n"
        "반드시 아래 한 줄 포맷만 출력하세요:\n"
        "분야: <정수> / 불공정여부: <유리|불리> / 근거: <간결한 문장 또는 '해당 없음'>"
    )
    user = f"다음 약관 조항의 문맥을 이해하여 분야 분류, 불공정 여부 판단, 판단 근거를 요약하시오.\n\n입력:\n{clause_text}"

    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out_ids = mdl.generate(
        **inputs, 
        max_new_tokens=256, 
        do_sample=False
    )
    out_txt = tok.decode(out_ids[0], skip_special_tokens=True)
    # 마지막 assistant만 추출
    ans = out_txt.split("assistant\n")[-1].strip()
    return ans

def parse_reason(answer_line: str) -> str:
    # "분야: X / 불공정여부: 유리|불리 / 근거: Y" 에서 근거만 추출
    parts = [p.strip() for p in answer_line.split("/") if p.strip()]
    reason = ""
    for p in parts:
        if p.startswith("근거:"):
            reason = p.replace("근거:", "").strip()
            break
    return reason

# ==== 3) 검색 ====
def embed(embedder, texts):
    embs = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

def search(index, query_vec, topk=5):
    D, I = index.search(query_vec, topk)
    return I[0], D[0]

# ==== 4) 최종 리포트 생성 ====
def build_report(clause_text, sft_answer, meta, hits=None):
    answer_str = sft_answer.strip()
    is_unfair = "불공정여부: 불리" in answer_str

    report = {
        "input_clause": clause_text,
        "llm_output": answer_str,
    }

    laws = []
    if is_unfair and hits is not None and len(hits) > 0:
        for idx in hits:
            rec = meta[int(idx)]
            laws.append({
                "clauseField": rec.get("clauseField"),
                "law_text": rec.get("law_text")
            })
    report["retrieved_laws"] = laws
    return report



def main():
    index, meta = load_index_and_meta()
    tok, mdl = load_sft_model()
    embedder = load_embedder()

    print("✅ 모델 및 인덱스 로드 완료.")
    print("엔터만 누르면 종료됩니다.\n")

    while True:
        clause = input("🔍 약관 문장을 입력하세요:\n> ").strip()
        if not clause:
            print("\n👋 종료합니다.")
            break

        # 1) LLM 추론
        answer = run_sft(tok, mdl, clause)
        reason = parse_reason(answer)

        # 2) 유리한 경우 검색 스킵
        if "불공정여부: 유리" in answer:
            report = build_report(clause, answer, meta, hits=None)
        else:
            # 불리한 경우만 근거 + 원문으로 검색
            fused_query = f"{clause}\n\n판단근거: {reason}" if reason else clause
            qv = embed(embedder, [fused_query])
            ids, _ = search(index, qv, topk=5)
            report = build_report(clause, answer, meta, hits=ids)

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
