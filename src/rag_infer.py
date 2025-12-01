# rag_infer.py
import os
import faiss, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../nlp_project/src
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))# .../nlp_project

INDEX_PATH = os.path.join(PROJECT_DIR, "rag_index_retriever", "faiss.index")
META_PATH  = os.path.join(PROJECT_DIR, "rag_index_retriever", "meta.pkl")

EMB_MODEL  = os.path.join(PROJECT_DIR, "models", "kure-law-retriever", "checkpoint-94")



# 제안서 지정 임베딩 모델

# ==== 1) 로드 ====
def load_index_and_meta():
    import pickle
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    # meta: list of dicts with keys: id, law_text, clauseField, file_name, ...
    return index, meta
from peft import PeftModel
from transformers import BitsAndBytesConfig

def load_sft_model():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ADAPTER_DIR = SFT_DIR   # 위에서 만든 절대경로 그대로 사용

    print(f"SFT(LLM) 모델 로딩 중... ({ADAPTER_DIR})")

    # 1) 토크나이저
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 2) 4bit base
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        attn_implementation="sdpa",
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    # 3) 어댑터 경로 체크 (디버깅용)
    config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json 이 여기 없음: {config_path}")

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    print("✅ SFT(LLM) 모델 로드 완료.")
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
