# [API] API 서버 구동을 위한 라이브러리 임포트
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ▼▼▼▼▼▼▼▼▼▼▼▼ [rag_infer.py 원본 코드 시작] ▼▼▼▼▼▼▼▼▼▼▼▼
# (경로, 로직, 함수 등 아무것도 수정하지 않음)
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

import argparse, json, os
import faiss, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

INDEX_PATH = "rag_index/faiss.index"
META_PATH  = "rag_index/meta.pkl"          # id -> {law_text, clauseField, ...}
SFT_DIR    = "models/llama31-8b-sft-fold1"          # 너가 저장한 SFT 체크포인트
EMB_MODEL  = "nlpai-lab/KURE-v1"                    # 제안서 지정 임베딩 모델

# ==== 1) 로드 ====
def load_index_and_meta():
    import pickle
    print("FAISS 인덱스 로딩 중...") # [API] 서버 시작 시 1회 실행됩니다.
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    # meta: list of dicts with keys: id, law_text, clauseField, file_name, ...
    print("✅ 인덱스/메타데이터 로드 완료.")
    return index, meta

def load_sft_model():
    import os
    from peft import PeftModel
    from transformers import BitsAndBytesConfig
    
    print("SFT(LLM) 모델 로딩 중... (시간 소요)") # [API] 서버 시작 시 1회 실행됩니다.
N = os.environ.get("HF_TOKEN")  # 또는 문자열로 직접 넣어도 됨
    BA
    HF_TOKESE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ADAPTER_DIR = "models/llama31-8b-sft-fold1"  # 너의 LoRA 체크포인트 경로

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
    #   model = model.merge_and_unload()  # 4bit면 에러/의미없음
    # except Exception:
    #   pass
    print("✅ SFT(LLM) 모델 로드 완료.")
    return tok, model

def load_embedder():
    print("임베딩 모델(KURE) 로딩 중...") # [API] 서버 시작 시 1회 실행됩니다.
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
    # D = Similarity Scores (유사도 점수 - KURE는 코사인 유사도)
    # I = Indices (ID)
    D, I = index.search(query_vec, topk)
    return I[0], D[0]

# [수정] 변수명을 distances -> similarities로 변경
def build_report(clause_text, sft_answer, meta, hits=None, similarities=None):
    answer_str = sft_answer.strip()
    is_unfair = "불공정여부: 불리" in answer_str

    report = {
        "input_clause": clause_text,
        "llm_output": answer_str,
    }

    laws = []
    if is_unfair and hits is not None and len(hits) > 0:
        
        # [수정] 변수명을 similarities로 변경
        if similarities is not None and len(similarities) == len(hits):
            hit_data = zip(hits, similarities)
        else:
            hit_data = zip(hits, [None] * len(hits))

        # [수정] 변수명을 sim (similarity)으로 변경
        for idx, sim in hit_data:
            rec = meta[int(idx)]
            law_entry = {
                "clauseField": rec.get("clauseField"),
                "law_text": rec.get("law_text"),
                # [수정] 키 이름을 "similarity"로 변경
                "similarity": float(sim) if sim is not None else None
            }
            laws.append(law_entry)
            
    report["retrieved_laws"] = laws
    return report



# [API] FastAPI 앱 생성
app = FastAPI(title="Law RAG API")

# [API] 모델을 저장할 전역 변수
index = None
meta = None
tok = None
mdl = None
embedder = None

@app.on_event("startup")
def startup_event():
    """ [API] 서버 시작 시 모델을 전역 변수에 미리 로드 (rag_infer.py의 main() 시작 부분) """
    global index, meta, tok, mdl, embedder
    
    index, meta = load_index_and_meta()
    tok, mdl = load_sft_model()
    embedder = load_embedder() # VRAM 8GB에서 OOM 발생 시 이 부분 확인 필요
    
    print("\n✅✅✅ 모든 모델 로드 완료. API 서버가 준비되었습니다. ✅✅✅")


# [API] 입력 형식 정의
class AnalyzeRequest(BaseModel):
    clause_text: str

@app.post("/analyze")
def analyze_clause(request: AnalyzeRequest):
    """ [API] 약관 텍스트를 입력받아 RAG 추론 수행 (rag_infer.py의 while 루프 로직) """
    global index, meta, tok, mdl, embedder # 미리 로드된 모델 사용
    
    clause = request.clause_text
    if not clause:
        return {"error": "clause_text가 비어있습니다."}

    # 1) LLM 추론
    answer = run_sft(tok, mdl, clause)
    reason = parse_reason(answer)

    # 2) 유리한 경우 검색 스킵
    if "불공정여부: 유리" in answer:
        # [수정] similarities=None 추가
        report = build_report(clause, answer, meta, hits=None, similarities=None)
    else:
        # 불리한 경우만 근거 + 원문으로 검색
        fused_query = f"{clause}\n\n판단근거: {reason}" if reason else clause
        qv = embed(embedder, [fused_query])
        
        # [수정] 변수명을 'similarities'로 받음
        ids, similarities = search(index, qv, topk=5)
        
        # [수정] 'similarities' 변수 전달
        report = build_report(clause, answer, meta, hits=ids, similarities=similarities)

    return report


if __name__ == "__main__":
    # [API] Uvicorn 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)