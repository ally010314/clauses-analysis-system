import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import argparse, json, os
import faiss, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

INDEX_PATH = "rag_index/faiss.index"
META_PATH  = "rag_index/meta.pkl"          
SFT_DIR    = "models/llama31-8b-sft-fold1"        
EMB_MODEL  = "nlpai-lab/KURE-v1"                    

def load_index_and_meta():
    import pickle
    print("FAISS 인덱스 로딩 중...") 
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    print("✅ 인덱스/메타데이터 로드 완료.")
    return index, meta

def load_sft_model():
    import os
    from peft import PeftModel
    from transformers import BitsAndBytesConfig
    
    print("SFT(LLM) 모델 로딩 중... (시간 소요)") 

    HF_TOKEN = os.environ.get("HF_TOKEN")  
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ADAPTER_DIR = "models/llama31-8b-sft-fold1" 

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        attn_implementation="sdpa",
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    print("✅ SFT(LLM) 모델 로드 완료.")
    return tok, model

def load_embedder():
    print("임베딩 모델(KURE) 로딩 중...") 
    return SentenceTransformer(EMB_MODEL) 
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
    ans = out_txt.split("assistant\n")[-1].strip()
    return ans

def parse_reason(answer_line: str) -> str:
    parts = [p.strip() for p in answer_line.split("/") if p.strip()]
    reason = ""
    for p in parts:
        if p.startswith("근거:"):
            reason = p.replace("근거:", "").strip()
            break
    return reason

def embed(embedder, texts):
    embs = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

def search(index, query_vec, topk=5):
    D, I = index.search(query_vec, topk)
    return I[0], D[0]

def build_report(clause_text, sft_answer, meta, hits=None, similarities=None):
    answer_str = sft_answer.strip()
    is_unfair = "불공정여부: 불리" in answer_str

    report = {
        "input_clause": clause_text,
        "llm_output": answer_str,
    }

    laws = []
    if is_unfair and hits is not None and len(hits) > 0:
        
        if similarities is not None and len(similarities) == len(hits):
            hit_data = zip(hits, similarities)
        else:
            hit_data = zip(hits, [None] * len(hits))

        for idx, sim in hit_data:
            rec = meta[int(idx)]
            law_entry = {
                "clauseField": rec.get("clauseField"),
                "law_text": rec.get("law_text"),
                "similarity": float(sim) if sim is not None else None
            }
            laws.append(law_entry)
            
    report["retrieved_laws"] = laws
    return report



app = FastAPI(title="Law RAG API")

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
    embedder = load_embedder() 
    
    print("\n✅✅✅ 모든 모델 로드 완료. API 서버가 준비되었습니다. ✅✅✅")


class AnalyzeRequest(BaseModel):
    clause_text: str

@app.post("/analyze")
def analyze_clause(request: AnalyzeRequest):
    """ [API] 약관 텍스트를 입력받아 RAG 추론 수행 (rag_infer.py의 while 루프 로직) """
    global index, meta, tok, mdl, embedder 
    
    clause = request.clause_text
    if not clause:
        return {"error": "clause_text가 비어있습니다."}

    answer = run_sft(tok, mdl, clause)
    reason = parse_reason(answer)

    if "불공정여부: 유리" in answer:
        report = build_report(clause, answer, meta, hits=None, similarities=None)
    else:
        fused_query = f"{clause}\n\n판단근거: {reason}" if reason else clause
        qv = embed(embedder, [fused_query])
        ids, similarities = search(index, qv, topk=5)
        report = build_report(clause, answer, meta, hits=ids, similarities=similarities)

    return report


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)