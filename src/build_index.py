# build_index_json.py
import os
import faiss
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ===== 설정 =====
JSON_PATH = "nlp_project/data/relateLaword_index.json"  # 🔸 JSON 파일 사용
INDEX_DIR = "nlp_project/rag_index"
EMBEDDING_MODEL = os.environ.get("EMB_MODEL", "nlpai-lab/KURE-v1")
BATCH = 128


def get_model(name: str):
    print(f"[Embedding] loading: {name}")
    model = SentenceTransformer(name)
    return model


def e5_encode_query(text: str) -> str:
    # E5/KURE 계열은 prefix 필요
    return "passage: " + text.strip()


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    df = pd.read_json(JSON_PATH)
    assert "law_text" in df.columns, "JSON에 'law_text' 필드가 필요합니다."

    # 중복 제거
    df = df.drop_duplicates(subset=["law_text"]).reset_index(drop=True)
    print(f"[Data] unique rows: {len(df)}")

    model = get_model(EMBEDDING_MODEL)
    texts = [e5_encode_query(t) for t in df["law_text"].tolist()]

    # 배치 임베딩
    all_vecs = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding"):
        batch = texts[i:i + BATCH]
        vecs = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
        all_vecs.append(vecs)

    X = np.vstack(all_vecs).astype("float32")
    dim = X.shape[1]

    # FAISS Index
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    # 메타데이터 저장 (LLM 검색용)
    meta = df[["id", "file_name", "clauseField", "law_text"]].to_dict(orient="records")
    with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    with open(os.path.join(INDEX_DIR, "dim.txt"), "w", encoding="utf-8") as f:
        f.write(str(dim))

    print(f"✅ Done. vectors={len(df)}, dim={dim}")
    print(f"- FAISS : {os.path.join(INDEX_DIR, 'faiss.index')}")
    print(f"- META  : {os.path.join(INDEX_DIR, 'meta.pkl')}")
    print(f"- DIM   : {os.path.join(INDEX_DIR, 'dim.txt')}")


if __name__ == "__main__":
    main()
