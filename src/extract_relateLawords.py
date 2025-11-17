import os
import json
import pandas as pd

# 약관 JSON들이 저장된 상위 디렉토리
BASE_DIR = "nlp_project/data/original_data/TL_2.약관/02.불리"

# 결과를 저장할 파일
OUTPUT_CSV = "nlp_project/data/relateLaword_index.csv"
OUTPUT_JSON = "nlp_project/data/relateLaword_index.json"

# 결과 리스트
records = []
law_id = 0

for root, _, files in os.walk(BASE_DIR):
    for fname in files:
        if not fname.endswith(".json"):
            continue

        path = os.path.join(root, fname)
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[WARN] {fname} 파싱 실패: {e}")
                continue

        # relateLaword가 리스트 형태로 존재하는 경우
        laws = data.get("relateLaword", [])
        if not isinstance(laws, list):
            continue

        for law_text in laws:
            law_text = law_text.strip()
            if not law_text:
                continue
            records.append({
                "id": law_id,
                "file_name": fname,
                "clauseField": data.get("clauseField"),
                "law_text": law_text
            })
            law_id += 1

# DataFrame으로 저장
df = pd.DataFrame(records)
df = df.drop_duplicates(subset=["law_text"]).reset_index(drop=True)
df = df.sort_values(by="law_text", ascending=True).reset_index(drop=True)
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

print(f"✅ 추출 완료: {len(df)}개 법령 조문을 저장했습니다.")
print(f"- CSV: {OUTPUT_CSV}")
print(f"- JSON: {OUTPUT_JSON}")
