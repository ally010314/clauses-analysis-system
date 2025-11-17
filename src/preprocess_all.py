import os
import json
import glob
from tqdm import tqdm

# --- 설정 ---
# 원본 데이터 폴더들 (TL, VL 모두 포함)
DATA_DIRS = ["원본데이터/TL_2.약관", "원본데이터/VL_2.약관"]

# SFT 학습용/검증용 출력 파일 (JSON Lines 형식)
ALL_DATA_OUTPUT_FILE = "all_data.jsonl" # 1. 모든 데이터를 여기로 합칩니다.

# RAG DB 구축을 위한 법령 텍스트 원본 파일
LAW_DB_OUTPUT_FILE = "law_db_clauses.json"

# 제안서(2-2)에서 정의한 고정 Instruction
SFT_INSTRUCTION = "다음 약관 조항의 문맥을 이해하여 분야 분류, 불공정 여부 판단, 판단 근거를 요약하시오."

# RAG DB용 법령 조항을 중복 없이 저장하기 위한 Set
law_clauses_set = set()
# --- ---

def process_all_data():
    """
    TL과 VL 폴더 구분 없이 모든 데이터를 읽어 all_data.jsonl로 합칩니다.
    """
    
    print(f"--- 1. 전체 SFT 데이터셋 생성 시작 ---")
    
    # 기존 출력 파일이 있다면 초기화
    with open(ALL_DATA_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass # 파일 내용을 비웁니다.

    all_json_files = []
    for input_dir in DATA_DIRS:
        json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
        if not json_files:
            print(f"[경고] {input_dir} 에서 .json 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        all_json_files.extend(json_files)

    if not all_json_files:
        print("[오류] 처리할 .json 파일이 전혀 없습니다. 원본데이터 폴더를 확인하세요.")
        return

    processed_count = 0
    # tqdm으로 진행 상황 표시
    for file_path in tqdm(all_json_files, desc=f"Processing all files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- SFT 데이터 포맷팅 (제안서 2-2 기준) ---
            input_text = "\n".join(data.get('clauseArticle', []))
            if not input_text:
                continue # 약관 조항이 없으면 건너뜁니다.

            field = data.get('clauseField', '기타')
            judgement = "유리" if data.get('dvAntageous') == "1" else "불리"
            basis_list = data.get('illdcssBasiss')
            
            if basis_list:
                basis = " ".join(basis_list).replace("\n", " ")
            else:
                basis = "해당 없음" 
            
            output_text = f"분야: {field} / 불공정여부: {judgement} / 근거: {basis}"

            sft_entry = {
                "instruction": SFT_INSTRUCTION,
                "input": input_text,
                "output": output_text,
                # [중요] RAG 평가를 위해 정답 법령도 원본 데이터에 포함시킵니다.
                "ground_truth_laws": data.get('relateLaword', []) 
            }

            # 파일에 JSONL 형식으로 추가 (append)
            with open(ALL_DATA_OUTPUT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
            
            processed_count += 1

            # --- RAG 법령 DB 추출 (제안서 2-1.나 기준) ---
            law_list = data.get('relateLaword')
            if law_list:
                for law_text in law_list:
                    stripped_text = law_text.strip()
                    if stripped_text: # 빈 문자열이 아닐 때만 추가
                        law_clauses_set.add(stripped_text)

        except json.JSONDecodeError:
            print(f"\n[오류] {file_path} 파일이 JSON 형식이 아닙니다.")
        except Exception as e:
            print(f"\n[오류] {file_path} 처리 중 문제 발생: {e}")
            
    print(f"--- 전체 데이터 처리 완료: 총 {processed_count}개 항목을 {ALL_DATA_OUTPUT_FILE}에 저장했습니다. ---")


# --- 메인 스크립트 실행 ---
if __name__ == "__main__":
    
    # 1. 모든 데이터(TL+VL) 처리
    process_all_data()

    # 2. RAG용 법령 DB 저장 (제안서 2-1.나)
    print(f"\n--- 2. RAG용 법령 DB 생성 시작 ---")
    if law_clauses_set:
        law_list = sorted(list(law_clauses_set)) 
        with open(LAW_DB_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(law_list, f, ensure_ascii=False, indent=4)
        print(f"--- RAG DB 생성 완료: 총 {len(law_list)}개의 고유 법령 조항을 {LAW_DB_OUTPUT_FILE}에 저장했습니다. ---")
    else:
        print("[경고] 추출된 'relateLaword'가 없어 RAG DB 파일을 생성하지 않았습니다.")
        
    print("\n=== 모든 전처리 작업이 완료되었습니다. ===")