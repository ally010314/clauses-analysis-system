import os
import json
import glob
from tqdm import tqdm

# --- 설정 ---
BASE_DATA_DIR = "원본데이터"
TRAIN_OUTPUT_FILE = "train.jsonl"
VALID_OUTPUT_FILE = "validation.jsonl"
LAW_DB_OUTPUT_FILE = "law_db_clauses.json"
SFT_INSTRUCTION = "다음 약관 조항의 문맥을 이해하여 분야 분류, 불공정 여부 판단, 판단 근거를 요약하시오."

# RAG DB용 법령 조항을 중복 없이 저장하기 위한 Set
law_clauses_set = set()
# --- ---

def process_directory(data_type):
    """
    지정된 타입(TL 또는 VL)의 디렉토리를 순회하며 SFT 데이터를 생성하고
    법령 DB 텍스트를 추출합니다.
    """
    
    if data_type == "TL":
        input_dir = os.path.join(BASE_DATA_DIR, "TL_2.약관")
        output_file = TRAIN_OUTPUT_FILE
        print(f"--- 1. SFT 학습 데이터셋 생성 시작 ({input_dir}) ---")
    elif data_type == "VL":
        input_dir = os.path.join(BASE_DATA_DIR, "VL_2.약관")
        output_file = VALID_OUTPUT_FILE
        print(f"\n--- 2. SFT 검증 데이터셋 생성 시작 ({input_dir}) ---")
    else:
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        pass 

    json_files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print(f"[경고] {input_dir} 에서 .json 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    processed_count = 0
    for file_path in tqdm(json_files, desc=f"Processing {data_type} files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- SFT 데이터 포맷팅 (제안서 2-2 기준) ---
            input_text = "\n".join(data.get('clauseArticle', []))
            if not input_text:
                continue

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
                "output": output_text
            }

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
            
            processed_count += 1

            # --- RAG 법령 DB 추출 (제안서 2-1.나 기준) ---
            law_list = data.get('relateLaword')
            if law_list:
                
                #
                # =================== [수정된 부분] ===================
                # .strip()으로 각 항목의 앞뒤 공백/줄바꿈을 제거한 후
                # 빈 문자열이 아닌 경우에만 set에 추가합니다.
                #
                for law_text in law_list:
                    stripped_text = law_text.strip()
                    if stripped_text: # 빈 문자열이 아닐 때만 추가
                        law_clauses_set.add(stripped_text)
                # =======================================================
                #

        except json.JSONDecodeError:
            print(f"\n[오류] {file_path} 파일이 JSON 형식이 아닙니다.")
        except Exception as e:
            print(f"\n[오류] {file_path} 처리 중 문제 발생: {e}")
            
    print(f"--- {data_type} 처리 완료: 총 {processed_count}개 항목을 {output_file}에 저장했습니다. ---")


# --- 메인 스크립트 실행 ---
if __name__ == "__main__":
    
    process_directory("TL")
    process_directory("VL")

    print(f"\n--- 3. RAG용 법령 DB 생성 시작 ---")
    if law_clauses_set:
        law_list = sorted(list(law_clauses_set)) 
        with open(LAW_DB_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(law_list, f, ensure_ascii=False, indent=4)
        print(f"--- RAG DB 생성 완료: 총 {len(law_list)}개의 고유 법령 조항을 {LAW_DB_OUTPUT_FILE}에 저장했습니다. ---")
    else:
        print("[경고] 추출된 'relateLaword'가 없어 RAG DB 파일을 생성하지 않았습니다.")
        
    print("\n=== 모든 전처리 작업이 완료되었습니다. ===")