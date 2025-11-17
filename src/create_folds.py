import json
import os
from sklearn.model_selection import KFold
from tqdm import tqdm

# --- 설정 ---
INPUT_FILE = "all_data.jsonl" # 1단계에서 생성된 전체 데이터 파일
OUTPUT_DIR = "kfold_data"   # 10-fold 데이터가 저장될 폴더
N_SPLITS = 10               # 10-fold
RANDOM_SEED = 42            # 재현 가능성을 위한 시드
# --- ---

def create_kfold_datasets():
    print(f"--- 1. {INPUT_FILE} 파일 읽기 시작 ---")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            # .jsonl 파일이므로 한 줄씩 읽어서 리스트로 만듭니다.
            all_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"[오류] {INPUT_FILE} 파일을 찾을 수 없습니다.")
        print("먼저 `preprocess_all.py` 스크립트를 실행하여 all_data.jsonl 파일을 생성해야 합니다.")
        return
    except json.JSONDecodeError:
        print(f"[오류] {INPUT_FILE} 파일의 형식이 잘못되었습니다. (JSONL 아님)")
        return

    if len(all_data) < N_SPLITS:
        print(f"[오류] 데이터 개수({len(all_data)}개)가 폴드 수({N_SPLITS}개)보다 적어 10-fold를 나눌 수 없습니다.")
        return
        
    print(f"--- 총 {len(all_data)}개의 데이터를 읽었습니다. ---")
    print(f"--- 2. {N_SPLITS}-Fold 데이터 분할 시작 (Seed={RANDOM_SEED}) ---")

    # KFold 객체 생성
    # shuffle=True: 데이터를 섞어서 공정하게 나눕니다.
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # kf.split()은 데이터의 (학습용 인덱스, 검증용 인덱스) 쌍을 반환합니다.
    # enumerate로 1부터 시작하는 폴드 번호를 매깁니다.
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_data), 1):
        
        # 1. 학습용 데이터 추출 및 저장
        train_file_name = os.path.join(OUTPUT_DIR, f"train_fold_{fold_idx}.jsonl")
        with open(train_file_name, 'w', encoding='utf-8') as f:
            for i in tqdm(train_indices, desc=f"Fold {fold_idx} (Train)", leave=False):
                f.write(json.dumps(all_data[i], ensure_ascii=False) + "\n")

        # 2. 검증용 데이터 추출 및 저장
        val_file_name = os.path.join(OUTPUT_DIR, f"val_fold_{fold_idx}.jsonl")
        with open(val_file_name, 'w', encoding='utf-8') as f:
            for i in val_indices:
                f.write(json.dumps(all_data[i], ensure_ascii=False) + "\n")
                
        print(f"[Fold {fold_idx}] 완료: Train {len(train_indices)}개, Val {len(val_indices)}개")

    print(f"\n--- 3. 모든 K-Fold 데이터 생성이 완료되었습니다. ({OUTPUT_DIR} 폴더 확인) ---")

# --- 메인 스크립트 실행 ---
if __name__ == "__main__":
    # scikit-learn이 필요합니다
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        print("[오류] scikit-learn 라이브러리가 필요합니다.")
        print("터미널에서 `pip install scikit-learn` 명령어를 실행해주세요.")
        exit()
        
    create_kfold_datasets()