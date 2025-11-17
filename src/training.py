import torch, os, json, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ----------------------------------------------------
# 1. 기본 설정
# ----------------------------------------------------
MY_TOKEN = os.environ.get("HF_TOKEN")            # 실제 토큰
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LEN = 512
OUTPUT_DIR = "models/llama31-8b-sft-fold10"

# torch compile 완전 비활성화
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ----------------------------------------------------
# 2. 모델 / 토크나이저 로드
# ----------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,    # ✅ 3070용
    bnb_4bit_use_double_quant=True,
)

print(f"'{BASE_MODEL}' 로드 중 ...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=MY_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    attn_implementation="sdpa",              # ✅ 빠른 Attention
    device_map="auto",
    dtype=torch.float16,
    token=MY_TOKEN,
)
print("--- 모델 로딩 완료 ---")

# ----------------------------------------------------
# 3. 4-bit + LoRA 준비
# ----------------------------------------------------
mdl = prepare_model_for_kbit_training(mdl)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj",
                    "o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
mdl = get_peft_model(mdl, lora_cfg)

# ----------------------------------------------------
# 4. 데이터 로드
# ----------------------------------------------------
TRAIN_PATH = "data/kfold_data/train_fold_10.jsonl"
VAL_PATH   = "data/kfold_data/val_fold_10.jsonl"

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train_rows = load_jsonl(TRAIN_PATH)
val_rows = load_jsonl(VAL_PATH)

train_df = pd.DataFrame(train_rows)
val_df = pd.DataFrame(val_rows)

# ✅ 테스트용으로 마지막 100개만 사용 (필요 시 제거)
#train_df = train_df.iloc[-100:].reset_index(drop=True)
print(f"Train 샘플 수: {len(train_df)}, Val 샘플 수: {len(val_df)}")

SYSTEM_PROMPT = (
    "당신은 약관의 공정성을 분석하는 법률 전문가입니다.\n"
    "문맥상 주체 (고객/ 사업자) 를 명확히 구분하세요.\n"
    "반드시 아래 한 줄 포맷만 출력하세요:\n"
    "분야: <정수> / 불공정여부: <유리|불리> / 근거: <간결한 문장 또는 '해당 없음'>"
)

def to_messages(r):
    inst, inp, out = r.get("instruction",""), r.get("input",""), r.get("output","")
    user_text = inst if not inp else f"{inst}\n\n입력:\n{inp}"
    return [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user_text},
        {"role":"assistant","content":out.strip()},
    ]
def format_example(ex):
    text = tok.apply_chat_template(
        to_messages(ex), tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

# ----------------------------------------------------
# 데이터셋 변환 (버그 수정)
# ----------------------------------------------------
train_ds = Dataset.from_pandas(train_df)
train_ds = train_ds.map(format_example, remove_columns=list(train_df.columns))

val_ds = Dataset.from_pandas(val_df)
val_ds = val_ds.map(format_example, remove_columns=list(val_df.columns))


print(f"데이터셋: train {len(train_ds)}, val {len(val_ds)}")

# ----------------------------------------------------
# 5. 학습 설정
# ----------------------------------------------------
sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    logging_steps=10,
    eval_strategy="epoch",             # ✅ 평가 간소화
    save_strategy="epoch",             # ✅ 저장 간소화
    save_total_limit=1,
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    report_to="none",
    fp16=True, bf16=False,
    dataloader_num_workers=0,          # ✅ WSL 안정화
    dataloader_pin_memory=False,
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    group_by_length=True,               # ✅ 속도/안정 ↑
    seed=42,
)

# ----------------------------------------------------
# 6. Trainer 실행
# ----------------------------------------------------
trainer = SFTTrainer(
    model=mdl,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_cfg,
    processing_class=tok,
)

print("--- 🚀 파인튜닝 시작 ---")
trainer.train()
print("--- 🏁 파인튜닝 완료 ---")

trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"✅ 저장 완료: {OUTPUT_DIR}")
