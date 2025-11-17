import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ----- 경로/모델 -----
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 허깅페이스에서 받은 동일 베이스
ADAPTER_DIR = "/home/ally010314/nlp_project/models/llama31-8b-sft-fold2"       # 네가 저장한 폴더

# (옵션) HF 토큰이 필요하면 아래 변수에 넣어줘
HF_TOKEN = "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"  # 없으면 None
token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

# ----- 토크나이저 -----
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True, **token_kwargs)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ----- 4-bit 로 베이스 모델 로드 (VRAM 8GB용) -----
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    **token_kwargs
)

# ----- LoRA 어댑터 얹기 -----
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# ----- 프롬프트 구성 (학습 때와 동일한 템플릿) -----
SYSTEM_PROMPT = (
    "당신은 약관의 공정성을 분석하는 법률 전문가입니다.\n"
    "문맥상 주체 (고객/ 사업자) 를 명확히 구분하세요.\n"
    "반드시 아래 한 줄 포맷만 출력하세요:\n"
    "분야: <정수> / 불공정여부: <유리|불리> / 근거: <간결한 문장 또는 '해당 없음'>"
)

def infer(clause: str, max_new_tokens=512):
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": f"다음 약관 조항의 문맥을 이해하여 분야 분류, 불공정 여부 판단, 판단 근거를 요약하시오.\n\n입력:\n{clause}"}
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # 평가 일관성 위해 greedy (원하면 True)
            temperature=0.7, top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    gen = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    return gen.strip()

# ----- 사용 예시 -----
# ----- 사용 예시 -----
clause = """제8조 (입실 전 계약의 해제)
② 제1항에도 불구하고 이용자가 다음 각 호의 사유로 계약을 해제하는 경우에는 사업자는 이용자에게 계약금을 환급하여야 합니다.
2. 특정 병원에서의 출산을 조건으로 계약을 체결하였으나 응급상황의 발생으로 산모가 다른 병원에서 출산한 경우"""
print(infer(clause))

clause = """제8조 (입실 전 계약의 해제)
② 제1항에도 불구하고 이용자가 다음 각 호의 사유로 계약을 해제하는 경우에는 사업자는 이용자에게 계약금을 환급합니다.
3. 산모 또는 신생아가 질병,상해 등으로 4일 이상의 입원치료가 필요하여 산후조리원을 이용할 수 없는 경우"""
print(infer(clause))

clause = """제8조 (입실 전 계약의 해제)
② 제1항에도 불구하고 이용자가 다음 각 호의 사유로 계약을 해제하는 경우에는 사업자는 이용자에게 계약금을 환급해야 합니다.
4. 천재지변, 지진, 풍수해, 벼락, 화재, 붕괴, 전쟁, 이와 유사한 재해 등 기타 불가항력적인 사유로 인하여 산후조리원을 이용할 수 없는 경우"""
print(infer(clause))

clause = """제9조 (입실 후 계약의 해지)
② 이용자가 개인 사정의 변경이나 단순변심 등 정당한 사유 없이 조기 퇴실하는 경우에는 사업자는 이용자에게 총 이용금액에서 실제 이용기간에 해당하는 요금 더하기 총 이용금액의 22.5퍼센트을 공제한 잔액을 환급합니다. 다만, 산모 또는 신생아의 발병 등으로 인한 입원치료의 필요로 조기퇴실이 불가피한 경우는 총 이용금액에서 실제 이용기간에 해당하는 요금만을 공제합니다."""
print(infer(clause))

clause = """제9조 (입실 후 계약의 해지)
② 이용자가 개인 사정의 변경이나 단순변심 등 정당한 사유 없이 조기 퇴실하는 경우에는 사업자는 이용자에게 총 이용금액에서 실제 이용기간에 해당하는 요금 더하기 총 이용금액의 35퍼센트을 공제한 잔액을 환급합니다. 다만, 산모 또는 신생아의 발병 등으로 인한 입원치료의 필요로 조기퇴실이 불가피한 경우는 총 이용금액에서 실제 이용기간에 해당하는 요금만을 공제합니다."""
print(infer(clause))

clause = """제10조 (사업자의 의무)
④ 사업자는 통신판매 매체(홈페이지 등)를 이용하여 표시․광고하거나 계약을 체결하고자 하는 경우에는 다음 각 호의 사항을 고객이 알 수 있도록 통신판매 매체에 게시하여야 한다."""
print(infer(clause))

clause = """제12조(인출한도 제한) 
고객이 보유한 계좌가 12년 이상 장기미사용 계좌에 해당하는 경우 자동화기기(ATM,CD)를 이용한 1일 인출한도는 회사가 고객에게 안내한 금액 이하로 제한됩니다."""
print(infer(clause))

clause = """제11조 (이용자의 의무)
① 이용자는 입실하는 때에 산모와 신생아의 건강 상태 및 기왕력을 고지합니다."""
print(infer(clause))

clause = """제 10 조 (계약해제)
1. 을이 본 계약에서 정하는 사항을 위반하거나 아래 각 호에 해당하는 행위를 하였을 경우에는 갑은 이행의 최고등 다른 절차를 취함이 없이 일방적으로 본 계약을 해제할 수 있다.
(1) 기타 관리상 필요에 의한 갑의 요구에 불응하였을 때"""
print(infer(clause))
clause = """제5조(지체상금)
갑 및 병은 제1조 3항에서 정한 입주예정 기일에 입주를 시키지 못할 경우에는 기 납부한 중도금에 대하여 본조 제2항에 의한 연체료율에 의거 을에게 지체상금을 지불하거나 잔여대금에서 공제할 수 있다."""
print(infer(clause))
clause = """제9조(계약해제 및 해지)
1. 을이 아래 각호에 해당하는 행위를 하였을 경우에는 갑은 이 계약을 해제하거나 해지할 수 있다.
(1) 허위, 기타 부정한 방법으로 분양을 받았을 경우
(2) 중도금 및 할부금(할부이자 포함)을 납부기한내에 납부하지 아니하고 연체한 경우
(3) 계약서상의 의무를 위반한 경우
(4) 갑은 이계약이 1항에 의거 해제된 때에는 을이 기불입한 분양대금, 이자, 연체료(관리비연체료는 제외)등 납부금에서 위약금, 물건사용료, 관리비등 체납금을 공제한 나머지 금액을 이자없이 반환한다. 또한 을이 위 물건 인도시 계약당시 원상으로 복구하지 아니한 경우에는 원상복구비를 기불입한 금액에서 공제한다."""
print(infer(clause))
clause = """제 10 조 (입양 후 15 이내 질병 발생 시)
00000가 제반비용을 들여 회복시켜 소비자에게 인도를 원칙으로 한다.(환불X)(다만, 00000가 책임하의 회복기간이 30일 경과하거나 판매자 관리 중 폐사 시에는 동종의 애완동물로 교환) 사인이 소비자의 중대한 과실이나 불분명한 경우 보상을 요구할 수 없다.
ᄋ특약사항
00000에서 도매로 강아지, 고양이를 분양하였기에 폐사시 동종교환시 추가비용 15만원 발생 및 환불 불가"""
print(infer(clause))
clause = """제3조 (위약금)
제5항. 을의 귀책사유로 인한 해약으로 갑에게 손해를 끼쳤을 경우에는 위약금과 별도의손해배상금을 지급하여야 한다."""
print(infer(clause))
