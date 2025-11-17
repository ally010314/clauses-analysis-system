# [UI] Streamlit 라이브러리 임포트
import streamlit as st
import requests
import json

# [UI] FastAPI 서버 주소
API_URL = "http://127.0.0.1:8000/analyze"

# --- [추가] 분야 코드-이름 매핑 딕셔너리 ---
CATEGORY_MAP = {
    "1": "가맹계약",
    "2": "공급계약",
    "3": "분양계약",
    "4": "신탁계약",
    "5": "임대차계약",
    "6": "입소, 입주, 입점계약",
    "7": "신용카드",
    "8": "은행여신",
    "9": "은행전자금융서비스",
    "10": "전자결제수단",
    "11": "전자금융거래",
    "12": "상해보험",
    "13": "손해보험",
    "14": "질병보험",
    "15": "연금보험",
    "16": "자동차보험",
    "17": "책임보험",
    "18": "화재보험",
    "19": "증권사1",
    "20": "증권사2",
    "21": "증권사3",
    "22": "여객운송",
    "23": "화물운송",
    "24": "개인정보취급방침",
    "25": "게임",
    "26": "국내·외 여행",
    "27": "결혼정보서비스",
    "28": "렌트(자동차 이외)",
    "29": "마일리지/포인트",
    "30": "보증",
    "31": "사이버몰",
    "32": "산후조리원",
    "33": "상조서비스",
    "34": "상품권",
    "35": "생명보험",
    "36": "예식업",
    "37": "온라인서비스",
    "38": "자동차 리스 및 렌트",
    "39": "체육시설",
    "40": "택배",
    "41": "통신, 방송서비스",
    "42": "교육",
    "43": "매매계약"
}
# --- [추가] 종료 ---

# --- [UI] Streamlit UI 구성 ---
st.set_page_config(page_title="불공정 약관 분석 시스템", layout="wide")
st.title("⚖️ 불공정 약관 분석 시스템")

# [UI] 입력창
st.header("1. 분석할 약관을 입력하세요")
clause_text = st.text_area("약관 내용 입력", height=150, label_visibility="collapsed")

# [UI] 실행 버튼
if st.button("분석 실행", type="primary"):
    if not clause_text.strip():
        st.error("약관 내용을 입력해주세요.")
    else:
        # [UI] 로딩 스피너
        with st.spinner("AI 모델이 분석 중입니다... (약 10~30초 소요)"):
            try:
                # [API] FastAPI 서버에 POST 요청
                payload = {"clause_text": clause_text}
                response = requests.post(API_URL, json=payload, timeout=300)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # --- [수정] LLM 결과 파싱 및 UI 통일 ---
                    st.header("2. 분석 결과")
                    llm_output = result.get("llm_output", "분석 결과 없음")
                    
                    # 파싱된 결과를 저장할 변수 초기화
                    category_name = "N/A"
                    fairness = "N/A"
                    reason = "분석 내용 없음"

                    try:
                        # 1. ' / ' 기준으로 파싱
                        parts = llm_output.split(' / ')
                        parsed_data = {}
                        for part in parts:
                            if ':' in part:
                                key, value = part.split(':', 1)
                                parsed_data[key.strip()] = value.strip()

                        # 2. 파싱된 데이터 추출
                        category_num = parsed_data.get('분야', 'N/A') 
                        fairness = parsed_data.get('불공정여부', 'N/A')
                        reason = parsed_data.get('근거', '분석 내용 없음')
                        
                        category_name = CATEGORY_MAP.get(category_num, f"유형 {category_num}") 

                        # 3. [수정] st.metric 대신 st.subheader와 st.info/error/success 사용
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("판단 분야")
                            st.info(category_name) # [수정] 영역 통일

                        with col2:
                            st.subheader("불공정 여부") # [수정] 글씨 크기 통일
                            # [수정] 영역 통일 및 시각적 강조
                            if fairness == "불리":
                                st.error(fairness) # 빨간색 박스
                            elif fairness == "유리":
                                st.success(fairness) # 초록색 박스
                            else:
                                st.info(fairness) # 파란색 박스

                        # 4. 근거 표시 (기존과 동일하게 유지하여 통일성 맞춤)
                        st.subheader("판단 근거")
                        st.info(reason)

                    except Exception as e:
                        st.warning(f"결과 포맷 파싱 중 오류 발생: {e}")
                        st.info(llm_output)
                    # --- [수정] 종료 ---

                    
                    # --- '유리'가 아닐 경우에만 3번 섹션 표시 (기존과 동일) ---
                    if fairness != "유리":
                        st.header("3. 관련 법령 검색 결과")
                        retrieved_laws = result.get("retrieved_laws")
                        
                        if retrieved_laws:
                            st.subheader("유사한 관련 법령 조항 (Top 5)")
                            
                            # [추가] 거리(Distance) 값에 대한 설명 추가
                            st.caption("ℹ️  AI가 계산한 유사도 점수가 1에 가까울수록 입력된 약관과 관련성이 큰 조항입니다.")
                            
                            for i, law in enumerate(retrieved_laws, 1):
                                law_text = law.get('law_text', '내용 없음')
                                # [추가] API에서 보낸 distance 값 가져오기
                                similarity = law.get('similarity') 
                                
                                # [수정] expander 제목에 거리(distance) 값을 함께 표시
                                if similarity is not None:
                                    # 예: 1. (거리: 0.1234) 사업자의 귀책사유로...
                                    title = f"**{i}. (유사도: {similarity:.4f})** {law_text[:50]}..."
                                else:
                                    # (Fallback) api.py가 distance를 안 보냈을 경우
                                    title = f"**{i}.** {law_text[:50]}..."
                                
                                with st.expander(title):
                                    st.code(law_text, language="text")
                        
                        elif fairness == "불리":
                            # RAG 결과가 없지만 LLM이 '불리'로 판단한 경우
                            st.warning("LLM이 '불리'로 판단했으나, RAG 검색 결과 일치하는 법령이 없습니다.")

                else:
                    st.error(f"API 서버 오류: {response.status_code}\n{response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("API 서버에 연결할 수 없습니다. (FastAPI 서버가 실행 중인지 확인하세요)")
            except Exception as e:
                st.error(f"알 수 없는 오류 발생: {e}")