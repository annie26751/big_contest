import streamlit as st
import pandas as pd
import numpy as np
import random

# --- 모듈에서 필요한 함수와 상수를 가져옵니다 ---
from data_processor import load_fixed_data, analyze_merchant, FIXED_DATA_PATH, AGE_GENDER_COLS, AGE_GENDER_NAMES
from gemini_api import generate_marketing_text_with_gemini

# Streamlit 데이터 로드 및 메인 함수
@st.cache_resource(ttl=3600) 
def cached_load_data(path):
    """Streamlit 캐싱을 적용한 데이터 로드 함수입니다."""
    try:
        # data_processor의 함수를 호출
        return load_fixed_data(path)
    except (FileNotFoundError, UnicodeDecodeError, ValueError, Exception) as e:
        # data_processor에서 발생한 에러를 여기서 Streamlit 에러로 표시
        st.error(f"{e}")
        st.stop()
        
def main():
    st.set_page_config(layout="wide", page_title="가맹점 통합 분석 대시보드")
    st.title("📊 빅콘테스트 AI 비밀상담사")

    # --- 데이터 로드 섹션 (고정 경로) ---
    st.sidebar.header("1. 데이터 로드 상태")
    st.sidebar.info(f"데이터 경로: `{FIXED_DATA_PATH}`")
    
    df_profile = None
    # 데이터 로드
    if 'df_profile' not in st.session_state:
        with st.spinner('데이터 로드 및 전처리 중...'):
            # 캐싱된 로드 함수 사용
            df_profile = cached_load_data(FIXED_DATA_PATH)
            st.session_state['df_profile'] = df_profile
    
    df_profile = st.session_state['df_profile']
    st.sidebar.success(f"✅ 데이터 로드 성공! 총 {len(df_profile)}개 가맹점 분석 준비 완료.")
    
    # --- 분석 실행 섹션 ---
    st.sidebar.header("2. 가맹점 선택")
    merchant_ids = df_profile['ENCODED_MCT'].unique()
    selected_mct = st.sidebar.selectbox(
        "분석할 가맹점 구분 번호를 선택하세요:",
        merchant_ids
    )

    # 선택된 가맹점 데이터 추출 및 분석
    mct_data = df_profile[df_profile['ENCODED_MCT'] == selected_mct].iloc[0]
    analysis_result = analyze_merchant(mct_data)
    
    # --- 결과 표시 영역 (기본 정보 및 지표) ---
    st.header(f"'{selected_mct}' 가맹점 통합 진단 리포트")
    
    # I. 가맹점 기본 정보 
    st.subheader("1. 가맹점 기본 정보")
    static_info = analysis_result['static_info']
    status = "운영 중" if pd.isna(static_info.get('MCT_ME_D')) else f"폐업 ({static_info.get('MCT_ME_D')})"
    st.markdown(f"**주소:** {static_info.get('MCT_BSE_AR')}")
    
    cols = st.columns(4)
    cols[0].info(f"**업종:** {static_info.get('HPSN_MCT_ZCD_NM')}")
    cols[1].info(f"**상권:** {static_info.get('HPSN_MCT_BZN_CD_NM')}")
    cols[2].info(f"**개설일:** {static_info.get('ARE_D')}")
    cols[3].info(f"**상태:** {status}")

    st.markdown("---")
    
    # II. 3가지 핵심 진단 결과 출력
    st.subheader("2. 🔍 3가지 핵심 진단 결과")
    
    # 가. 고객층 분석
    st.markdown("#### 가. 고객층 분석 (타겟 고객 파악)")
    st.success(analysis_result['cust_analysis_text'])
    
    # 나. 재방문율 확인
    st.markdown("#### 나. 재방문율 확인 (고객 유지력)")
    st.info(analysis_result['retention_analysis_text'])

    # 다. 경쟁 환경 내 위치 파악
    st.markdown("#### 다. 경쟁 환경 내 위치 파악 (상권 및 업종)")
    st.warning(analysis_result['comp_analysis_text'])

    st.markdown("---")
    
    # ----------------------------------------
    # III. Gemini 기반 맞춤형 마케팅 제안 (버튼으로 실행)
    # ----------------------------------------
    st.subheader("3. 🧠 맞춤형 마케팅 제안")
    
    if 'last_mct' not in st.session_state or st.session_state['last_mct'] != selected_mct:
        st.session_state['marketing_proposal'] = "아래 버튼을 눌러 맞춤형 마케팅 전략을 생성하세요."
        st.session_state['last_mct'] = selected_mct
    
    
    # 버튼을 눌렀을 때 API 호출
    if st.button("마케팅 제안 생성 요청", type="primary"):
        # gemini_api.py에서 직접 키를 확인
        with st.spinner('Gemini API가 3가지 진단 결과를 바탕으로 마케팅 전략을 생성 중입니다...'):
            marketing_proposal = generate_marketing_text_with_gemini(analysis_result, selected_mct)
            st.session_state['marketing_proposal'] = marketing_proposal

    # 저장된 제안 텍스트 표시
    st.markdown(st.session_state.get('marketing_proposal', "아래 버튼을 눌러 맞춤형 마케팅 전략을 생성하세요."))

    st.markdown("---")
    
    # --- 추가 정보 표시 (데이터 차트) ---
    with st.expander("📊 상세 고객 비율 차트 보기"):
        st.caption("고객 비율은 월별 데이터의 2년 평균값입니다.")
        # data_processor에서 가져온 상수를 사용
        ag_data = mct_data[AGE_GENDER_COLS].rename(index=AGE_GENDER_NAMES).round(1).sort_values(ascending=False).head(10).reset_index()
        ag_data.columns = ['연령/성별 그룹', '비중 (%)']
        st.bar_chart(ag_data.set_index('연령/성별 그룹'))


if __name__ == '__main__':
    main()
