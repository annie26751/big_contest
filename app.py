import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 모듈에서 필요한 함수와 상수를 가져옵니다 ---
from data_processor import load_fixed_data, analyze_merchant, FIXED_DATA_PATH, AGE_GENDER_COLS, AGE_GENDER_NAMES
from gemini_api import generate_marketing_text_with_gemini

@st.cache_resource(ttl=3600)
def cached_load_data(path):
    """Streamlit 캐싱을 적용한 데이터 로드 함수입니다."""
    try:
        return load_fixed_data(path)
    except (FileNotFoundError, UnicodeDecodeError, ValueError, Exception) as e:
        st.error(f"{e}")
        st.stop()

def main():
    st.set_page_config(layout="wide", page_title="내 가게를 살리는 AI 비밀상담사")
    st.title("💡 내 가게를 살리는 AI 비밀상담사")
    st.markdown("데이터와 AI를 기반으로 우리 가게의 문제를 진단하고, 핵심 고객을 위한 맞춤 마케팅 전략을 찾아보세요.")

    # --- 사이드바 ---
    st.sidebar.header("1. 데이터 로드")
    st.sidebar.info(f"분석 데이터: `{FIXED_DATA_PATH}`")

    if 'df_profile' not in st.session_state:
        with st.spinner('데이터 로드 및 전처리 중...'):
            df_profile = cached_load_data(FIXED_DATA_PATH)
            st.session_state['df_profile'] = df_profile

    df_profile = st.session_state['df_profile']
    st.sidebar.success(f"✅ 총 {len(df_profile)}개 가맹점 분석 완료.")

    st.sidebar.header("2. 가맹점 선택")
    merchant_ids = df_profile['ENCODED_MCT'].unique()
    selected_mct = st.sidebar.selectbox(
        "분석할 가맹점 구분 번호를 선택하세요:",
        merchant_ids,
        key="merchant_selector"
    )

    # --- 분석 실행 및 결과 저장 ---
    if 'last_mct' not in st.session_state or st.session_state['last_mct'] != selected_mct:
        st.session_state['analysis_result'] = analyze_merchant(df_profile[df_profile['ENCODED_MCT'] == selected_mct].iloc[0])
        st.session_state['marketing_proposal'] = ""
        st.session_state['show_mbti_description'] = False # 유형 설명 표시 상태 초기화
        st.session_state['last_mct'] = selected_mct

    analysis_result = st.session_state['analysis_result']
    summary = analysis_result['summary']
    persona = analysis_result['persona']
    mbti_result = analysis_result['mbti']
    mct_data = analysis_result['raw_data']

    # --- 메인 화면 구성 ---
    # 헤더와 가게 유형 버튼을 한 줄에 배치
    col_header1, col_header2 = st.columns([2, 1])
    with col_header1:
        st.header(f"'{selected_mct}' 가맹점 종합 리포트")
    with col_header2:
        # [수정] 버튼 스타일링 CSS 제거
        if st.button(f"✨ 우리 가게 유형: {mbti_result['name']}"):
            # 버튼 클릭 시 설명 표시 상태를 토글
            st.session_state.show_mbti_description = not st.session_state.get('show_mbti_description', False)

    # 버튼 클릭 상태에 따라 유형 설명을 표시
    if st.session_state.get('show_mbti_description', False):
        st.info(f"**{mbti_result['name']}**: {mbti_result['description']}")
    st.markdown("---")


    # 1. 가맹점 기본 정보
    with st.expander("① 가맹점 기본 정보 보기", expanded=True):
        static_info = summary['static_info']
        status = "운영 중" if pd.isna(static_info.get('MCT_ME_D')) else f"폐업 ({static_info.get('MCT_ME_D')})"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📂 업종:** {static_info.get('HPSN_MCT_ZCD_NM')}")
            st.markdown(f"**📍 주소:** {static_info.get('MCT_BSE_AR')}")
        with col2:
            st.markdown(f"**🏪 상권:** {static_info.get('HPSN_MCT_BZN_CD_NM', '정보 없음')}")
            st.markdown(f"**📈 상태:** {status}")

    # 2. 데이터 기반 핵심 진단
    st.subheader("② 데이터 기반 핵심 진단")
    st.success(f"**[고객층 분석]** {summary['cust_analysis_text']}")
    st.info(f"**[고객 유지력]** {summary['retention_analysis_text']}")
    st.warning(f"**[경쟁 환경]** {summary['comp_analysis_text']}")
    st.markdown("---")

    # 3. 페르소나 분석 결과
    st.subheader("👤 ③ AI가 분석한 핵심 고객 페르소나")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(persona['image_url'], caption=persona['name'], use_container_width=True)
    with col2:
        st.markdown(f"#### {persona['name']}")
        st.write(persona['description'])
        st.markdown("##### 이 고객이 우리 가게를 찾는 이유 (Goals)")
        for goal in persona['goals']:
            st.markdown(f"- {goal}")
        st.markdown("##### 이 고객이 겪는 어려움 (Pain Points)")
        for pp in persona['pain_points']:
            st.markdown(f"- {pp}")
    st.markdown("---")

    # 4. 맞춤형 마케팅 제안 (Gemini)
    st.subheader("🧠 ④ AI 비밀상담사의 맞춤 마케팅 제안")
    if st.button("AI 마케팅 전략 생성하기", type="primary"):
        with st.spinner('AI 비밀상담사가 페르소나와 데이터를 분석해 맞춤 전략을 짜고 있어요...'):
            marketing_proposal = generate_marketing_text_with_gemini(summary, persona, mbti_result, selected_mct)
            st.session_state['marketing_proposal'] = marketing_proposal

    if st.session_state.get('marketing_proposal'):
        st.markdown(st.session_state['marketing_proposal'])
    else:
        st.info("버튼을 눌러 우리 가게만을 위한 맞춤 마케팅 전략을 확인해보세요!")


if __name__ == '__main__':
    main()
