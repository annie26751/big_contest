import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from docx import Document
import time
from data_processor import load_fixed_data, analyze_merchant, FIXED_DATA_PATH
from gemini_api import generate_marketing_text_with_gemini, generate_chat_response_with_gemini

@st.cache_resource(ttl=3600)
def cached_load_data(path):
    """Streamlit 캐싱을 적용한 데이터 로드 함수"""
    try:
        return load_fixed_data(path)
    except (FileNotFoundError, UnicodeDecodeError, ValueError, Exception) as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.info("FIXED_DATA_PATH 변수가 정확한지, 파일이 해당 경로에 존재하는지 확인해주세요.")
        st.stop()
        return None

def create_docx_report(mct_id, proposal, chat_history):
    """마케팅 전략과 챗봇 대화 내용으로 Word 문서를 생성하여 바이트 객체로 반환"""
    doc = Document()
    doc.add_heading(f"'{mct_id}' 가맹점 AI 마케팅 분석 리포트", level=1)
    doc.add_paragraph()

    doc.add_heading("🚀 AI 비밀상담사의 맞춤형 마케팅 플랜", level=2)
    for line in proposal.split('\n'):
        doc.add_paragraph(line)
    
    if len(chat_history) > 1:
        doc.add_paragraph()
        doc.add_heading("🤖 추가 상담 내용 (Q&A)", level=2)
        for message in chat_history[1:]:
            role = "Q (사용자)" if message["role"] == "user" else "A (AI 상담사)"
            p = doc.add_paragraph()
            p.add_run(f"{role}: ").bold = True
            p.add_run(message['content'])
            doc.add_paragraph()
            
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

def main():
    st.set_page_config(layout="wide", page_title="💡 내 가게를 살리는 AI 비밀상담사")

    st.markdown("""
    <div style="background-color:#f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <h1 style="text-align: center; color: black; margin: 0; font-size: 2.5rem;">💡 내 가게를 살리는 AI 비밀상담사</h1>
    </div>
    """, unsafe_allow_html=True)

    # --- Session State 초기화 ---
    if 'generating' not in st.session_state:
        st.session_state['generating'] = False

    if 'df_profile' not in st.session_state:
        with st.spinner('초기 데이터를 로드하는 중입니다...'):
            df_profile = cached_load_data(FIXED_DATA_PATH)
            if df_profile is not None:
                st.session_state['df_profile'] = df_profile
                st.session_state['merchant_ids'] = df_profile['ENCODED_MCT'].unique()
            else:
                st.error("데이터 로드에 실패하여 앱을 실행할 수 없습니다.")
                st.stop()
    
    all_merchant_ids = st.session_state.get('merchant_ids', [])
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.header("가맹점 선택")
        
        search_term = st.text_input(
            "가맹점 번호 검색:", 
            placeholder="여기에 번호 일부를 입력하세요"
        )

        if search_term:
            filtered_merchants = [
                mct for mct in all_merchant_ids if search_term in str(mct)
            ]
        else:
            filtered_merchants = all_merchant_ids

        selected_mct = st.selectbox(
            "분석할 가맹점을 선택하세요:",
            filtered_merchants,
            key="merchant_selector",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("이 솔루션은 빅콘테스트 2025 AI데이터 활용분야 참여를 위해 제작되었습니다.")

    if not selected_mct:
        st.info("사이드바에서 분석할 가맹점을 선택해주세요.")
        st.stop()

    if 'last_mct' not in st.session_state or st.session_state['last_mct'] != selected_mct:
        with st.spinner(f"'{selected_mct}' 가맹점 데이터를 분석하는 중..."):
            df_profile = st.session_state['df_profile']
            st.session_state['analysis_result'] = analyze_merchant(df_profile[df_profile['ENCODED_MCT'] == selected_mct].iloc[0])
            st.session_state['marketing_proposal'] = ""
            st.session_state['last_mct'] = selected_mct
            st.session_state.chat_messages = [] 

    analysis_result = st.session_state['analysis_result']
    summary = analysis_result['summary']
    persona = analysis_result['persona']
    mbti_result = analysis_result['mbti']

    tab1, tab2 = st.tabs(["📊 **종합 대시보드**", "🎯 **고객 페르소나 & AI 맞춤 전략**"])

    with tab1:
        st.subheader("가맹점 현황 요약 (At-a-glance)")
        with st.container(border=True):
            cols = st.columns(4)
            static_info = summary['static_info']
            status = "운영 중" if pd.isna(static_info.get('MCT_ME_D')) else f"폐업 ({static_info.get('MCT_ME_D')})"
            cols[0].metric("🏪 업종", static_info.get('HPSN_MCT_ZCD_NM', 'N/A'))
            cols[1].metric("📍 상권", static_info.get('HPSN_MCT_BZN_CD_NM', '정보 없음'))
            cols[2].metric("📈 상태", status)
            cols[3].metric("✨ 가게 유형", mbti_result['name'], help=mbti_result['description'])
        st.subheader("AI 데이터 진단")
        with st.container(border=True):
            cols = st.columns(3)
            with cols[0]:
                st.markdown("🎯 **고객층 분석**")
                st.success(summary['cust_analysis_text'], icon="👥")
            with cols[1]:
                st.markdown("🔄 **고객 유지력**")
                st.info(summary['retention_analysis_text'], icon="💖")
            with cols[2]:
                st.markdown("⚔️ **경쟁 환경**")
                st.warning(summary['comp_analysis_text'], icon="🛡️")

    with tab2:
        st.subheader("🎯 우리 가게의 핵심 고객은 누구일까요?")
        
        with st.container(border=True):
            persona_icon = persona.get("icon", "")
            st.markdown(f"### {persona_icon} {persona['name']}")

            description_html = persona['description'].replace('\n', '<br>')
            goals_html = "<ul>" + "".join([f"<li>{g}</li>" for g in persona['goals']]) + "</ul>"
            pain_points_html = "<ul>" + "".join([f"<li>{p}</li>" for p in persona['pain_points']]) + "</ul>"

            st.markdown(f"""
            <style>
                .persona-table {{
                    width: 100%;
                    border-collapse: collapse;
                    border: none;
                }}
                .persona-table th, .persona-table td {{
                    border: 1px solid #dfe3e8;
                    padding: 12px 15px;
                    text-align: left;
                    vertical-align: top;
                }}
                .persona-table th {{
                    background-color: #f0f2f6;
                    width: 20%;
                    font-weight: bold;
                }}
                .persona-table ul {{
                    padding-left: 20px;
                    margin: 0;
                }}
            </style>
            <table class="persona-table">
                <tr>
                    <th>소개</th>
                    <td>{description_html}</td>
                </tr>
                <tr>
                    <th>찾는 이유 (Goals)</th>
                    <td>{goals_html}</td>
                </tr>
                <tr>
                    <th>겪는 어려움 (Pain Points)</th>
                    <td>{pain_points_html}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🧠 AI 비밀상담사의 맞춤형 마케팅 플랜")
        st.warning("아래는 입력된 데이터와 페르소나를 기반으로 Gemini AI 마케팅 전략을 실시간으로 생성합니다.")
        
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            button_text = "🚀 생성중..." if st.session_state.generating else "🚀 AI 마케팅 전략 생성하기"
            if st.button(button_text, use_container_width=True, type="primary", disabled=st.session_state.generating):
                st.session_state.generating = True
                proposal = generate_marketing_text_with_gemini(summary, persona, mbti_result, selected_mct)
                st.session_state['marketing_proposal'] = proposal
                st.session_state.chat_messages = []
                st.session_state.generating = False
                st.rerun()

        st.markdown("---")

        if st.session_state.get('marketing_proposal'):
            st.markdown(st.session_state['marketing_proposal'])
            
            st.markdown("---")
            st.subheader("🤖 AI 마케팅 도구 추천")
            st.info("아래 도구들과 프롬프트 예시를 활용하여 마케팅 콘텐츠를 손쉽게 제작해보세요.")

            reel_tab, blog_tab, image_tab = st.tabs(["🎬 **릴스/숏폼 제작**", "✍️ **블로그 포스팅**", "🎨 **이미지 생성**"])

            with reel_tab:
                st.link_button(
                    "Vrew 바로가기", 
                    "https://vrew.voyagerx.com/",
                    help="영상과 음성을 분석해 자동으로 자막을 생성하고, 텍스트 편집만으로 영상을 손쉽게 컷 편집할 수 있는 도구입니다."
                )
                with st.expander("📝 **Vrew 활용 프롬프트 예시 펼쳐보기**"):
                    st.code(f"""
                    ### 릴스 대본 생성 프롬프트
    
                    **역할:**
                    당신은 '{summary['static_info'].get('HPSN_MCT_ZCD_NM')}' 가게를 운영하는 사장님 역할을 맡은 SNS 마케터입니다.
                    우리의 핵심 고객인 '{persona['name']}'의 관심을 끌 수 있는 30초 분량의 인스타그램 릴스 대본을 작성해주세요.
    
                    **릴스 컨셉:**
                    [사장님이 직접 가게의 매력을 소개하는 컨셉 / 고객이 직접 경험하는 듯한 1인칭 시점 컨셉 등]
    
                    **핵심 메시지:**
                    '{persona['goals'][0]}' 와 같은 고객의 니즈를 충족시키고, '{persona['pain_points'][0]}' 같은 불편함을 해결해준다는 점을 강조해주세요.
    
                    **포함할 내용:**
                    - 시선을 사로잡는 오프닝 멘트 (3초 이내)
                    - 가게의 핵심 메뉴 또는 서비스 소개
                    - 고객에게 제공하는 특별한 혜택 (이벤트, 할인 등)
                    - 행동 유도 문구 (예: "지금 바로 프로필 링크를 확인하세요!")
                    - 영상 장면에 대한 간단한 설명 (예: #1. 음식이 클로즈업되는 장면)
    
                    **분위기:**
                    [활기찬 / 감성적인 / 유머러스한] 분위기로 작성해주세요.
                    """, language="markdown")

            with blog_tab:
                b_cols = st.columns(2)
                with b_cols[0]:
                    st.link_button(
                        "Gemini 바로가기", 
                        "https://gemini.google.com/",
                        help="강력한 대규모 언어 모델(LLM)을 활용하여 전문적인 블로그 포스트를 손쉽게 작성할 수 있습니다.",
                        use_container_width=True
                    )
                with b_cols[1]:
                    st.link_button(
                        "뤼튼(Wrtn) 블로그", 
                        "https://wrtn.ai/tools/67b2e7901b44a4d864b127a5",
                        help="다양한 글쓰기 도구를 제공하는 한국형 AI 포털입니다. 블로그 포스팅에 특화된 툴을 활용할 수 있습니다.",
                        use_container_width=True
                    )
                
                with st.expander("📝 **블로그 포스팅용 프롬프트 예시 펼쳐보기**"):
                    st.code(f"""
                    ### 블로그 포스트 생성 프롬프트
    
                    **역할:**
                    당신은 '{summary['static_info'].get('HPSN_MCT_BZN_CD_NM')}' 상권의 맛집을 소개하는 전문 블로거입니다.
    
                    **주제:**
                    '{summary['static_info'].get('HPSN_MCT_ZCD_NM')}' 가게 방문 후기
    
                    **타겟 독자:**
                    '{persona['name']}' ({persona['description']})
    
                    **글의 목적:**
                    타겟 독자가 이 글을 읽고 우리 가게에 방문하고 싶게 만드는 것.
                    특히, '{persona['goals'][0]}'와 같은 독자의 목표를 우리 가게가 어떻게 만족시켜주는지 자연스럽게 녹여내 주세요.
    
                    **포함할 내용:**
                    1.  독자의 흥미를 유발하는 제목 (SEO 키워드: [지역명] 맛집, [업종명])
                    2.  가게의 첫인상 및 분위기 묘사
                    3.  주문한 메뉴와 맛에 대한 상세한 설명
                    4.  '{persona['pain_points'][0]}'과 같은 독자의 불편함을 우리 가게가 어떻게 해결해주는지에 대한 포인트 강조
                    5.  가게 위치, 운영 시간, 팁 등 방문 정보
                    6.  독자의 방문을 유도하는 마무리 문단
    
                    **글의 톤앤매너:**
                    [친근하고 솔직한 / 전문적이고 신뢰감 있는] 톤앤매너로 작성해주세요.
                    """, language="markdown")

            with image_tab:
                i_cols = st.columns(3)
                with i_cols[0]:
                    st.link_button(
                        "뤼튼(Wrtn) 이미지", 
                        "https://wrtn.ai/tools/67b2e7901b44a4d864b127b9",
                        help="한국어 프롬프트에 강점을 보이는 AI 포털로, 손쉽게 원하는 이미지를 생성할 수 있습니다.",
                        use_container_width=True
                    )
                with i_cols[1]:
                    st.link_button(
                        "Hailo AI", 
                        "https://hailuoai.video/ko/agent",
                        help="AI 에이전트를 활용하여 다양한 스타일의 이미지를 생성하고 편집할 수 있는 도구입니다.",
                        use_container_width=True
                    )
                with i_cols[2]:
                    st.link_button(
                        "Gemini 이미지", 
                        "https://gemini.google.com/app",
                        help="Google의 Gemini를 통해서도 프롬프트를 입력하여 이미지를 생성할 수 있습니다.",
                        use_container_width=True
                    )

                with st.expander("📝 **이미지 생성 프롬프트 예시 펼쳐보기**"):
                    st.code(f"""
                    ### 마케팅 이미지 생성 프롬프트

                    **스타일:**
                    [실사 사진 / 디지털 아트 / 수채화 / 애니메이션 스타일]

                    **상세 설명:**
                    SNS 광고에 사용할 생동감 있고 매력적인 이미지.
                    '{summary['static_info'].get('HPSN_MCT_ZCD_NM')}' 식당에서 '{persona['name']}' 고객이 만족스럽게 식사를 즐기고 있는 장면.
                    '{persona['goals'][0]}'와 같은 기분을 느끼며 매우 만족스러워 보이는 표정.
                    분위기는 [아늑하고 따뜻한 / 밝고 현대적인 / 활기차고 트렌디한] 느낌.
                    메인 메뉴가 테이블 위에 아름답게 플레이팅 되어 있음.
                    디테일이 뛰어나고 따뜻하며 매력적인 조명에 초점을 맞출 것.

                    **핵심 키워드:**
                    맛있는 음식, 행복한 고객, {summary['static_info'].get('HPSN_MCT_BZN_CD_NM')}, 라이프스타일, 고품질
                    """, language="markdown")

            st.markdown("---")
            st.subheader("🤖 추가 상담하기")
            
            if not st.session_state.chat_messages:
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": "방금 생성된 마케팅 전략에 대해 궁금한 점을 질문해보세요."}
                )

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("여기에 질문을 입력하세요..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("AI가 답변을 생각 중입니다..."):
                        response = generate_chat_response_with_gemini(
                            base_context=f"분석정보: {summary}, 페르소나: {persona}, 원본전략: {st.session_state.marketing_proposal}",
                            messages_history=st.session_state.chat_messages
                        )
                        st.markdown(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            st.markdown("---") 
            
            docx_data = create_docx_report(
                selected_mct,
                st.session_state['marketing_proposal'],
                st.session_state.get('chat_messages', [])
            )
            
            st.download_button(
                label="📄 전체 내용 문서로 저장하기 (.docx)",
                data=docx_data,
                file_name=f"report_{selected_mct}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.info("👆 버튼을 눌러 우리 가게만을 위한 맞춤 마케팅 전략을 확인해보세요!")


if __name__ == '__main__':
    main()