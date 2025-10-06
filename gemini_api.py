# gemini_api.py
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any

# ==============================================================================
# --- 🌟 Gemini API Configuration ---
# ==============================================================================
GEMINI_API_KEY = "your_gemini_api_key" 
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# QUARTILE_MAPPING은 분석 결과 요약에 필요하므로 여기서 다시 정의하거나 data_processor에서 import합니다.
QUARTILE_MAPPING = {
    '10%이하': '상위 10%', '10-25%': '상위 10-25%', '25-50%': '중위 25-50%', 
    '50-75%': '하위 50-75%', '75-90%': '하위 75-90%', '90%초과': '하위 90% 초과'
}

def generate_marketing_text_with_gemini(analysis_data: Dict[str, Any], mct_id: str):
    """Gemini API를 호출하여 분석 기반 마케팅 제안 텍스트를 생성합니다."""
    
    # 🚨 키 미설정 에러 처리
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return "🚨 Gemini API 키가 설정되지 않았습니다. API 키를 실제 키로 교체하세요."

    # 1. 시스템 지침 설정
    system_prompt = (
        "당신은 외식업 컨설팅 전문가입니다. 제공된 3가지 분석 결과(고객층, 재방문율, 경쟁 환경)를 바탕으로 가맹점의 강점과 약점을 진단하고, "
        "구체적이고 실현 가능한 3~5가지의 마케팅 전략을 간결하고 전문적인 한국어로 Markdown 포맷으로 작성해 주세요. "
        "결과를 `## 🎯 핵심 전략` 제목 아래에 배치하고, 각 전략은 번호를 붙여주세요."
    )

    # 2. 사용자 쿼리 (분석 데이터 요약) 생성
    summary = f"""
    가맹점 ID: {mct_id}
    [1] 기본 정보: 업종={analysis_data['static_info'].get('HPSN_MCT_ZCD_NM')}, 상권={analysis_data['static_info'].get('HPSN_MCT_BZN_CD_NM')}, 개설일={analysis_data['static_info'].get('ARE_D')}
    
    [2] 3가지 핵심 진단 결과:
        - 가. 고객층 분석: {analysis_data['cust_analysis_text']}
        - 나. 재방문율 확인: {analysis_data['retention_analysis_text']}
        - 다. 경쟁 환경 내 위치 파악: {analysis_data['comp_analysis_text']}
        
    [3] 핵심 운영 지표 (최빈값/월평균):
        - 매출 구간: {QUARTILE_MAPPING.get(analysis_data['metric_info'].get('RC_M1_SAA'), '정보 없음')}
        - 객단가 구간: {QUARTILE_MAPPING.get(analysis_data['metric_info'].get('RC_M1_AV_NP_AT'), '정보 없음')}
    
    위 정보를 종합적으로 분석하여 이 가맹점에 특화된 마케팅 전략을 제안하세요.
    """
    
    # 3. API Payload 구성 및 호출
    payload = {
        "contents": [{"parts": [{"text": summary}]}],
        "config": {"systemInstruction": system_prompt}
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() 
        result = response.json()
        
        # ⚠️ 응답 구조 변경: systemInstruction을 config로 넣을 경우 구조가 달라집니다.
        # 기존 코드의 응답 처리 로직을 유지하기 위해 'systemInstruction'을 payload 루트에 넣는 대신,
        # 'contents'와 함께 'config'를 루트에 넣는 형태로 변경합니다.
        
        # 최신 API 구조에 맞춰 generateContent 응답 파싱
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Gemini로부터 응답을 받지 못했습니다.')
        return text

    except requests.exceptions.RequestException as e:
        return f"🚨 API 호출 오류 발생: {e}"
    except Exception as e:
        # st.json(result) # 디버깅용
        return f"🚨 응답 처리 오류: {e}"
