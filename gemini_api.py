import requests
import os # API 키를 안전하게 관리하기 위해 os 모듈을 import 합니다.
from typing import Dict, Any

# ==============================================================================
GEMINI_API_KEY = "AIzaSyCm9d2tg5Gout-f6NAPXw4zy0M9iGwqLbc"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
# QUARTILE_MAPPING은 data_processor에서도 사용되지만, 프롬프트 생성 시 필요하여 여기서도 정의합니다.
QUARTILE_MAPPING = {
    '10%이하': '상위 10%', '10-25%': '상위 10-25%', '25-50%': '중위 25-50%',
    '50-75%': '하위 50-75%', '75-90%': '하위 75-90%', '90%초과': '하위 90% 초과'
}

# [수정] mbti_result 인자를 추가하여 4개의 인자를 받도록 함수 정의를 변경합니다.
def generate_marketing_text_with_gemini(
    analysis_summary: Dict[str, Any],
    persona_info: Dict[str, Any],
    mbti_result: Dict[str, str],
    mct_id: str
):
    """Gemini API를 호출하여 페르소나 및 가게 유형 기반 마케팅 제안 텍스트를 생성합니다."""

    if "YOUR_API_KEY_HERE" in GEMINI_API_KEY:
        return "### 🚨 API 키 설정 필요\n.env 파일에 Gemini API 키를 설정해주세요."

    # 1. 시스템 프롬프트 (AI의 역할 정의)
    system_prompt = (
        "당신은 대한민국 소상공인을 위한 최고의 마케팅 컨설턴트 AI입니다. "
        "제공된 [가게 유형], [핵심 진단], [핵심 고객 페르소나] 정보를 종합적으로 분석하여, "
        "가게 사장님이 **바로 실행할 수 있는 구체적이고 창의적인 마케-팅 액션 플랜**을 제안해야 합니다. "
        "친절하고 이해하기 쉬운 전문가의 말투를 사용해주세요."
    )

    # 2. 사용자 프롬프트 (AI에게 전달할 데이터)
    user_prompt = f"""
    ###  분석 대상 가맹점: {mct_id}

    #### [가게 유형 분석]
    - **우리 가게 유형:** {mbti_result['name']} ({mbti_result['description']})

    #### [핵심 진단 결과]
    - **고객층:** {analysis_summary['cust_analysis_text']}
    - **고객 유지력:** {analysis_summary['retention_analysis_text']}
    - **경쟁 환경:** {analysis_summary['comp_analysis_text']}

    #### [핵심 고객 페르소나]
    - **이름:** {persona_info['name']}
    - **특징:** {persona_info['description']}
    - **찾는 이유(Goals):** {', '.join(persona_info['goals'])}
    - **어려움(Pain Points):** {', '.join(persona_info['pain_points'])}

    ---
    ### [요청 사항]
    위 모든 정보를 바탕으로, 이 가게의 **강점은 극대화**하고 **약점은 보완**할 수 있는 맞춤형 마케팅 전략을 아래 형식에 맞춰 제안해주세요.

    **1. 한 줄 요약:** (우리 가게의 현재 상황과 나아갈 방향을 한 문장으로 요약)
    **2. 데이터 기반 강점 및 약점 진단:** (데이터를 근거로 어떤 점이 강하고 약한지 분석)
    **3. 맞춤형 마케팅 액션 플랜 (3가지):**
        - **전략명:** (예: '점심시간 단골 확보를 위한 타임어택 이벤트')
        - **데이터 근거:** (이 전략을 왜 제안하는지 데이터에 기반하여 설명)
        - **실행 방법:** (사장님이 따라 할 수 있도록 구체적인 실행 방법 제시)
        - **홍보 문구 예시:** (고객 페르소나의 눈길을 사로잡을 SNS 또는 문자 메시지 예시)
    """

    # 3. API Payload 구성 및 호출
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        }
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if 'candidates' in result and result['candidates']:
            text = result['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '오류: 응답 내용이 비어있습니다.')
            return text
        else:
            return f"### 🚨 API 응답 오류\n응답 형식에 'candidates'가 없습니다. API 키와 모델명을 확인해주세요.\n\n**응답 내용:**\n```json\n{result}\n```"

    except requests.exceptions.RequestException as e:
        return f"🚨 API 호출 중 네트워크 오류가 발생했습니다: {e}"
    except Exception as e:
        return f"🚨 응답 처리 중 알 수 없는 오류가 발생했습니다: {e}"

