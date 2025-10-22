import requests
import os
from typing import Dict, Any, List

# — API 설정 —
GEMINI_API_KEY = "AIzaSyD18eAdaAvP7FB-Dzp5ZbGNcIln8h-umOc" 
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"


def generate_marketing_text_with_gemini(
    analysis_summary: Dict[str, Any],
    persona_info: Dict[str, Any],
    mbti_result: Dict[str, str],
    mct_id: str
) -> str:
    """Gemini API를 호출하여 페르소나 및 가게 유형 기반 마케팅 제안 텍스트를 생성합니다."""

    if "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY:
        return "### 🚨 API 키 설정 필요\n`gemini_api.py` 파일에서 `GEMINI_API_KEY`를 실제 키로 변경해주세요."

    # 1. 시스템 프롬프트 
    system_prompt = (
        "당신은 대한민국 소상공인을 위한 최고의 마케팅 컨설턴트 AI입니다. "
        "제공된 [가게 유형], [핵심 진단], [핵심 고객 페르소나] 정보를 종합적으로 분석하여, "
        "가게 사장님이 **바로 실행할 수 있는 구체적이고 창의적인 마케팅 액션 플랜**을 제안해야 합니다. "
        "친절하고 이해하기 쉬운 전문가의 말투를 사용해주세요."
    )

    # 2. 사용자 프롬프트 
    user_prompt = f"""
    ### 분석 대상 가맹점: {mct_id}

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

    —
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

    # 3. API Payload 구성 및 호출 (단일 프롬프트)
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


def generate_chat_response_with_gemini(base_context: str, messages_history: List[Dict[str, str]]) -> str:
    """
    AI와 후속 대화를 생성합니다. (REST API 방식)
    base_context: 상점 요약, 페르소나, 원본 전략 등 기본 정보
    messages_history: [{"role": "user", …}, {"role": "assistant", …}] 형식의 리스트
    """
    
    # 1. 시스템 프롬프트
    system_prompt_text = f"""
    당신은 상권 분석 및 마케팅 전문 AI 어시스턴트입니다. 
    사용자는 방금 다음 기본 정보를 바탕으로 마케팅 전략을 생성했습니다:
    —
    [기본 분석 정보 및 원본 전략]
    {base_context}
    —

    이제 사용자가 이 전략에 대해 추가 질문을 하고 있습니다. 
    이어지는 대화 내용을 바탕으로 사용자의 마지막 질문에 친절하고 전문적으로 답변해주세요.
    """
    
    # 2. Streamlit의 대화 기록을 API가 이해할 수 있도록 변환
    api_contents = []
    for msg in messages_history:
        api_role = "model" if msg["role"] == "assistant" else "user"
        api_contents.append({
            "role": api_role,
            "parts": [{"text": msg["content"]}]
        })
    
    # 3. API Payload 구성 (대화 형식)
    payload = {
        "contents": api_contents,
        "systemInstruction": {
            "parts": [{"text": system_prompt_text}]
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
            return f"### 🚨 API 응답 오류\n(챗봇) 응답 형식에 'candidates'가 없습니다.\n\n**응답 내용:**\n```json\n{result}\n```"

    except requests.exceptions.RequestException as e:
        return f"🚨 (챗봇) API 호출 중 네트워크 오류가 발생했습니다: {e}"
    except Exception as e:
        return f"🚨 (챗봇) 응답 처리 중 알 수 없는 오류가 발생했습니다: {e}"
