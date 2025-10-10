import pandas as pd
import numpy as np
import random
from typing import Dict, Any

# 페르소나 생성을 위한 이름 목록
FEMALE_NAMES = ["김지아", "박서연", "이하윤", "최지우", "정민서"]
MALE_NAMES = ["박도윤", "이시우", "김주원", "정은우", "최지호"]

# 페르소나 특성 템플릿
PERSONA_TEMPLATES = {
    '직장인': {
        'goals': ["점심시간에 동료들과 맛있는 식사를 하며 스트레스를 풀고 싶어해요.", 
                  "빠르고 간편하지만 건강도 챙길 수 있는 점심 메뉴를 찾고 있어요.",
                  "퇴근 후 간단하게 동료들과 한잔하며 하루를 마무리할 곳이 필요해요."],
        'pain_points': ["매일 비슷한 점심 메뉴가 지겨워요.", 
                        "유명 맛집은 웨이팅이 너무 길어서 점심시간이 부족해요.",
                        "가성비와 맛을 모두 만족시키는 곳을 찾기 어려워요."],
        'channels': ["네이버 지도", "인스타그램", "직장인 커뮤니티 앱(블라인드 등)"]
    },
    '거주자': {
        'goals': ["주말에 가족과 함께 외식할 만한 편안한 장소를 찾고 있어요.", 
                  "집 근처에서 혼자 또는 친구와 부담 없이 식사할 곳을 원해요.",
                  "동네의 숨은 맛집을 발견하고 단골이 되고 싶어해요."],
        'pain_points': ["배달 음식은 질리고, 직접 가서 먹고 싶을 때가 많아요.",
                        "아이들을 데리고 갈 만한 식당이 마땅치 않아요.",
                        "매번 가던 곳만 가게 되어 새로운 가게를 시도하기가 망설여져요."],
        'channels': ["지역 맘카페", "당근마켓", "배달 앱(포장/방문 기능 활용)"]
    },
    '유동인구': {
        'goals': ["약속 장소로 이동하기 전에 빠르고 간단하게 끼니를 해결하고 싶어요.",
                  "이 지역의 특색있는 음식을 경험해보고 싶어요.",
                  "SNS에 올릴 만한 예쁘고 맛있는 디저트나 음식을 찾고 있어요."],
        'pain_points': ["어떤 가게가 맛집인지 알기 어려워 정보 탐색에 시간이 걸려요.",
                        "혼자 들어가서 먹기에는 부담스러운 가게들이 많아요.",
                        "유명 관광지는 너무 비싸고 불친절한 경험을 할 때가 있어요."],
        'channels': ["인스타그램(해시태그 검색)", "망고플레이트/다이닝코드", "유튜브(지역 맛집 VLOG)"]
    }
}


def create_persona(merchant_row: pd.Series, analysis_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    데이터 분석 결과를 바탕으로 가상의 고객 페르소나를 생성합니다.
    """
    # 1. 페르소나 기본 정보 설정 (주 고객층 기반)
    dominant_ag_group = analysis_summary['dominant_ag_group']
    primary_cust_type = analysis_summary['primary_cust_type']
    
    gender = '여성' if '여성' in dominant_ag_group else '남성'
    age_group = dominant_ag_group.replace('남성 ', '').replace('여성 ', '')
    
    # 이름 랜덤 선택
    name = random.choice(FEMALE_NAMES) if gender == '여성' else random.choice(MALE_NAMES)

    # 2. 페르소나 유형 및 직업 추정 (상권, 업종, 고객 유형 기반)
    persona_type = '직장인' # 기본값
    if primary_cust_type == '거주':
        persona_type = '거주자'
    elif primary_cust_type == '유동인구':
        persona_type = '유동인구'
        
    job = f"{analysis_summary['static_info']['HPSN_MCT_BZN_CD_NM']} 인근의 {persona_type}"
    
    # 3. 페르소나의 목표(Goals)와 어려움(Pain Points) 설정
    template = PERSONA_TEMPLATES.get(persona_type, PERSONA_TEMPLATES['직장인'])
    
    # 재방문율에 따라 Pain Point 추가
    pain_points = random.sample(template['pain_points'], 2)
    if '재방문 고객 비중이 낮아' in analysis_summary['retention_analysis_text']:
        pain_points.append("마음에 드는 가게를 찾으면 정착하고 싶지만, 아직 단골이 될 만큼 만족스러운 곳을 발견하지 못했어요.")

    # 4. 페르소나 종합 정보 생성
    persona_description = (
        f"{name}님은 **{age_group} {gender}**으로, "
        f"주로 **'{analysis_summary['static_info']['HPSN_MCT_BZN_CD_NM']}'** 상권에서 활동하는 **{persona_type}**입니다. "
        f"가게의 전체 고객 중 **{analysis_summary['dominant_ag_ratio']:.1f}%**를 차지하는 핵심 고객 그룹을 대표합니다."
    )
    
    return {
        'name': f"{name} ({age_group} {gender})",
        'description': persona_description,
        'goals': random.sample(template['goals'], 2),
        'pain_points': pain_points,
        'channels': template['channels']
    }