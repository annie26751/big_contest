# persona_generator.py

import pandas as pd
import numpy as np
import random
from typing import Dict, Any

FEMALE_NAMES = ["김지아", "박서연", "이하윤", "최지우", "정민서"]
MALE_NAMES = ["박도윤", "이시우", "김주원", "정은우", "최지호"]

PERSONA_TEMPLATES = {
    # ------------------------------------
    # 1. 오피스 상권 (직장인)
    # ------------------------------------
    '직장인_2030_가성비': {
        'icon': "💼",  
        'roles': ["주니어 마케터", "신입 개발자", "스타트업 직원", "신입사원"],
        'goals': ["점심시간에 밖에서 먹으며 잠시 스트레스를 풀고 싶어요.", 
                  "빠르고 가성비 좋은 점심 메뉴를 동료들과 찾고 있어요."],
        'pain_points': ["매일 탕/찌개가 지겨워요.", 
                        "점심값이 너무 부담돼요.",
                        "웨이팅이 긴 곳은 절대 갈 수 없어요.",
                        "회사와 너무 먼 곳은 부담돼요"],
        'channels': ["직장동료 추천", "네이버 지도 (가격순)", "직장인 익명 커뮤니티 (맛집)", "인스타그램"]
    },
    '직장인_4050_프리미엄': {
        'icon': "👔",  
        'roles': ["부장/이사급 관리자", "전문직 (변호사, 회계사)", "영업직"],
        'goals': ["중요한 미팅이나 손님 접대를 위한 품격 있는 장소를 원해요.",
                  "퇴근 후 깔끔한 분위기에서 격식 없이 술 한 잔 할 곳이 필요해요."],
        'pain_points': ["시끄럽거나 캐주얼한 분위기는 불편해요.",
                        "주차 공간이 확보되지 않은 곳은 방문하기 어려워요.",
                        "대충 만든 듯한 점심 메뉴는 돈이 아까워요."],
        'channels': ["지인 추천", "네이버 예약"]
    },
    # ------------------------------------
    # 2. 주거 상권 (거주자)
    # ------------------------------------
    '거주자_가족_패밀리': {
        'icon': "👨‍👩‍👧‍👦",  
        'roles': ["육아맘", "주부","가장"],
        'goals': ["아이들과 함께 편안하게 식사할 수 있는 공간을 찾고 있어요.", 
                  "외식으로 집밥 같은 건강하고 맛있는 한 끼를 해결하고 싶어요.",
                  "가족 다같이 밖에서 식사를 하고싶어요"],
        'pain_points': ["아이 메뉴가 없으면 난감해요.",
                        "다른 테이블에 피해를 줄까 봐 조용하지 않은 곳은 꺼려져요.",
                        "주말 저녁은 예약하지 않으면 앉을 곳이 없어요."],
        'channels': ["지역 맘카페", "당근마켓 (동네생활)", "배달 앱 (포장 주문)"]
    },
    '거주자_1인가구_혼밥족': {
        'icon': "👤",  
        'roles': ["대학생", "싱글 직장인", "자취생", "프리랜서"],
        'goals': ["집 근처에서 혼자 부담 없이 간단하게 끼니를 때우고 싶어요.",
                  "주말에 소소하게 힐링할 수 있는 '나만의 아지트' 같은 공간을 찾아요."],
        'pain_points': ["혼자 방문하기 어색한 분위기의 식당이 너무 많아요.",
                        "양이 너무 많아서 남기게 될까 봐 걱정돼요.",
                        "배달비가 비싸서 포장/방문을 이용하고 싶어요."],
        'channels': ["네이버 지도 (내 주변)", "인스타그램 (혼밥 해시태그)", "유튜브 (자취 VLOG,먹방)"]
    },
    # ------------------------------------
    # 3. 기타/유동인구 (탐색)
    # ------------------------------------
    '유동인구_관광_탐색형': {
        'icon': "👥", 
        'roles': ["여행객", "타지 방문객", "데이트 커플"],
        'goals': ["이 지역에서 꼭 먹어야 하는 '시그니처' 메뉴를 맛보고 싶어요.",
                  "사진 찍기 좋은 예쁜 장소에서 경험을 공유하고 싶어요."],
        'pain_points': ["정보가 너무 많아서 맛집을 고르기 어려워요.",
                        "유명한 곳은 대기가 너무 길어 시간을 낭비해요.",
                        "SNS 후기만 믿고 갔다가 실망한 경험이 많아요."],
        'channels': ["인스타그램 (지역명 해시태그)", "블로그 (최신 후기)", "캐치테이블", "네이버 플레이스", "카카오맵 리뷰"]
    },
    '유동인구_대중교통_환승객': {
        'icon': "🚆", 
        'roles': ["출퇴근 환승객", "약속 전 시간 때우는 사람", "급한 일정의 비즈니스맨"],
        'goals': ["약속 장소로 이동하기 전에 빠르고 간단하게 식사를 해결하고 싶어요.",
                  "대기 없이 바로 이용 가능한 카페나 식당이 필요해요."],
        'pain_points': ["역 근처는 너무 붐비고 정신이 없어요.",
                        "커피 한 잔만 시키기 눈치 보이는 곳은 싫어요.",
                        "화장실이 깨끗하지 않은 곳은 불편해요."],
        'channels': ["네이버 지도 (현재 위치, 빠른 검색)", "카카오맵", "대중교통 커뮤니티"]
    }
}

def create_persona(merchant_row: pd.Series, analysis_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    데이터 분석 결과를 바탕으로 가상의 고객 페르소나를 생성합니다.
    """
    # 1. 분석 결과 추출
    info = analysis_summary['static_info']
    dominant_ag_group = analysis_summary['dominant_ag_group']
    primary_cust_type = analysis_summary['primary_cust_type']    
    
    gender = '여성' if '여성' in dominant_ag_group else '남성'
    age_group = dominant_ag_group.replace('남성 ', '').replace('여성 ', '')
    age_code = age_group.split('대')[0] 
    
    name = random.choice(FEMALE_NAMES) if gender == '여성' else random.choice(MALE_NAMES)

    # 2. 가게 특징 파악
    avg_price = analysis_summary.get('RC_M1_AV_NP_AT', 0)
    is_premium = avg_price > analysis_summary.get('RC_M1_BZN_AV_NP_AT', 0) * 1.2
    
    # 3. 페르소나 유형 결정 로직
    persona_key = ""

    if primary_cust_type == '직장':
        if age_code in ['20', '30'] or not is_premium:
            persona_key = '직장인_2030_가성비'
        elif age_code in ['40', '50'] and is_premium:
            persona_key = '직장인_4050_프리미엄'
        else:
            persona_key = '직장인_2030_가성비'
            
    elif primary_cust_type == '거주':
        if info.get('HPSN_MCT_BZN_CD_NM') in ['주택가']:
            if age_code in ['40', '50'] and gender == '여성':
                persona_key = '거주자_가족_패밀리'
            elif age_code in ['20', '30'] and analysis_summary.get('MCT_UE_CLN_NEW_RAT', 0) > 0.6:
                persona_key = '거주자_1인가구_혼밥족'
            else:
                persona_key = '거주자_가족_패밀리'
        else: # 주택가가 아니면 1인 가구로 기본 설정
            persona_key = '거주자_1인가구_혼밥족'
                
    elif primary_cust_type == '유동인구':
        if info.get('HPSN_MCT_BZN_CD_NM') in ['관광특구', '명소', '복합단지']:
            persona_key = '유동인구_관광_탐색형'
        elif info.get('HPSN_MCT_BZN_CD_NM') in ['역세권']:
            persona_key = '유동인구_대중교통_환승객'
        else:
            persona_key = '유동인구_관광_탐색형'
    
    # 4. 템플릿 로드 및 구체화
    if not persona_key:
        persona_key = '직장인_2030_가성비' 
    template = PERSONA_TEMPLATES[persona_key]
    
    job = random.choice(template['roles'])
    
    # 5. 페르소나의 목표(Goals)와 어려움(Pain Points) 설정
    pain_points = random.sample(template['pain_points'], 2)
    
    if '재방문 고객 비중이 낮아' in analysis_summary.get('retention_analysis_text', ''):
        pain_points.append("마음에 드는 가게를 찾으면 정착하고 싶지만, 아직 단골이 될 만큼 만족스러운 곳을 발견하지 못했어요.")

    # 6. 페르소나 종합 정보 생성
    persona_description = (
    f"<b>{name}</b> 님은 <b>{age_group} {gender}</b> 으로, 직업은 <b>{job}</b> 입니다. "
    f"주로 '<b>{info.get('h_name', info.get('HPSN_MCT_BZN_CD_NM'))}</b>' 상권에서 활동하며, "
    f"가게의 전체 고객 중 <b>{analysis_summary['dominant_ag_ratio']:.1f}%</b> 를 차지하는 핵심 고객 그룹을 대표합니다."
)
    
    return {
        'icon': template['icon'],
        'name': f"{name} ({job} / {age_group} {gender})",
        'description': persona_description,
        'goals': random.sample(template['goals'], 2),
        'pain_points': pain_points,
        'channels': template['channels']
    }
