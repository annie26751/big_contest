import pandas as pd
from typing import Dict


# --- 가게 유형 정의 ---
STORE_TYPE_DEFINITIONS = {
    "동네 사랑방": "안정적인 단골 고객층을 기반으로 꾸준히 운영되는 동네 터주대감",
    "오피스 핫플": "주변 직장인들의 점심/저녁을 책임지는, 피크 타임이 확실한 가게",
    "숨은 맛집": "우연히 들른 손님도 단골로 만드는, 맛과 매력으로 승부하는 가게",
    "배달의 고수": "홀보다는 배달/포장에 집중하여 높은 회전율과 효율성을 자랑하는 가게",
    "반짝 스타": "SNS나 미디어의 힘으로 급부상한, 신규 고객을 단골로 만드는 것이 과제인 가게",
    "성장 꿈나무": "이제 막 시작하여 가능성을 보여주는, 잠재력이 기대되는 신생 가게",
    "고독한 미식가": "소수의 매니아층을 확실하게 사로잡은, 객단가가 높은 전문점",
    "가격 파괴자": "뛰어난 가성비를 무기로 박리다매 전략을 통해 많은 고객을 유치하는 가게",
    "상권의 지배자": "맛, 위치, 고객 관리 모든 면에서 압도적인 성과를 보이는 동네 1등 가게",
    "위기의 소상공인": "전반적인 지표 개선이 시급한, 변화와 혁신이 필요한 가게"
}


# --- 가게 유형 분류 함수 ---
def classify_merchant_mbti(merchant_row: pd.Series) -> Dict[str, str]:
    """
    가맹점의 데이터 지표를 기반으로 10가지 유형 중 하나로 분류합니다.
    """
    # --- 데이터 지표 추출 ---
    ry_rank = merchant_row.get('M12_SME_RY_SAA_PCE_RT', 100)
    bzn_rank = merchant_row.get('M12_SME_BZN_SAA_PCE_RT', 100)
    repeat_rate = merchant_row.get('MCT_UE_CLN_REU_RAT', 0)
    new_rate = merchant_row.get('MCT_UE_CLN_NEW_RAT', 0)
    delivery_rate = merchant_row.get('DLV_SAA_RAT', 0)
    cust_count_rank = merchant_row.get('RC_M1_UE_CUS_CN', '90%초과')
    avg_spend_rank = merchant_row.get('RC_M1_AV_NP_AT', '90%초과')

    cust_type_ratios = {
        '거주': merchant_row.get('RC_M1_SHC_RSD_UE_CLN_RAT', 0),
        '직장': merchant_row.get('RC_M1_SHC_WP_UE_CLN_RAT', 0),
        '유동': merchant_row.get('RC_M1_SHC_FLP_UE_CLN_RAT', 0)
    }
    main_cust_type = max(cust_type_ratios, key=cust_type_ratios.get)
    store_type_name = "위기의 소상공인"  

    # --- 분류 로직 (규칙 기반) ---
    if ry_rank < 30 and bzn_rank < 30:
        store_type_name = "상권의 지배자"
    elif ry_rank > 80 and bzn_rank > 80 and ('90%초과' in cust_count_rank or '75-90%' in cust_count_rank):
        store_type_name = "위기의 소상공인"
    elif delivery_rate > 50:
        store_type_name = "배달의 고수"
    elif new_rate > 60 and repeat_rate < 30:
        store_type_name = "반짝 스타"
    elif '하위' in avg_spend_rank and '상위' in cust_count_rank:
         store_type_name = "가격 파괴자"
    elif '상위' in avg_spend_rank and '하위' in cust_count_rank:
         store_type_name = "고독한 미식가"
    elif main_cust_type == '직장' and ry_rank < 50:
        store_type_name = "오피스 핫플"
    elif main_cust_type == '거주' and repeat_rate > 50:
        store_type_name = "동네 사랑방"
    elif main_cust_type == '유동' and repeat_rate > 40 and bzn_rank > 50:
        store_type_name = "숨은 맛집"
    elif merchant_row.get('MCT_OPE_MS_CN') in ['75-90%', '90%초과']:
        store_type_name = "성장 꿈나무"

    return {
        "name": store_type_name,
        "description": STORE_TYPE_DEFINITIONS[store_type_name]
    }