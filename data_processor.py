import pandas as pd
import numpy as np
import os
from collections import Counter
from typing import Dict, Any
from persona_generator import create_persona
from mbti_classifier import classify_merchant_mbti

FIXED_DATA_PATH = "./data/merged_data.csv"
STATIC_COLS = [
    'MCT_BSE_AR', 'MCT_SIGUNGU_NM', 'HPSN_MCT_ZCD_NM', 'HPSN_MCT_BZN_CD_NM',
    'ARE_D', 'MCT_ME_D'
]
QUARTILE_METRIC_COLS = [
    'MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT',
    'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT'
]
QUARTILE_MAPPING = {
    '10%이하': '상위 10%', '10-25%': '상위 10-25%', '25-50%': '중위 25-50%',
    '50-75%': '하위 50-75%', '75-90%': '하위 75-90%', '90%초과': '하위 90% 초과'
}
AGE_GENDER_COLS = [
    'M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT', 'M12_MAL_50_RAT', 'M12_MAL_60_RAT',
    'M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT', 'M12_FME_50_RAT', 'M12_FME_60_RAT'
]
AGE_GENDER_NAMES = {
    'M12_MAL_1020_RAT': '남성 20대이하', 'M12_MAL_30_RAT': '남성 30대', 'M12_MAL_40_RAT': '남성 40대',
    'M12_MAL_50_RAT': '남성 50대', 'M12_MAL_60_RAT': '남성 60대이상',
    'M12_FME_1020_RAT': '여성 20대이하', 'M12_FME_30_RAT': '여성 30대', 'M12_FME_40_RAT': '여성 40대',
    'M12_FME_50_RAT': '여성 50대', 'M12_FME_60_RAT': '여성 60대이상'
}
CUST_TYPE_COLS = [
    'RC_M1_SHC_RSD_UE_CLN_RAT', 'RC_M1_SHC_WP_UE_CLN_RAT', 'RC_M1_SHC_FLP_UE_CLN_RAT'
]
CUST_TYPE_NAMES = {
    'RC_M1_SHC_RSD_UE_CLN_RAT': '거주 이용 고객',
    'RC_M1_SHC_WP_UE_CLN_RAT': '직장 이용 고객',
    'RC_M1_SHC_FLP_UE_CLN_RAT': '유동인구 이용 고객'
}
MEAN_COLS_FOR_AGG = [
    'DLV_SAA_RAT', 'M12_SME_RY_SAA_PCE_RT', 'M12_SME_BZN_SAA_PCE_RT',
] + AGE_GENDER_COLS + CUST_TYPE_COLS + ['MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT']
SV_VALUE = -999999.9

def get_mode_or_first(series):
    counts = Counter(series.dropna())
    if not counts: return None
    return counts.most_common(1)[0][0]

def preprocess_data(df: pd.DataFrame):
    df[MEAN_COLS_FOR_AGG] = df[MEAN_COLS_FOR_AGG].apply(pd.to_numeric, errors='coerce')
    df = df.replace(SV_VALUE, np.nan)
    df_static = df.groupby('ENCODED_MCT')[STATIC_COLS].first().reset_index()
    df_avg = df.groupby('ENCODED_MCT')[MEAN_COLS_FOR_AGG].mean().reset_index()
    agg_quartile_funcs = {col: get_mode_or_first for col in QUARTILE_METRIC_COLS}
    df_quartile = df.groupby('ENCODED_MCT').agg(agg_quartile_funcs).reset_index()
    df_final = pd.merge(df_static, df_avg, on='ENCODED_MCT', how='left')
    df_final = pd.merge(df_final, df_quartile, on='ENCODED_MCT', how='left')
    return df_final

def load_fixed_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 파일 없음: `{path}`")
    df_raw = pd.read_csv(path, encoding='cp949')
    return preprocess_data(df_raw)


# --- 가맹점 통합 분석 함수 ---
def analyze_merchant(merchant_row: pd.Series) -> Dict[str, Any]:
    """선택된 가맹점의 모든 분석(진단, 페르소나, MBTI)을 수행합니다."""

    # --- 1. 3가지 핵심 진단 ---
    age_gender_ratios = merchant_row[AGE_GENDER_COLS].dropna()
    max_ag_ratio = age_gender_ratios.max() if not age_gender_ratios.empty else 0.0
    dominant_ag_name = AGE_GENDER_NAMES.get(age_gender_ratios.idxmax(), '정보 없음') if not age_gender_ratios.empty else '정보 없음'
    primary_ct_ratios = merchant_row[CUST_TYPE_COLS].dropna()
    primary_ct_name = CUST_TYPE_NAMES.get(primary_ct_ratios.idxmax(), '정보 없음').replace(' 이용 고객', '') if not primary_ct_ratios.empty else '정보 없음'
    cust_analysis_text = f"우리 가게는 **{dominant_ag_name} {primary_ct_name}** 고객 비중이 가장 높습니다."

    repeat_rate = merchant_row.get('MCT_UE_CLN_REU_RAT', np.nan)
    retention_analysis_text = (f"**재방문율({repeat_rate:.1f}%)이 양호**합니다." if repeat_rate > 30 else f"**재방문율({repeat_rate:.1f}%)이 낮아** 단골 확보가 필요합니다.") if pd.notna(repeat_rate) else "재방문율 정보 부족"

    ry_rank = merchant_row.get('M12_SME_RY_SAA_PCE_RT', np.nan)
    comp_analysis_text = (f"업종 내 **상위 {(100 - ry_rank):.1f}%**의 준수한 경쟁력입니다." if ry_rank < 70 else f"업종 내 **하위 {(100-ry_rank):.1f}%**로 경쟁력 강화가 시급합니다.") if pd.notna(ry_rank) else "경쟁 환경 정보 부족"

    summary_data = {
        'cust_analysis_text': cust_analysis_text,
        'retention_analysis_text': retention_analysis_text,
        'comp_analysis_text': comp_analysis_text,
        'static_info': merchant_row[STATIC_COLS].to_dict(),
        'metric_info': merchant_row[QUARTILE_METRIC_COLS].to_dict(),
        'dominant_ag_group': dominant_ag_name,
        'primary_cust_type': primary_ct_name,
        'dominant_ag_ratio': max_ag_ratio
    }

    # --- 2. 페르소나 및 가게 유형 분석 ---
    persona_info = create_persona(merchant_row, summary_data)
    mbti_info = classify_merchant_mbti(merchant_row)

    # --- 3. 최종 결과 종합 ---
    return {
        'summary': summary_data,
        'persona': persona_info,
        'mbti': mbti_info,
        'raw_data': merchant_row
    }