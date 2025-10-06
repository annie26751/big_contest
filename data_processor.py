# data_processor.py
import pandas as pd
import numpy as np
import os
from collections import Counter
from typing import Dict, Any

# ==============================================================================
# --- 데이터 파일 경로 설정 및 상수 정의 ---
# ==============================================================================
FIXED_DATA_PATH = "./data/merged_data.csv" 

# --- 분석에 필요한 상수 정의 ---
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

RETENTION_COLS = ['MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT']
COMPETITION_COLS = ['M12_SME_RY_SAA_PCE_RT', 'M12_SME_BZN_SAA_PCE_RT']

# 평균을 계산할 모든 숫자형 컬럼 목록
MEAN_COLS_FOR_AGG = [
    'DLV_SAA_RAT', 'M1_SME_RY_SAA_RAT', 'M1_SME_RY_CNT_RAT',
    'M12_SME_RY_SAA_PCE_RT', 'M12_SME_BZN_SAA_PCE_RT',
] + AGE_GENDER_COLS + CUST_TYPE_COLS + RETENTION_COLS

ALL_METRIC_COLS = QUARTILE_METRIC_COLS + MEAN_COLS_FOR_AGG
SV_VALUE = -999999.9


# --- 데이터 처리 함수 ---

def get_mode_or_first(series):
    """시리즈에서 가장 빈도가 높은 값(최빈값)을 반환합니다."""
    counts = Counter(series.dropna())
    if not counts:
        return None
    return counts.most_common(1)[0][0]

def preprocess_data(df: pd.DataFrame):
    """로드된 통합 데이터프레임을 전처리하고 가맹점별 정보를 집계합니다."""
    
    required_cols = ['ENCODED_MCT', 'TA_YM'] + STATIC_COLS + ALL_METRIC_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ 데이터 파일에 다음 필수 컬럼이 누락되었습니다: {', '.join(missing_cols)}")
        
    numeric_cols = MEAN_COLS_FOR_AGG 
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.replace(SV_VALUE, np.nan)
    
    # 1. 정적 정보 (Data 1) - 첫 번째 값
    df_static = df.groupby('ENCODED_MCT')[STATIC_COLS].first().reset_index()
    
    # 2. 숫자 지표 (Data 2, 3) - 평균
    df_avg = df.groupby('ENCODED_MCT')[MEAN_COLS_FOR_AGG].mean().reset_index()

    # 3. 구간 지표 (Data 2) - 최빈값 (Mode)
    agg_quartile_funcs = {col: get_mode_or_first for col in QUARTILE_METRIC_COLS}
    df_quartile = df.groupby('ENCODED_MCT').agg(agg_quartile_funcs).reset_index()

    # 4. 최종 병합
    df_final = df_static.merge(df_avg, on='ENCODED_MCT')
    df_final = df_final.merge(df_quartile, on='ENCODED_MCT')
    
    return df_final


def load_fixed_data(path):
    """지정된 경로에서 데이터를 로드하고 전처리합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 오류: 지정된 경로에 파일이 존재하지 않습니다. 경로를 확인하세요: `{path}`")
        
    try:
        df_raw = pd.read_csv(path, encoding='cp949')
        df_profile = preprocess_data(df_raw)
        return df_profile
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"❌ 파일 인코딩 오류! 지정된 파일은 `cp949` 인코딩이어야 합니다. 파일 `{path}`를 확인해주세요.")
    except Exception as e:
        raise Exception(f"❌ 파일을 로드하거나 전처리하는 중 오류가 발생했습니다. 에러: {e}")


def analyze_merchant(merchant_row: pd.Series) -> Dict[str, Any]:
    """선택된 가맹점의 3가지 분석을 수행하고 결과를 딕셔너리로 반환합니다."""
    
    # --- A. 고객층 분석 (타겟 고객 파악) ---
    age_gender_ratios = merchant_row[AGE_GENDER_COLS].dropna()
    max_ag_ratio = age_gender_ratios.max() if not age_gender_ratios.empty else 0.0
    
    dominant_ag_name = AGE_GENDER_NAMES.get(age_gender_ratios.idxmax(), '정보 없음') if not age_gender_ratios.empty else '정보 없음'
    primary_ct_ratios = merchant_row[CUST_TYPE_COLS].dropna()
    
    if not primary_ct_ratios.empty:
        primary_ct_name = CUST_TYPE_NAMES.get(primary_ct_ratios.idxmax(), '정보 없음').replace(' 이용 고객', '')
        primary_ct_ratio = primary_ct_ratios.max()
    else:
        primary_ct_name = '정보 없음'
        primary_ct_ratio = 0.0
        
    cust_analysis_text = f"AI 분석 결과, 우리 가게는 **{dominant_ag_name} {primary_ct_name}**가 전체 고객의 **{max_ag_ratio:.1f}%**를 차지하여 고객층이 {'특정 그룹에 집중되어 있습니다' if max_ag_ratio >= 50.0 else '비교적 다양하게 분포되어 있습니다'}."


    # --- B. 재방문율 확인 ---
    repeat_rate = merchant_row.get('MCT_UE_CLN_REU_RAT', np.nan)
    new_rate = merchant_row.get('MCT_UE_CLN_NEW_RAT', np.nan)
    
    retention_analysis_text = "재방문율 및 신규 고객 비중 정보가 불충분합니다."
    if not pd.isna(repeat_rate) and not pd.isna(new_rate):
        if repeat_rate <= 30.0:
            if new_rate > 50.0:
                retention_analysis_text = f"신규 고객은 꾸준히 유입({new_rate:.1f}%)되고 있으나, **재방문 고객 비중이 {repeat_rate:.1f}%로 낮아** 한번 방문한 고객을 단골로 만드는 데 어려움을 겪고 있습니다."
            else:
                retention_analysis_text = f"**재방문율이 {repeat_rate:.1f}%로 낮고**, 신규 고객 유입({new_rate:.1f}%)도 활발하지 않아 전반적인 고객 확보에 어려움이 있습니다."
        else:
            retention_analysis_text = f"재방문 고객 비중이 {repeat_rate:.1f}%로 **양호한 수준**이며, 고객 유지가 잘 이루어지고 있습니다. 신규 고객 비중은 {new_rate:.1f}% 입니다."


    # --- C. 경쟁 환경 내 위치 파악 ---
    ry_rank = merchant_row.get('M12_SME_RY_SAA_PCE_RT', np.nan) # 업종 내 순위 (0이 상위)
    bzn_rank = merchant_row.get('M12_SME_BZN_SAA_PCE_RT', np.nan) # 상권 내 순위 (0이 상위)
    
    comp_analysis_text = "경쟁 환경 순위 정보가 불충분합니다."
    rank_threshold = 70.0 # 하위 30% (순위 70% 이상)

    if not pd.isna(ry_rank) and not pd.isna(bzn_rank):
        # 순위가 낮을수록 상위권이므로, 퍼센트를 뒤집어 직관적으로 표시
        ry_pos_text = f"상위 {(100 - ry_rank):.1f}%"
        bzn_pos_text = f"상위 {(100 - bzn_rank):.1f}%"
        
        if ry_rank < 30.0 and bzn_rank < 30.0:
            comp_analysis_text = f"**동일 업종({ry_pos_text})과 상권({bzn_pos_text})** 모두에서 최상위권의 압도적인 매출 경쟁력을 보이고 있습니다."
        elif ry_rank < 30.0 and bzn_rank >= rank_threshold:
            comp_analysis_text = f"현재 **동일 업종 내에서는 {ry_pos_text} 수준의 준수한 매출**을 보이고 있으나, **가게가 위치한 상권 내에서는 하위권**({bzn_pos_text})으로, 주변 다른 업종 가게들과의 경쟁에서 밀리고 있습니다."
        elif ry_rank >= rank_threshold and bzn_rank >= rank_threshold:
            comp_analysis_text = f"**동일 업종({ry_pos_text})과 상권({bzn_pos_text})** 모두에서 매출 순위가 하위권에 머물고 있어 경쟁력 확보가 시급합니다."
        else:
            comp_analysis_text = f"동일 업종 내 매출 순위는 {ry_pos_text}, 상권 내 매출 순위는 {bzn_pos_text} 수준으로, 상세한 전략 검토가 필요합니다."
    
    
    # 모든 분석 결과를 담아 반환
    return {
        'cust_analysis_text': cust_analysis_text,
        'retention_analysis_text': retention_analysis_text,
        'comp_analysis_text': comp_analysis_text,
        'static_info': merchant_row[STATIC_COLS].to_dict(),
        'metric_info': merchant_row[QUARTILE_METRIC_COLS].to_dict(),
        'age_gender_cols': AGE_GENDER_COLS, # 차트 생성을 위해 추가 정보 반환
        'age_gender_names': AGE_GENDER_NAMES,
    }
