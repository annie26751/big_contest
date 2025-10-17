# visualize.py
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# 한글 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 경로
DATA_PATH = Path('./data/data_dong.csv')

# 데이터 로드 (우선 새로 로드, 추후 통합)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path, encoding='utf-8')
    # 날짜 처리
    if 'TA_YM_DT' in df.columns:
        df['TA_YM_DT'] = pd.to_datetime(df['TA_YM_DT'].astype(str).str[:7] + '-01', errors='coerce')
    return df

# ── KPI 보드 시각화 ────────────────────────────────────────────
def kpi_board(df: pd.DataFrame, selected_mct: str, REF: pd.Timestamp | None = None):
    df_mct = df[df['ENCODED_MCT'] == selected_mct].copy()
    if df_mct.empty:
        st.warning(f"가맹점 데이터가 없습니다: {selected_mct}")
        return

    if 'TA_YM_DT' in df_mct.columns and not np.issubdtype(df_mct['TA_YM_DT'].dtype, np.datetime64):
        df_mct['TA_YM_DT'] = pd.to_datetime(df_mct['TA_YM_DT'].astype(str).str[:7] + '-01', errors='coerce')

    if REF is None:
        REF = df_mct['TA_YM_DT'].max()
    REF = pd.to_datetime(REF)
    PREV = (REF.to_period('M') - 1).to_timestamp()

    KPI_COLS = [
        'M1_SME_RY_SAA_RAT',      # 업종 평균 대비 매출비율
        'MCT_UE_CLN_REU_RAT',     # 재방문 고객 비중
        'MCT_UE_CLN_NEW_RAT'      # 신규 고객 비중
    ]

    dfm = df_mct.groupby('TA_YM_DT')[KPI_COLS].mean().sort_index()

    if REF not in dfm.index:
        st.warning("기준월 데이터가 없습니다.")
        return

    now = dfm.loc[REF]
    prev = dfm.loc[PREV] if PREV in dfm.index else pd.Series(index=KPI_COLS, dtype='float64')

    def delta_value(col, now_v, prev_v):
        if pd.isna(prev_v):
            return np.nan
        if col.endswith('PCE_RT'):
            return prev_v - now_v
        else:
            return (now_v / prev_v - 1) * 100

    def fmt_val(col, v):
        return "데이터 없음" if pd.isna(v) else (f"{v:.1f}p" if col.endswith('PCE_RT') else f"{v:.1f}%")

    def fmt_delta(col, d):
        if pd.isna(d):
            return "–"
        unit = "p" if col.endswith('PCE_RT') else "%"
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.1f}{unit}"

    def color_for(d):
        if pd.isna(d):
            return '#888888'
        if abs(d) < 0.05:
            return '#888888'
        return 'red' if d > 0 else 'blue'

    labels = {
        'M1_SME_RY_SAA_RAT': '업종 평균 대비 매출비율',
        'MCT_UE_CLN_REU_RAT': '재방문 고객 비중',
        'MCT_UE_CLN_NEW_RAT': '신규 고객 비중'
    }

    cards = []
    for col in KPI_COLS:
        dv = delta_value(col, now[col], prev[col] if col in prev.index else np.nan)
        cards.append((labels[col], now[col], dv, col))

    fig, axes = plt.subplots(1, len(cards), figsize=(13, 3.4))
    fig.suptitle(f"{REF:%Y-%m} 전월대비 성과", fontsize=15, weight='bold')

    if len(cards) == 1:
        axes = [axes]

    for ax, (name, val, dlt, col) in zip(axes, cards):
        ax.axis('off')
        ax.set_facecolor('#f8f9fa')

        c = color_for(dlt)
        if pd.isna(dlt):
            arrow, change_str = '–', '데이터 없음'
        elif abs(dlt) < 0.05:
            arrow, change_str = '–', '변화 없음'
        else:
            arrow = '▲' if dlt > 0 else '▼'
            change_str = fmt_delta(col, dlt)

        val_str = fmt_val(col, val)
        ax.text(0.5, 0.65, f"{arrow} {change_str}", ha='center', va='center', fontsize=22, color=c, weight='bold')
        ax.text(0.5, 0.42, f"현재: {val_str}", ha='center', va='center', fontsize=12, color='black')
        ax.text(0.5, 0.18, name, ha='center', va='center', fontsize=13, weight='semibold')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    st.pyplot(fig)

# ── 성별 도넛 차트 ────────────────────────────────────────────
def gender_age_pie(df: pd.DataFrame, selected_mct: str, REF: pd.Timestamp | None = None):
    df_mct = df[df['ENCODED_MCT'] == selected_mct].copy()
    if df_mct.empty:
        st.warning(f"가맹점 데이터가 없습니다: {selected_mct}")
        return

    # 날짜 컬럼 처리
    if 'TA_YM_DT' in df_mct.columns and not np.issubdtype(df_mct['TA_YM_DT'].dtype, np.datetime64):
        df_mct['TA_YM_DT'] = pd.to_datetime(df_mct['TA_YM_DT'].astype(str).str[:7] + '-01', errors='coerce')

    # 기준월 결정
    if REF is None:
        REF = df_mct['TA_YM_DT'].max()
    REF = pd.to_datetime(REF)

    # 성별·연령 컬럼
    cols = [
        'M12_MAL_1020_RAT','M12_MAL_30_RAT','M12_MAL_40_RAT','M12_MAL_50_RAT','M12_MAL_60_RAT',
        'M12_FME_1020_RAT','M12_FME_30_RAT','M12_FME_40_RAT','M12_FME_50_RAT','M12_FME_60_RAT'
    ]
    labels = [
        '남≤20','남30','남40','남50','남60+',
        '여≤20','여30','여40','여50','여60+'
    ]

    # 선택월 데이터 평균
    tmp = df_mct.loc[df_mct['TA_YM_DT'] == REF, cols].mean().fillna(0)
    vals = tmp.values
    total = vals.sum()
    if total > 0 and (total < 95 or total > 105):  # 합이 100% 아닌 경우 보정
        vals = vals / total * 100

    # 색상 팔레트 (남=파랑, 여=빨강)
    colors = [
        '#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd',   # 남
        '#b91c1c', '#dc2626', '#ef4444', '#f87171', '#fca5a5'    # 여
    ]

    # 도넛 차트
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, autopct='%1.0f%%',
        startangle=90, counterclock=False,
        wedgeprops={'width':0.35, 'edgecolor':'white'},
        colors=colors, pctdistance=0.8
    )

    plt.setp(autotexts, size=10, weight='bold', color='white')
    ax.set_title(f"{REF:%Y-%m} 기준 성별·연령 고객 구성", fontsize=14, weight='semibold')
    plt.tight_layout()
    st.pyplot(fig)

# ── 고객 유형 도넛 차트 ────────────────────────────────────────────
def customer_type_pie(df: pd.DataFrame, selected_mct: str, REF: pd.Timestamp | None = None):
    df_mct = df[df['ENCODED_MCT'] == selected_mct].copy()
    if df_mct.empty:
        st.warning(f"가맹점 데이터가 없습니다: {selected_mct}")
        return

    # 날짜 컬럼 처리
    if 'TA_YM_DT' in df_mct.columns and not np.issubdtype(df_mct['TA_YM_DT'].dtype, np.datetime64):
        df_mct['TA_YM_DT'] = pd.to_datetime(df_mct['TA_YM_DT'].astype(str).str[:7] + '-01', errors='coerce')

    # 기준월 결정
    if REF is None:
        REF = df_mct['TA_YM_DT'].max()
    REF = pd.to_datetime(REF)

    # 컬럼 및 레이블 정의
    cols_left  = ['MCT_UE_CLN_REU_RAT','MCT_UE_CLN_NEW_RAT']
    cols_right = ['RC_M1_SHC_RSD_UE_CLN_RAT','RC_M1_SHC_WP_UE_CLN_RAT','RC_M1_SHC_FLP_UE_CLN_RAT']
    labels_left  = ['재방문','신규']
    labels_right = ['거주','직장','유동']

    # 선택월 평균값 계산 및 합이 100% 아닌 경우 보정
    L = df_mct.loc[df_mct['TA_YM_DT']==REF, cols_left].mean().fillna(0)
    R = df_mct.loc[df_mct['TA_YM_DT']==REF, cols_right].mean().fillna(0)
    L = L if L.sum()==0 else L/L.sum()*100
    R = R if R.sum()==0 else R/R.sum()*100

    # 색상 팔레트
    left_colors  = ['#0ea5e9','#94a3b8']      # 재방문 파랑, 신규 회색블루
    right_colors = ['#22c55e','#f59e0b','#64748b']  # 거주 초록, 직장 앰버, 유동 슬레이트

    # 좌측 도넛
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    ax.pie(L.values, labels=labels_left, autopct='%1.0f%%', startangle=90,
           wedgeprops={'width':0.35,'edgecolor':'white'}, colors=left_colors, pctdistance=0.8)
    ax.set_title(f'{REF:%Y-%m} 재방문 vs 신규', fontsize=12, weight='semibold')
    plt.tight_layout()
    st.pyplot(fig)

    # 우측 도넛
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    ax.pie(R.values, labels=labels_right, autopct='%1.0f%%', startangle=90,
           wedgeprops={'width':0.35,'edgecolor':'white'}, colors=right_colors, pctdistance=0.8)
    ax.set_title(f'{REF:%Y-%m} 거주/직장/유동', fontsize=12, weight='semibold')
    plt.tight_layout()
    st.pyplot(fig)