# visualization_area.py

import os
import re
import unicodedata
from math import pi

import numpy as np
import pandas as pd
import matplotlib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from pathlib import Path

font_path = Path(__file__).parent / "NanumGothic.ttf"
font_manager.fontManager.addfont(str(font_path))

rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False

CODE_TO_CUSTOM = {
    "한식음식점": "한식-단품요리일반",
    "중식음식점": "중식당",
    "일식음식점": "일식당",
    "양식음식점": "양식",
    "제과점": "베이커리",
    "패스트푸드점": "햄버거",
    "치킨전문점": "치킨",
    "분식전문점": "분식",
    "호프-간이주점": "호프/맥주",
    "커피-음료": "커피전문점",
    "슈퍼마켓": "식료품",
    "편의점": "식료품",
    "주류도매": "주류",
    "미곡판매": "식료품",
    "육류판매": "축산물",
    "수산물판매": "수산물",
    "청과상": "청과물",
    "반찬가게": "반찬",
}

def render_area_dashboard(
    df_filtered: pd.DataFrame,
    selected_mct: str | int | None = None,
    base_dir: str | None = None,
):
    """
    상권 분석 시각화 엔트리.
    Parameters
    ----------
    df_filtered : pd.DataFrame
        mapping.csv를 읽은 전처리 결과(행 단위). 필수 컬럼:
        ['행정동_코드_명','업종_매핑','점포_수','유사_업종_점포_수','개업_율','폐업_률']
    selected_mct : str|int|None
        사이드바에서 선택한 ENCODED_MCT
    base_dir : str|None
        mapping.csv가 위치한 폴더 경로(동일 폴더에 data_dong.csv 존재 가정)
    """
    _validate_columns(
        df_filtered,
        req=["행정동_코드_명", "업종_매핑", "점포_수", "유사_업종_점포_수", "개업_율", "폐업_률"],
    )

    # 내부 비교 안정화를 위해 '업종_매핑' 정규화 보조 컬럼 생성(표시는 원본 라벨 유지)
    df_filtered = df_filtered.copy()
    df_filtered["_업종_매핑_norm"] = df_filtered["업종_매핑"].map(_norm_label)

    # 1) 행 단위 파생지표
    dfm = _build_row_metrics(df_filtered)

    # 2) 업종_매핑 기준 집계
    industry_indicators = _build_industry_indicators(dfm)

    # 3) 선택 가맹점 → data_dong.csv에서 업종명 → 업종_매핑 자동 선택(스마트)
    auto_selected_ind, raw_industry_name, dbg = None, None, {}
    if selected_mct is not None and base_dir:
        mapping_values = sorted(df_filtered["업종_매핑"].dropna().unique().tolist())
        auto_selected_ind, raw_industry_name, dbg = _auto_pick_industry_by_mct_smart(
            selected_mct=selected_mct,
            base_dir=base_dir,
            mapping_values=mapping_values,
            code_to_custom=CODE_TO_CUSTOM,
        )

    st.subheader("📈 상권 분석 시각화")

    _render_metric_glossary()

    with st.container(border=True):
        cols = st.columns([5, 2, 2, 2])

        with cols[0]:
            inds_all = list(industry_indicators.index)
            manual_override = False

            if auto_selected_ind:
                st.success(
                    f"자동 매핑 업종: **{auto_selected_ind}**"
                    + (f"  \n(원본 업종명: `{raw_industry_name}`)" if raw_industry_name else "")
                )
                manual_override = st.toggle(
                    "수동으로 변경", value=False, help="체크하면 업종을 직접 선택할 수 있어요."
                )
            else:
                st.warning("자동 매핑된 업종이 없습니다. 아래에서 직접 선택해 주세요.")
                manual_override = True

            # 최종 선택 업종
            if manual_override:
                default_idx = 0
                if auto_selected_ind in inds_all:
                    default_idx = max(0, inds_all.index(auto_selected_ind))
                selected_ind = st.selectbox("업종_매핑 선택", inds_all, index=default_idx)
            else:
                selected_ind = auto_selected_ind

        with cols[1]:
            topN = st.slider("매트릭스 TOP N", 10, 50, 30, 5)
        with cols[2]:
            k_top = st.slider("레이더 비교 수", 3, 8, 5, 1)
        with cols[3]:
            max_cols = st.slider("히트맵 업종 수", 5, 20, 10, 1)

        col_a, col_b = st.columns(2)
        with col_a:
            scaling_method = st.selectbox(
                "레이더 스케일 방법",
                ["robust-minmax", "zscore"],
                index=0,
                help="robust-minmax: 10~90 분위 기반 / zscore: 평균·표준편차 기반"
            )
        with col_b:
            scope = st.selectbox(
                "스케일 참조 범위",
                ["global", "compare-set"],
                index=0,
                help="global: 전체 업종 기준 / compare-set: 현재 비교대상 k개 기준"
            )

        with st.expander("🔎 행정동 필터 (선택)"):
            dongs = sorted(dfm["행정동_코드_명"].unique().tolist())
            dong_sel = st.multiselect("행정동 선택", dongs, default=[])

    st.markdown("#### 1) 🎯 성장-안정성 매트릭스")
    _plot_growth_stability_matrix(
        industry_indicators, selected_ind=selected_ind, topN=topN
    )

    st.markdown("#### 2) 🛡️ 레이더 차트 (업종_매핑 비교)")
    
    # 레이더 차트 설정 설명
    with st.expander("ℹ️ 레이더 차트 설정 가이드", expanded=False):
        st.markdown("""
        **🔧 레이더 스케일 방법**
        
        레이더 차트의 각 축을 0~100 범위로 변환하는 방법을 선택합니다:
        
        - **robust-minmax**: 극단값의 영향을 줄이는 방법
          - 10% 분위수(하위 10%)와 90% 분위수(상위 10%)를 기준으로 스케일링
          - 이상치(outlier)가 있어도 차트가 안정적으로 표시됨
          - **추천**: 데이터에 극단적인 값이 있을 때 유용
        
        - **zscore**: 평균과 표준편차를 기준으로 변환
          - 평균을 50으로, ±5 표준편차를 0~100으로 매핑
          - 통계적으로 표준화된 비교가 가능
          - **추천**: 정규분포를 따르는 일반적인 데이터에 적합
        
        ---
        
        **🎯 스케일 참조 범위**
        
        축의 0~100 스케일을 계산할 때 어떤 데이터를 기준으로 할지 선택합니다:
        
        - **global**: 전체 업종을 기준으로 스케일링
          - 모든 업종의 데이터를 포함하여 최소/최대값 또는 평균/표준편차 계산
          - 업종 간 절대적 비교가 가능
          - **추천**: 전체 시장에서의 상대적 위치를 파악할 때
        
        - **compare-set**: 현재 비교 중인 업종들만 기준으로 스케일링
          - 레이더에 표시되는 k개 업종의 데이터만으로 스케일 계산
          - 선택된 업종들 간의 상대적 차이가 더 명확하게 표시됨
          - **추천**: 유사한 업종끼리 세밀하게 비교할 때
        
        💡 **팁**: 처음에는 `robust-minmax` + `global` 조합으로 전체적인 모습을 파악한 후, 
        `compare-set`으로 변경하여 선택 업종들 간의 미세한 차이를 확인하는 것을 추천합니다.
        """)
    
    _plot_radar(
        industry_indicators,
        selected_ind=selected_ind,
        k_top=k_top,
        scaling_method=scaling_method,  
        scope=scope,                    
    )

    st.markdown("#### 3) 🗺️ 상권(행정동) × 업종_매핑 히트맵")
    _plot_heatmap(
        dfm,
        industry_indicators,
        selected_ind=selected_ind,
        max_cols=max_cols,
        dong_filter=dong_sel,
    )

def _validate_columns(df, req):
    miss = [c for c in req if c not in df.columns]
    if miss:
        st.error(f"필수 컬럼이 없습니다: {miss}")
        st.stop()


def _norm_label(x: str) -> str:
    """라벨 비교 안정화를 위한 정규화: 전각/반각, 공백, 제로폭, BOM 제거 등."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\ufeff", "") 
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s) 
    s = " ".join(s.strip().split())  
    return s


@st.cache_data(show_spinner=False)
def _build_row_metrics(df_filtered: pd.DataFrame) -> pd.DataFrame:
    dfm = df_filtered.copy()
    dfm["순증가율"] = dfm["개업_율"] - dfm["폐업_률"]
    dfm["시장점유율"] = (
        (dfm["점포_수"] / dfm["유사_업종_점포_수"]) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)
    dfm["안정성점수"] = (100 - dfm["폐업_률"] * 2).clip(0, 100)
    safe = dfm["점포_수"].replace(0, np.nan)
    dfm["경쟁강도_수치"] = (
        (dfm["유사_업종_점포_수"] / safe).replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    comp = (100 - (dfm["경쟁강도_수치"] * 10)).clip(0, 100)
    dfm["추천점수_근사"] = (
        dfm["안정성점수"] * 0.4
        + dfm["순증가율"] * 2
        + comp * 0.3
    )
    return dfm


@st.cache_data(show_spinner=False)
def _build_industry_indicators(dfm: pd.DataFrame) -> pd.DataFrame:
    grp = (
        dfm.groupby("업종_매핑")
        .agg(
            평균시장점유율=("시장점유율", "mean"),
            평균순증가율=("순증가율", "mean"),
            평균안정성점수=("안정성점수", "mean"),
            경쟁강도_수치=("경쟁강도_수치", "mean"),
            종합추천점수=("추천점수_근사", "mean"),
            총점포수=("점포_수", "sum"),
        )
        .sort_values("총점포수", ascending=False)
    )
    return grp


def _render_metric_glossary():
    st.markdown("#### 📘 지표 설명 (How to read)")
    data = [
        ("시장 점유율", "점포_수 ÷ 유사_업종_점포_수 × 100", "상권 내 영향력 (업종 내 비중)"),
        ("순증가율 (Net Growth)", "개업_율 - 폐업_률", "점포 수 변화율 (양수=증가, 음수=감소)"),
        ("사업안정성점수", "100 - (폐업_률 × 2) → 0~100", "폐업률이 낮을수록 높아지는 안정성 점수"),
        ("성장모멘텀", "순증가율 구간화", "급락(<-10), 하락(-10~-2), 정체(-2~2), 성장(2~10), 급성장(>10)"),
        ("경쟁강도", "경쟁강도_수치", "낮음(<1.2), 보통(<2.0), 높음(<3.5), 매우높음(<5.0), 초경쟁(≥5.0)"),
    ]
    df_gloss = pd.DataFrame(data, columns=["지표", "계산 방법", "의미"])
    st.dataframe(df_gloss, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def _auto_pick_industry_by_mct_smart(
    selected_mct: str | int,
    base_dir: str,
    mapping_values: list[str],
    code_to_custom: dict,
):
    """data_dong.csv에서 ENCODED_MCT별 업종명 → 업종_매핑 자동 선택"""
    debug_info = {}
    csv_path = os.path.join(base_dir, "data_dong.csv")
    if not os.path.exists(csv_path):
        debug_info["error"] = f"data_dong.csv not found at {csv_path}"
        return None, None, debug_info

    try:
        df_dong = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except Exception as e:
        debug_info["error"] = f"data_dong.csv read error: {e}"
        return None, None, debug_info

    if "ENCODED_MCT" not in df_dong.columns or "업종" not in df_dong.columns:
        debug_info["error"] = "data_dong.csv missing required columns"
        return None, None, debug_info

    df_dong["ENCODED_MCT"] = df_dong["ENCODED_MCT"].astype(str).str.strip()
    selected_mct_str = str(selected_mct).strip()
    debug_info["selected_mct"] = selected_mct_str

    mask = df_dong["ENCODED_MCT"] == selected_mct_str
    if not mask.any():
        debug_info["error"] = f"ENCODED_MCT={selected_mct_str} not found"
        return None, None, debug_info

    raw_name = df_dong.loc[mask, "업종"].iloc[0]
    debug_info["raw_industry"] = raw_name

    candidates = []
    for mv in mapping_values:
        mv_norm = _norm_label(mv)
        raw_norm = _norm_label(raw_name)

        if mv_norm == raw_norm:
            candidates.append((mv, 1000))
        elif mv_norm in raw_norm:
            candidates.append((mv, 800 + len(mv_norm) * 10))
        elif raw_norm in mv_norm:
            candidates.append((mv, 600 + len(raw_norm) * 10))

        for code_key, custom_val in code_to_custom.items():
            code_norm = _norm_label(code_key)
            custom_norm = _norm_label(custom_val)
            if raw_norm == code_norm and mv_norm == custom_norm:
                candidates.append((mv, 900))
            elif code_norm in raw_norm and mv_norm == custom_norm:
                candidates.append((mv, 700))

    debug_info["candidates"] = candidates
    if not candidates:
        return None, raw_name, debug_info

    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    best = candidates_sorted[0][0]
    debug_info["best_match"] = best
    return best, raw_name, debug_info


# =====================================================================================
# Plot 1: 성장-안정성 매트릭스 (수정됨)
# =====================================================================================

def _plot_growth_stability_matrix(
    ind_df: pd.DataFrame,
    selected_ind: str | None,
    topN: int,
):
    if ind_df.empty:
        st.warning("업종 집계 데이터가 비어 있습니다.")
        return

    mat_df = ind_df.nlargest(topN, "종합추천점수")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # 모든 점 표시 (회색)
    sc = ax.scatter(
        mat_df["평균순증가율"],
        mat_df["평균안정성점수"],
        s=((mat_df["총점포수"] / mat_df["총점포수"].max()) * 400) + 50,
        c=mat_df["종합추천점수"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="gray",
        linewidths=0.8,
    )

    # 4분할 기준선
    x_mean = mat_df["평균순증가율"].mean()
    y_mean = mat_df["평균안정성점수"].mean()
    ax.axvline(x_mean, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y_mean, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # 사분면 이름
    corner_font = dict(fontsize=7, alpha=0.9)
    ax.text(xmax, ymax, "고성장·고안정", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#c8e6c9", alpha=0.5), **corner_font)
    ax.text(xmin, ymax, "저성장·고안정", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#bbdefb", alpha=0.5), **corner_font)
    ax.text(xmax, ymin, "고성장·저안정", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe082", alpha=0.5), **corner_font)
    ax.text(xmin, ymin, "저성장·저안정", ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffcdd2", alpha=0.5), **corner_font)


    # 상위 5개 라벨
    for ind, row in mat_df.nlargest(5, "종합추천점수").iterrows():
        label = ind if len(ind) <= 10 else ind[:10] + "…"
        ax.annotate(
            label,
            xy=(row["평균순증가율"], row["평균안정성점수"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="yellow", alpha=0.4),
            arrowprops=dict(arrowstyle="->", lw=0.5),
        )

    # 선택 업종 하이라이트
    if selected_ind and selected_ind in mat_df.index:
        r = mat_df.loc[selected_ind]
        point_size = ((r["총점포수"] / mat_df["총점포수"].max()) * 400) + 50
        ax.scatter(
            r["평균순증가율"], 
            r["평균안정성점수"],
            s=point_size,
            facecolors="none", 
            edgecolors="blue", 
            linewidths=2.5,
            zorder=5
        )
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor='none', markeredgecolor='blue', 
                   markersize=8, markeredgewidth=2.5,
                   label=f'선택: {selected_ind}')
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=6,
                  frameon=True, framealpha=0.95, edgecolor="blue")

    ax.set_xlabel("순증가율 (%)", fontsize=6, fontweight="bold", labelpad=2)
    ax.set_ylabel("사업안정성점수", fontsize=6, fontweight="bold", labelpad=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("종합추천점수", fontsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    plt.tight_layout(pad=0.4)
    st.pyplot(fig)
    
    # 선택된 업종의 사분면 위치에 따른 설명 표시
    if selected_ind and selected_ind in mat_df.index:
        r = mat_df.loc[selected_ind]
        growth = r["평균순증가율"]
        stability = r["평균안정성점수"]
        
        # 사분면 판단
        if growth >= x_mean and stability >= y_mean:
            quadrant = "고성장·고안정"
            description = "안정적이며 성장 여력도 큰 시장입니다. 진입하기 유망한 업종으로, 시장이 확대되고 있으면서도 기존 사업자들의 폐업률이 낮아 안정적인 운영이 가능합니다."
            color = "#2e7d32"
        elif growth < x_mean and stability >= y_mean:
            quadrant = "저성장·고안정"
            description = "성장은 낮지만 안정적인 업종입니다. 급격한 성장보다는 꾸준한 운영이 중요하며, 유지·보수형 전략이 적합합니다. 기존 고객층 확보가 중요합니다."
            color = "#1565c0"
        elif growth >= x_mean and stability < y_mean:
            quadrant = "고성장·저안정"
            description = "빠르게 성장 중이나 변동성이 높은 업종입니다. 높은 수익 기회가 있지만 리스크 관리가 필수적입니다. 시장 트렌드 변화에 민감하게 대응해야 합니다."
            color = "#ef6c00"
        else:  # growth < x_mean and stability < y_mean
            quadrant = "저성장·저안정"
            description = "성장성과 안정성이 모두 낮은 업종입니다. 진입 전 신중한 검토가 필요하며, 차별화된 경쟁력이나 틈새 시장 전략이 있어야 성공 가능성이 높아집니다."
            color = "#c62828"
        
        st.markdown(f"""
        <div style="padding: 15px; border-left: 4px solid {color}; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
            <h4 style="color: {color}; margin-top: 0;">📍 선택 업종 '{selected_ind}'의 위치: {quadrant}</h4>
            <p style="margin-bottom: 0; line-height: 1.6;">{description}</p>
        </div>
        """, unsafe_allow_html=True)



# =====================================================================================
# Plot 2: 레이더 차트 (축별 정규화 옵션 반영)
# =====================================================================================

def _scale_block(values: pd.DataFrame, method: str = "robust-minmax") -> tuple[pd.Series, pd.Series]:
    """각 열(축)마다 스케일 파라미터(min/max 또는 mean/std) 계산"""
    if method == "robust-minmax":
        q10 = values.quantile(0.10)
        q90 = values.quantile(0.90)
        span = (q90 - q10).replace(0, np.nan)
        return q10, span
    elif method == "zscore":
        mean = values.mean()
        std = values.std(ddof=0).replace(0, np.nan)
        return mean, std
    else:
        raise ValueError("Unknown scaling method")


def _apply_scale(row: pd.Series, params_a: pd.Series, params_b: pd.Series, method: str) -> pd.Series:
    """스케일 적용 후 0~100에 매핑 (경쟁강도는 역수 처리)"""
    s = row.copy()

    if method == "robust-minmax":
        x = (s - params_a) / params_b
        x = x.clip(0, 1) * 100.0
    else:  # zscore
        z = (s - params_a) / params_b
        x = (50.0 + 10.0 * z).clip(0, 100)  # ±5σ ≈ 0~100

    # 경쟁강도 축 뒤집기(낮을수록 우수)
    if "경쟁강도_수치" in s.index:
        x["경쟁강도_수치"] = 100.0 - x["경쟁강도_수치"]

    # 순증가율은 대비 약하면 약간 강조
    if "평균순증가율" in x.index:
        x["평균순증가율"] = (x["평균순증가율"] * 1.05).clip(0, 100)

    return x


def _plot_radar(
    ind_df: pd.DataFrame,
    selected_ind: str | None,
    k_top: int,
    scaling_method: str = "robust-minmax",
    scope: str = "global",
):
    if ind_df.empty:
        st.warning("업종 집계 데이터가 비어 있습니다.")
        return

    # 1) 비교 그룹 선정
    if selected_ind and selected_ind in ind_df.index:
        X = ind_df[["종합추천점수", "평균안정성점수", "평균순증가율", "평균시장점유율", "경쟁강도_수치"]].copy()
        X_std = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
        d = ((X_std - X_std.loc[selected_ind]) ** 2).sum(axis=1)
        peers = d.nsmallest(k_top + 1).index.tolist()  # 자기 자신 포함
        inds = [i for i in peers if i != selected_ind][: (k_top - 1)]
        compare_list = [selected_ind] + inds
        title = f"선택 업종 중심 비교 (총 {len(compare_list)}개)"
    else:
        compare_list = ind_df.nlargest(k_top, "종합추천점수").index.tolist()
        title = f"종합추천 상위 {len(compare_list)}개 비교"

    # 2) 스케일 참조 데이터 (global / compare-set)
    cols = ["평균시장점유율", "평균순증가율", "평균안정성점수", "경쟁강도_수치", "종합추천점수"]
    ref = ind_df.loc[compare_list, cols] if scope == "compare-set" else ind_df[cols]

    # 3) 스케일 파라미터 계산
    a, b = _scale_block(ref, method=scaling_method)

    # 4) 폴라 차트
    categories = ["시장점유율", "순증가율", "안정성점수", "경쟁강도(역수)", "종합추천"]
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 2), subplot_kw=dict(projection="polar"))
    for ind in compare_list:
        row = ind_df.loc[ind, cols]
        scaled = _apply_scale(row, a, b, scaling_method)
        values = [
            float(scaled["평균시장점유율"]),
            float(scaled["평균순증가율"]),
            float(scaled["평균안정성점수"]),
            float(scaled["경쟁강도_수치"]),   # 이미 역수 반영
            float(scaled["종합추천점수"]),
        ]
        values += values[:1]
        lbl = ind if len(ind) <= 15 else ind[:15] + "…"
        ax.plot(angles, values, "o-", linewidth=2, label=lbl)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=5)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=5)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=5)
    st.pyplot(fig, use_container_width=False)

    st.caption(
        f"스케일: **{scaling_method}**, 참조: **{scope}** "
        f"(경쟁강도는 낮을수록 우수 → 축 반전)"
    )


# =====================================================================================
# Plot 3: 히트맵
# =====================================================================================

def _plot_heatmap(
    dfm: pd.DataFrame,
    ind_df: pd.DataFrame,
    selected_ind: str | None,
    max_cols: int,
    dong_filter: list[str],
):
    tmp = dfm.copy()

    # 히트맵 열 선택: 선택 업종 + 상위 업종
    if selected_ind and selected_ind in ind_df.index:
        top_others = (
            ind_df.drop(index=selected_ind)
            .nlargest(max(max_cols - 1, 1), "종합추천점수")
            .index.tolist()
        )
        col_inds = [selected_ind] + top_others
    else:
        col_inds = ind_df.nlargest(max_cols, "종합추천점수").index.tolist()

    if dong_filter:
        tmp = tmp[tmp["행정동_코드_명"].isin(dong_filter)]

    pivot = (
        tmp[tmp["업종_매핑"].isin(col_inds)]
        .pivot_table(
            index="행정동_코드_명",
            columns="업종_매핑",
            values="추천점수_근사",
            aggfunc="mean",
        )
        .reindex(columns=col_inds)
        .sort_index()
    )

    if pivot.empty:
        st.info("선택 조건에 해당하는 데이터가 없습니다.")
        return

    fig_h, ax = plt.subplots(
        figsize=(0.7 * len(col_inds) + 3, 0.35 * max(len(pivot.index), 6) + 2)
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [c if len(c) <= 8 else c[:8] + "…" for c in pivot.columns],
        rotation=45, ha="right", fontsize=10
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    cbar = plt.colorbar(im, ax=ax); cbar.set_label("추천점수(근사, 높을수록 우수)", fontsize=10, fontweight="bold")

    if len(pivot.index) * len(pivot.columns) <= 600:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")

    st.pyplot(fig_h)

    csv_bytes = pivot.reset_index().to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ 히트맵 피벗 CSV 다운로드",
        data=csv_bytes,
        file_name="행정동×업종_매핑_추천점수_히트맵.csv",
        mime="text/csv",
    )