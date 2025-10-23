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

    st.markdown("### 1) 🎯 성장-안정성 매트릭스")
    _plot_growth_stability_matrix(
        industry_indicators, selected_ind=selected_ind, topN=topN
    )

    st.write("")
    st.markdown("### 2) 🛡️ 레이더 차트 (업종_매핑 비교)")
    _plot_radar(
        industry_indicators,
        selected_ind=selected_ind,
        k_top=k_top,
        scaling_method=scaling_method,  
        scope=scope,                    
    )

    st.markdown("### 3) 🗺️ 상권(행정동) × 업종_매핑 히트맵")
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
) -> tuple[str | None, str | None, dict]:
    """
    반환:
      mapped_label : 업종_매핑(시각화용) 또는 None
      raw_industry_name : data_dong.csv의 HPSN_MCT_ZCD_NM 또는 None
      debug : 중간 디버그 정보(dict)
    매핑 순서:
      1) CODE_TO_CUSTOM 사전 매핑 (raw→mapped)
      2) mapping.csv의 업종_매핑 값들과 직접 일치 (정규화 후 비교)
    """
    debug = {}
    csv_path = os.path.join(base_dir, "data_dong.csv")
    if not os.path.exists(csv_path):
        debug["error"] = f"data_dong.csv not found: {csv_path}"
        return None, None, debug

    try:
        df_dong = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_dong = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        debug["error"] = f"read error: {e}"
        return None, None, debug

    if "ENCODED_MCT" not in df_dong.columns or "HPSN_MCT_ZCD_NM" not in df_dong.columns:
        debug["error"] = "missing columns in data_dong.csv"
        return None, None, debug

    df_dong["_ENC_MCT_STR"] = df_dong["ENCODED_MCT"].astype(str)
    key = str(selected_mct)
    row = df_dong.loc[df_dong["_ENC_MCT_STR"] == key]

    if row.empty:
        debug["error"] = f"no row for ENCODED_MCT={key}"
        return None, None, debug

    raw_name = row["HPSN_MCT_ZCD_NM"].iloc[0]
    raw_norm = _norm_label(raw_name)
    debug["raw_name"] = raw_name
    debug["raw_norm"] = raw_norm

    # 1) 사전 매핑 (키 정규화)
    dict_norm = { _norm_label(k): v for k, v in code_to_custom.items() }
    if raw_norm in dict_norm:
        mapped = dict_norm[raw_norm]
        debug["via"] = "dict"
        debug["mapped"] = mapped
        return mapped, raw_name, debug

    # 2) mapping.csv의 업종_매핑 값들과 직접 일치(정규화)
    mapping_norm_to_orig = {}
    for v in mapping_values:
        vn = _norm_label(v)
        mapping_norm_to_orig.setdefault(vn, v)

    if raw_norm in mapping_norm_to_orig:
        mapped = mapping_norm_to_orig[raw_norm]
        debug["via"] = "direct"
        debug["mapped"] = mapped
        return mapped, raw_name, debug

    # 실패
    debug["via"] = "none"
    return None, raw_name, debug


# =====================================================================================
# Plot 1: 성장–안정 매트릭스
# =====================================================================================

def _plot_growth_stability_matrix(ind_df: pd.DataFrame, selected_ind: str | None, topN: int):
    if ind_df.empty:
        st.warning("업종 집계 데이터가 비어 있습니다.")
        return

    mat_df = ind_df.sort_values("총점포수", ascending=False).head(topN)

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = (mat_df["총점포수"] / mat_df["총점포수"].max() * 600) + 60
    colors = mat_df["종합추천점수"]

    sc = ax.scatter(
        mat_df["평균순증가율"],
        mat_df["평균안정성점수"],
        s=sizes,
        c=colors,
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.8,
    )

    # 사분면 기준선
    avg_g = ind_df["평균순증가율"].mean()
    avg_s = ind_df["평균안정성점수"].mean()
    ax.axvline(avg_g, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(avg_s, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # 코너 라벨
    xmin, xmax = mat_df["평균순증가율"].min(), mat_df["평균순증가율"].max()
    ymin, ymax = mat_df["평균안정성점수"].min(), mat_df["평균안정성점수"].max()
    ax.text(xmax, ymax, "고성장·고안정", ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="#c8e6c9", alpha=0.6))
    ax.text(xmin, ymax, "저성장·고안정", ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="#bbdefb", alpha=0.6))
    ax.text(xmax, ymin, "고성장·저안정", ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="#ffe082", alpha=0.6))
    ax.text(xmin, ymin, "저성장·저안정", ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="#ffcdd2", alpha=0.6))

    # 상위 5 라벨
    for ind, row in mat_df.nlargest(5, "종합추천점수").iterrows():
        label = ind if len(ind) <= 12 else ind[:12] + "…"
        ax.annotate(
            label,
            xy=(row["평균순증가율"], row["평균안정성점수"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", lw=0.6),
        )

    # 자동/수동 선택 업종 하이라이트
    if selected_ind and selected_ind in mat_df.index:
        r = mat_df.loc[selected_ind]
        ax.scatter(
            r["평균순증가율"], r["평균안정성점수"],
            s=((r["총점포수"] / mat_df["총점포수"].max()) * 900) + 100,
            facecolors="none", edgecolors="blue", linewidths=2.5,
            label=f"선택: {selected_ind}",
        )
        ax.legend(loc="best")

    ax.set_xlabel("순증가율 (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("사업안정성점수", fontsize=12, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("종합추천점수", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


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

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
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
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    ax.set_title(f"레이더 차트 — {title}", fontsize=14, fontweight="bold", pad=22)
    st.pyplot(fig)

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
        figsize=(1.0 * len(col_inds) + 6, 0.45 * max(len(pivot.index), 6) + 3)
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [c if len(c) <= 8 else c[:8] + "…" for c in pivot.columns],
        rotation=45, ha="right", fontsize=10
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    ax.set_title("상권(행정동) × 업종_매핑 — 평균 추천점수(근사)", fontsize=13, fontweight="bold", pad=10)
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