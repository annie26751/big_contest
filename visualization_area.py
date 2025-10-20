# visualization_area.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import streamlit as st

# ---- 폰트(윈도우) ----
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# =====================================================================================
# 공개 API
# =====================================================================================
def render_area_dashboard(df_filtered: pd.DataFrame):
    """
    df_filtered = 행 단위 데이터 (필수 컬럼):
      - '행정동_코드_명', '업종_매핑', '점포_수', '유사_업종_점포_수', '개업_율', '폐업_률'
    """
    _validate_columns(
        df_filtered,
        req=[
            "행정동_코드_명",
            "업종_매핑",
            "점포_수",
            "유사_업종_점포_수",
            "개업_율",
            "폐업_률",
        ],
    )

    # 파생 지표 생성(행 단위)
    dfm = _build_row_metrics(df_filtered)

    # 업종_매핑 기준 집계 DF 생성
    industry_indicators = _build_industry_indicators(dfm)

    # ---- UI 컨트롤 ------------------------------------------------------------------
    st.subheader("📈 상권 분석 시각화")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            inds = list(industry_indicators.index)
            inds_show = ["(전체 보기)"] + inds
            selected_ind = st.selectbox("업종_매핑 선택", inds_show, index=0)
        with col2:
            topN = st.slider("매트릭스: TOP N(총점포수 기준)", 10, 50, 30, 5)
        with col3:
            k_top = st.slider("레이더: 비교 업종 수", 3, 8, 5, 1)
        with col4:
            max_cols = st.slider("히트맵: 업종 컬럼 수", 5, 20, 10, 1)

        # 선택된 업종에 따라 동 필터(선택)
        with st.expander("🔎 행정동 필터 (선택)"):
            dongs = sorted(dfm["행정동_코드_명"].unique().tolist())
            dong_sel = st.multiselect("행정동 선택", dongs, default=[])

    # ---- 시각화 1. 성장–안정 매트릭스 -------------------------------------------------
    st.markdown("### 1) 성장-안정성 매트릭스")
    _plot_growth_stability_matrix(
        industry_indicators,
        selected_ind=None if selected_ind == "(전체 보기)" else selected_ind,
        topN=topN,
    )

    # ---- 시각화 2. 레이더 차트 --------------------------------------------------------
    st.markdown("### 2) 레이더 차트 (업종_매핑 비교)")
    _plot_radar(
        industry_indicators,
        selected_ind=None if selected_ind == "(전체 보기)" else selected_ind,
        k_top=k_top,
    )

    # ---- 시각화 3. 히트맵 ------------------------------------------------------------
    st.markdown("### 3) 상권(행정동) × 업종_매핑 히트맵")
    _plot_heatmap(
        dfm,
        industry_indicators,
        selected_ind=None if selected_ind == "(전체 보기)" else selected_ind,
        max_cols=max_cols,
        dong_filter=dong_sel,
    )


# =====================================================================================
# 내부 유틸/계산
# =====================================================================================
def _validate_columns(df, req):
    miss = [c for c in req if c not in df.columns]
    if miss:
        st.error(f"필수 컬럼이 없습니다: {miss}")
        st.stop()


@st.cache_data(show_spinner=False)
def _build_row_metrics(df_filtered: pd.DataFrame) -> pd.DataFrame:
    dfm = df_filtered.copy()

    # 순증가율
    dfm["순증가율"] = dfm["개업_율"] - dfm["폐업_률"]

    # 시장점유율 (%)
    dfm["시장점유율"] = (
        (dfm["점포_수"] / dfm["유사_업종_점포_수"]) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)

    # 안정성 점수 (예: 폐업률이 낮을수록 우수)
    dfm["안정성점수"] = (100 - dfm["폐업_률"] * 2).clip(0, 100)

    # 경쟁강도(임의 정의: 유사 업종 대비 점포 밀도) — 값이 작을수록 우수로 간주
    # 분모 0 방지
    safe = dfm["점포_수"].replace(0, np.nan)
    dfm["경쟁강도_수치"] = (
        (dfm["유사_업종_점포_수"] / safe).replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    # 종합추천(근사): 안정성 0.4, 순증가율(×2) 0.3, 경쟁강도 역수 0.3
    comp = (100 - (dfm["경쟁강도_수치"] * 10)).clip(0, 100)  # 간단한 역수형 정규화
    dfm["추천점수_근사"] = (dfm["안정성점수"] * 0.4) + (dfm["순증가율"] * 2) + (comp * 0.3)

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


# =====================================================================================
# Plot 1: 성장–안정 매트릭스
# =====================================================================================
def _plot_growth_stability_matrix(ind_df: pd.DataFrame, selected_ind: str | None, topN: int):
    if ind_df.empty:
        st.warning("업종 집계 데이터가 비어 있습니다.")
        return

    mat_df = ind_df.sort_values("총점포수", ascending=False).head(topN)

    fig, ax = plt.subplots(figsize=(13, 9))
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
    ax.text(
        xmax, ymax, "⭐ 고성장·고안정", ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="#c8e6c9", alpha=0.6)
    )
    ax.text(
        xmin, ymax, "💎 저성장·고안정", ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="#bbdefb", alpha=0.6)
    )
    ax.text(
        xmax, ymin, "⚠️ 고성장·저안정", ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="#ffe082", alpha=0.6)
    )
    ax.text(
        xmin, ymin, "🚫 저성장·저안정", ha="left", va="bottom",
        bbox=dict(boxstyle="round", facecolor="#ffcdd2", alpha=0.6)
    )

    # 라벨: 종합추천 상위 5개
    label_src = mat_df.nlargest(5, "종합추천점수")
    for ind, row in label_src.iterrows():
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

    # 선택 업종 하이라이트
    if selected_ind and selected_ind in mat_df.index:
        r = mat_df.loc[selected_ind]
        ax.scatter(
            r["평균순증가율"],
            r["평균안정성점수"],
            s=((r["총점포수"] / mat_df["총점포수"].max()) * 900) + 100,
            facecolors="none",
            edgecolors="blue",
            linewidths=2.5,
            label=f"선택: {selected_ind}",
        )
        ax.legend(loc="best")

    ax.set_xlabel("순증가율 (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("사업안정성점수", fontsize=12, fontweight="bold")
    ax.set_title("🎯 성장-안정성 매트릭스 (업종_매핑 기준)", fontsize=14, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("종합추천점수", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


# =====================================================================================
# Plot 2: 레이더 차트
# =====================================================================================
def _plot_radar(ind_df: pd.DataFrame, selected_ind: str | None, k_top: int):
    if ind_df.empty:
        st.warning("업종 집계 데이터가 비어 있습니다.")
        return

    # 비교 대상 선정: 선택 업종이 있으면 그 업종 + 근접한 업종  (없으면 종합추천 상위 k_top)
    if selected_ind and selected_ind in ind_df.index:
        base = ind_df.loc[selected_ind]
        # 유클리드 거리 기반 유사 업종(종합추천, 안정성, 순증가율, 시장점유율, 경쟁강도)
        X = ind_df[
            ["종합추천점수", "평균안정성점수", "평균순증가율", "평균시장점유율", "경쟁강도_수치"]
        ].copy()
        # 표준화(간단)
        X_std = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
        d = ((X_std - X_std.loc[selected_ind]) ** 2).sum(axis=1)
        peers = d.nsmallest(k_top + 1).index.tolist()  # 자기 자신 포함
        inds = [i for i in peers if i != selected_ind][: (k_top - 1)]
        compare_list = [selected_ind] + inds
        title = f"선택 업종 중심 비교 (총 {len(compare_list)}개)"
    else:
        compare_list = ind_df.nlargest(k_top, "종합추천점수").index.tolist()
        title = f"종합추천 상위 {len(compare_list)}개 비교"

    categories = ["시장점유율", "순증가율(정규)", "안정성점수", "경쟁강도(역수)", "종합추천"]
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
    for ind in compare_list:
        row = ind_df.loc[ind]
        values = [
            float(row["평균시장점유율"]),                      # 0~100 가정
            float((row["평균순증가율"] + 50) * 0.8),          # -50~+50 → 0~100 근사
            float(row["평균안정성점수"]),                      # 0~100
            float(max(0, min(100, 100 - (row["경쟁강도_수치"] * 10)))),  # 낮을수록 우수
            float(row["종합추천점수"]),                        # 0~100
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
    ax.set_title(f"🛡️ 레이더 차트 — {title}", fontsize=14, fontweight="bold", pad=22)

    st.pyplot(fig)


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

    # 선택 업종이 있으면 해당 업종 + 종합추천 상위 업종으로 열 구성
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

    # 축 라벨
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [c if len(c) <= 8 else c[:8] + "…" for c in pivot.columns],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    title = "🏙️ 상권(행정동) × 업종_매핑 — 평균 추천점수(근사)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("추천점수(근사, 높을수록 우수)", fontsize=10, fontweight="bold")

    # 각 칸 값 라벨(가독성 제한)
    if len(pivot.index) * len(pivot.columns) <= 600:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")

    st.pyplot(fig_h)

    # 다운로드
    csv_bytes = pivot.reset_index().to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ 히트맵 피벗 CSV 다운로드",
        data=csv_bytes,
        file_name="행정동×업종_매핑_추천점수_히트맵.csv",
        mime="text/csv",
    )
