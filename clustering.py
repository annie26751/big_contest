# clustering.py
import os
import pandas as pd
from functools import lru_cache
from data_processor import FIXED_DATA_PATH

DTW_REL_PATH = "dtw_clustering.csv"

DTW_LABELS = {
    "0": {"name": "고성장 그룹", "tone": "success", "icon": "🚀",
          "intro": "DTW 기반 패턴분석 결과를 제시합니다.",
          "pattern": [
              "매출(SAA) 및 고객 수(CUS_CN): 가파른 우상향 추세가 매우 뚜렷합니다.",
              "신규 고객 비율(NEW_RAT): 초기에 매우 높았다가 점차 안정화됩니다.",
              "재방문율(REU_RAT): 꾸준히 상승하며 신규 고객 비율과 교차합니다."
          ],
          "interpretation": "새롭게 떠오르는 인기 상점 그룹입니다. 초기에 신규 고객을 성공적으로 대거 유치하고, 그 고객들을 충성 고객으로 전환시키면서 폭발적인 성장을 이뤄내고 있습니다. 가장 이상적인 성장 곡선을 그리고 있는 그룹입니다.",
          "key_industries": "커피전문점, 카페, 베이커리 등 트렌드에 민감하고 입소문을 통해 빠르게 성장할 수 있는 업종들이 주로 포함되어 있습니다."},
    "1": {"name": "안정적인 우량 그룹", "tone": "info", "icon": "💎",
          "intro": "DTW 기반 패턴분석 결과를 제시합니다.",
          "pattern": [
              "매출(SAA) 및 고객 수(CUS_CN): 높은 수준에서 큰 변동 없이 안정적으로 유지됩니다.",
              "신규 고객 비율(NEW_RAT): 낮고 안정적인 수준을 유지합니다.",
              "재방문율(REU_RAT): 매우 높은 수준에서 꾸준히 유지됩니다."
          ],
          "interpretation": "이미 자리를 잡은 핵심 우량 상점 그룹입니다. 강력한 충성 고객층을 바탕으로 꾸준하고 안정적인 실적을 내고 있습니다. 시장 변화에 크게 흔들리지 않는 저력을 가진 그룹입니다.",
          "key_industries": "한식-단품요리일반, 요리주점, 한식-육류/고기 등 단골 고객이 중요한 전통적인 외식업종이 다수 포함되어 있습니다."},
    "2": {"name": "쇠퇴/위험 그룹", "tone": "warning", "icon": "⚠️",
          "intro": "DTW 기반 패턴분석 결과를 제시합니다.",
          "pattern": [
              "매출(SAA) 및 고객 수(CUS_CN): 뚜렷한 우하향 추세입니다.",
              "신규 고객 비율(NEW_RAT): 매우 낮은 수준에 머물러 있습니다.",
              "재방문율(REU_RAT): 지속적으로 하락하고 있습니다."
          ],
          "interpretation": "경쟁력을 잃고 있는 위험 신호 그룹입니다. 신규 고객 유치에 실패하고 있으며, 기존의 충성 고객마저 잃어가고 있어 실적이 지속적으로 악화되고 있습니다. 적극적인 개입이 필요한 그룹입니다.",
          "key_industries": "커피전문점, 카페 등 경쟁이 매우 치열한 업종이 많습니다. 이는 동일 업종 내에서도 성장하는 그룹(Cluster 0)과 쇠퇴하는 그룹이 명확히 나뉨을 보여줍니다."},
    "3": {"name": "저활성/유지 그룹", "tone": "stay", "icon": "🌱",
          "intro": "DTW 기반 패턴분석 결과를 제시합니다.",
          "pattern": [
              "매출(SAA) 및 고객 수(CUS_CN): 가장 낮은 수준에서 큰 변화 없이 유지됩니다.",
              "신규 고객 비율/재방문율: 모두 낮은 수준에서 정체되어 있습니다."
          ],
          "interpretation": "소규모 현상 유지 그룹입니다. 성장을 위한 동력을 찾지 못하고 있으나, 동시에 급격한 실적 악화도 없는 상태입니다. 소수의 고정 고객에 의존하거나, 사업 규모가 매우 작은 상점들일 가능성이 높습니다.",
          "key_industries": "양식, 중식 등 다양한 업종이 포함되어 있으며, 특정 업종보다는 사업의 규모 자체가 작은 곳들이 이 그룹에 속할 것으로 추정됩니다."},
}

@lru_cache(maxsize=1)
def load_dtw_table() -> pd.DataFrame:
    base_dir = os.path.dirname(FIXED_DATA_PATH)
    csv_path = os.path.join(base_dir, DTW_REL_PATH)
    df = pd.read_csv(csv_path, encoding="euc-kr")
    df["ENCODED_MCT"] = df["ENCODED_MCT"].astype(str)
    df["dtw_cluster"] = df["dtw_cluster"].astype(str)
    return df

def get_dtw_cluster(mct_id: str) -> str | None:
    if not mct_id:
        return None
    df = load_dtw_table()
    row = df.loc[df["ENCODED_MCT"] == str(mct_id)]
    return None if row.empty else row.iloc[0]["dtw_cluster"]

def build_dtw_report(mct_id: str, merchant_name: str) -> dict:
    dtw = load_dtw_table()
    my_cluster = get_dtw_cluster(mct_id)
    label = DTW_LABELS.get(str(my_cluster), {
        "name": "미정", "tone": "secondary", "icon": "ℹ️",
        "intro": "DTW 기반 패턴분석 결과입니다.",
        "pattern": ["군집 라벨 매핑 준비 중"],
        "interpretation": "추가 데이터 필요.",
        "key_industries": "-"
    })


    intro_title = "성동구내 업장과 내 업장 비교분석 서비스 ❤️"
    intro_body = (
        "성동구 내에서 매출, 고객 수 등 변화 패턴이 비슷한 그룹들을 비교분석한 결과를 알려드릴게요. 😋<br>"
        f"점주님의 업장은 성동구 내에서 <b>[{label['name']}]</b>에 해당하네요!"
    )
    return {
        "intro_title": intro_title,
        "intro_body": intro_body,
        "cluster_badge": {"name": label["name"], "tone": label["tone"], "icon": label["icon"]},
        "notes": "DTW 기반 Clustering 결과 제시",
        "pattern": label["pattern"],
        "interpretation": label["interpretation"],
        "key_industries": label["key_industries"],
        "meta": {"model_ver": "dtw_v1", "data_source": DTW_REL_PATH}
    }
