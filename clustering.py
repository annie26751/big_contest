# clustering.py
import os
import pandas as pd
from functools import lru_cache
from data_processor import FIXED_DATA_PATH

DTW_REL_PATH = "dtw_clustering.csv"

DTW_LABELS = {
    # 🚀 신흥 고성장 그룹 (User's Cluster 3)
    "3": {
        "name": '신흥 고성장 그룹 ("트렌드세터")',
        "tone": "success",
        "icon": "🚀",
        "intro": "DTW 기반 패턴분석 결과입니다.",
        "pattern": (
            "**패턴 (What):**\n"
            "- 유일하게 폭발적인 우상향을 그립니다.\n"
            "- 핵심은 신규 고객 비율이 초기에 높았다가 재방문율이 이를 골든 크로스하며 역전하는, 가장 이상적인 성장 곡선을 보입니다."
        ),
        "interpretation": (
            "**세부 분석 (Why):**\n"
            "- **핵심 고객:** 30대 여성 비율과 10-20대 여성 비율이 4개 군집 중 가장 높습니다.\n"
            "- **고객 유형:** 유동인구 고객 비율이 압도적으로 높습니다.\n"
            "- **운영 전략:** 배달 매출 비율이 4개 군집 중 가장 높게 나타납니다.\n\n"
            '**결론 (Persona): "트렌드를 선도하는 디지털 네이티브 상점"**\n\n'
            "이 그룹은 30대 이하 여성을 핵심 타겟으로, 인스타그램 등 SNS 마케팅을 통해 외부 유동인구를 신규 고객으로 대거 유치하는 데 성공했습니다. "
            "단순 유입에 그치지 않고, 차별화된 맛과 경험을 제공하여 이들을 충성 고객(재방문)으로 전환시켰으며, 배달(디지털 전환)에도 가장 적극적으로 대응하여 성장을 극대화하고 있습니다."
        ),
        "key_industries": (
            "**업종 (Who):**\n"
            "커피전문점, 카페, 베이커리 등 트렌드와 SNS 바이럴에 가장 민감한 업종이 압도적으로 많습니다."
        )
    },
    # 💎 압도적인 초우량 그룹 (User's Cluster 0)
    "0": {
        "name": '신흥 고성장 그룹 ("트렌드세터")',
        "tone": "info",
        "icon": "💎",
        "intro": "DTW 기반 패턴분석 결과입니다.",
        "pattern": (
            "**패턴 (What):**\n"
            "- 다른 모든 그룹을 압도하는 가장 높은 수준의 매출과 고객 수를 안정적으로 유지합니다.\n"
            "- 재방문율(REU_RAT)이 극도로 높아 강력한 '경제적 해자'를 구축했습니다."
        ),
        "interpretation": (
            "**세부 분석 (Why):**\n"
            "- **핵심 고객:** 모든 인구통계가 고르게 발달해 있습니다. 특정 연령/성별에 치우치지 않고 전 고객층을 흡수합니다.\n"
            "- **고객 유형:** 유동인구, 거주민, 직장인 비율이 모두 높은 수준에서 균형을 이룹니다.\n"
            "- **운영 전략:** 배달 매출 비율이 4개 군집 중 가장 높게 나타납니다.\n\n"
            '**결론 (Persona): "상권을 대표하는 대체 불가능한 브랜드"**\n\n'
            "이 그룹은 특정 업종이나 고객층에 의존하지 않습니다. 이미 '상권을 대표하는 랜드마크' 또는 '대체 불가능한 브랜드'로 자리 잡아, 모든 유형의 고객(거주/직장/유동)과 모든 연령층을 끌어모읍니다. 이미 경쟁의 단계를 초월하여 시장 자체를 지배하고 있는 그룹입니다."
        ),
        "key_industries": (
            "**업종 (Who):**\n"
            "커피전문점, 한식-단품요리일반, 포장마차, 청과물 등 업종을 가리지 않고 다양한 분야의 1등 상점들이 포진해 있습니다."
        )
    },

    # ⚠️ 쇠퇴/경쟁 도태 그룹 (User's Cluster 2)
    "2": {
        "name": '쇠퇴/경쟁 도태 그룹 ("위기의 상점")',
        "tone": "warning",
        "icon": "⚠️",
        "intro": "DTW 기반 패턴분석 결과입니다.",
        "pattern": (
            "**패턴 (What):**\n"
            "- 4개 그룹 중 유일하게 매출과 고객 수가 뚜렷한 우하향을 그립니다.\n"
            "- 재방문율과 신규 유입률 모두 활력을 잃고 동반 하락합니다."
        ),
        "interpretation": (
            "**세부 분석 (Why):**\n"
            "- **핵심 고객:** Cluster 3과 비교했을 때, 핵심 타겟인 30대 여성 비율이 현저히 낮습니다. 반면, 고객층이 불명확하고 분산되어 있습니다.\n"
            "- **고객 유형:** Cluster 3보다 유동인구 비율이 낮고, 그렇다고 Cluster 1처럼 거주민/직장인 비율이 높지도 않은, 어중간한 고객 구성을 보입니다.\n"
            "- **운영 전략:** 배달 매출 비율이 Cluster 3(고성장 그룹)보다 훨씬 낮습니다.\n\n"
            '**결론 (Persona): "트렌드와 디지털 전환에 모두 실패한 상점"**\n\n'
            "이 그룹은 '어떻게' 실패했는지를 보여줍니다. Cluster 3과 같은 커피/카페 업종임에도 불구하고, (1) 트렌드를 주도하는 30대 여성 핵심 고객을 빼앗기고, (2) 배달과 같은 디지털 전환에 대응하지 못했으며, (3) Cluster 1처럼 확실한 지역 단골을 확보하지도 못하면서 양쪽 모두에게서 경쟁력을 잃고 쇠퇴하고 있습니다."
        ),
        "key_industries": (
            "**업종 (Who):**\n"
            "커피전문점, 카페, 한식-단품요리일반이 많습니다. 신흥 고성장 그룹과 업종이 겹친다는 점이 핵심입니다."
        )
    },
    # 🏦 안정적인 중견 그룹 (User's Cluster 1)
    "1": {
        "name": '안정적인 중견 그룹 ("지역 터줏대감")',
        "tone": "primary",
        "icon": "🏦",
        "intro": "DTW 기반 패턴분석 결과입니다.",
        "pattern": (
            "**패턴 (What):**\n"
            "- 중간 수준 이상의 매출과 고객 수를 큰 변동 없이 꾸준히 유지합니다.\n"
            "- 재방문율이 신규 유입률보다 일관되게 높아, 이미 안정적인 궤도에 올랐음을 보여줍니다."
        ),
        "interpretation": (
            "**세부 분석 (Why):**\n"
            "- **핵심 고객:** 40대 남성 비율과 50대 남성 비율이 4개 군집 중 가장 높습니다.\n"
            "- **고객 유형:** 지역 거주민 비율과 직장인 고객 비율이 매우 높게 나타납니다.\n"
            "- **운영 전략:** 배달 매출 비율은 상대적으로 낮습니다.\n\n"
            '**결론 (Persona): "직장인/거주민 단골을 확보한 로컬 맛집"**\n\n'
            "이 그룹은 40-50대 남성을 중심으로 한 지역 직장인과 거주민을 핵심 고객으로 확보하고 있습니다. "
            "트렌드나 유동인구보다는, 점심/저녁 식사 및 회식을 위한 '단골 장사'에 특화되어 있습니다. 유행과 무관하게 강력한 로컬 고객층을 기반으로 안정적인 실적을 내는 상권의 '허리'입니다."
        ),
        "key_industries": (
            "**업종 (Who):**\n"
            "한식-단품요리일반, 요리주점, 한식-육류/고기 등 지역 기반의 전통적인 외식업이 주를 이룹니다."
        )
    },
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

    image_path = f"./data/plots/dtw_cluster_{my_cluster}.png"
    images = [image_path] if os.path.exists(image_path) else []

    return {
        "intro_title": intro_title,
        "intro_body": intro_body,
        "cluster_badge": {"name": label["name"], "tone": label["tone"], "icon": label["icon"]},
        "notes": "DTW 기반 Clustering 결과 제시",
        "pattern": label["pattern"],
        "interpretation": label["interpretation"],
        "key_industries": label["key_industries"],
        "images": images,
        "meta": {"model_ver": "dtw_v1", "data_source": DTW_REL_PATH}
    }
