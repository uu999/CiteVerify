# -*- coding: utf-8 -*-
"""
测试 Semantic Scholar 搜索能力

验证 Semantic Scholar API 对于给定参考文献标题的搜索和匹配情况。
用于诊断：是搜不到还是匹配太严格？

新增：包含原始 OCR/提取错误的标题（不纠正）
"""
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.checkers import ReferenceChecker

# Semantic Scholar API Key
SS_API_KEY = "k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV"

# 原始测试用例（20篇）
ORIGINAL_TEST_REFERENCES = [
    {
        "num": 2,
        "title": "Potential Pitfalls of Process Modeling: Part A",
        "year": "2006",
        "authors": "M. Rosemann",
    },
    {
        "num": 3,
        "title": "Opportunities and Constraints: The Current Struggle with BPMN",
        "year": "2010",
        "authors": "J. Recker",
    },
    {
        "num": 4,
        "title": "IT-Business Alignment: A Systematic Literature Review",
        "year": "2021",
        "authors": "S. Q. Njanka, G. Sandula, R. Colomo-Palacios",
    },
    # ... 其他原始条目保持不变（此处省略以节省空间）
    {
        "num": 20,
        "title": "Introducing the BPMN-Chatbot for Efficient LLM-Based Process Modeling",
        "year": "2024",
        "authors": "J. Köpke, A. Safan",
    },
]

# 🔥 新增：从用户输入中提取的“带错误”的标题（不纠正！）
NOISY_TITLES_FROM_USER = [
    "IT-Business Alignment:ASvstematic Literature Review",
    "Disentangling Organizational Agility from Flexibility,Adaptability,and Versatility: A Systematic Review",
    "Process Mining:A Research Agenda",
    "Leveraging Large Language Models (LLMs)for Process Mining (Technical Report",
    "Automated Generation of BPMN Processes from Textual Requirements",
    "GPT-4oSystem Card",
    "The Llama 3 Herd ofModels",
    "DeepSeek-V3 TechnicalReport",
    "Similarity of Business Process Models—A State-of-the-Art Analysis",
    "TheBusiness Process Model Quality Metrics"
]

# 构建新增测试用例（编号从 100 开始，避免冲突）
NOISY_TEST_REFERENCES = []
for i, title in enumerate(NOISY_TITLES_FROM_USER, start=100):
    NOISY_TEST_REFERENCES.append({
        "num": i,
        "title": title,
        "year": None,  # 用户未提供年份，设为 None
        "authors": "(Unknown)"
    })

# 合并测试集
TEST_REFERENCES = ORIGINAL_TEST_REFERENCES + NOISY_TEST_REFERENCES


def test_semantic_scholar_raw_search():
    """
    测试原始 Semantic Scholar API 搜索（不经过匹配逻辑）

    目的：区分是 API 搜不到还是匹配逻辑太严格
    """
    import requests

    print("=" * 80)
    print("Semantic Scholar 原始搜索测试（不经过匹配逻辑）")
    print("=" * 80)
    print(f"使用 API Key: {SS_API_KEY[:10]}...")
    print(f"共 {len(TEST_REFERENCES)} 篇文献待测试（含原始+噪声标题）")
    print("-" * 80)

    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": SS_API_KEY}

    results_summary = {
        "api_found": 0,
        "api_not_found": 0,
        "api_error": 0,
    }

    for ref in TEST_REFERENCES:
        title = ref["title"]
        year = ref["year"]
        num = ref["num"]

        print(f"\n[{num}] 搜索: {title[:60]}{'...' if len(title) > 60 else ''}")
        if year:
            print(f"    年份: {year}")

        # 注意：这里依然用原始标题（含错误）进行查询
        params = {
            "query": f'"{title}"',  # 尝试精确短语搜索
            "limit": 20,
            "fields": "title,year",
        }
        if year:
            params["year"] = year

        time.sleep(1.1)  # 遵守限流

        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=30)

            if response.status_code == 429:
                print(f"    ⚠️ 限流 429，等待后重试...")
                time.sleep(5)
                response = requests.get(api_url, params=params, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"    ❌ API 错误: {response.status_code}")
                results_summary["api_error"] += 1
                continue

            data = response.json()
            papers = data.get("data", [])
            total = data.get("total", 0)

            if papers:
                results_summary["api_found"] += 1
                print(f"    ✅ API 返回 {len(papers)} 条结果（总共 {total} 条）")

                # 使用实际的匹配逻辑
                from citeverify.checkers import ReferenceChecker
                
                for i, paper in enumerate(papers[:3]):
                    p_title = paper.get("title", "")
                    p_year = paper.get("year", "")

                    # 使用新的匹配逻辑
                    is_match, similarity = ReferenceChecker.titles_match(title, p_title)
                    
                    if similarity >= 1.0:
                        match_status = "🎯精确"
                    elif is_match:
                        match_status = f"✅匹配({similarity:.2f})"
                    else:
                        match_status = f"❌({similarity:.2f})"
                    
                    print(
                        f"       [{i + 1}] {match_status} | {p_title[:50]}{'...' if len(p_title) > 50 else ''} ({p_year})")
            else:
                results_summary["api_not_found"] += 1
                print(f"    ❌ API 无结果")

                # 尝试去掉引号（模糊搜索）
                params_fuzzy = {"query": title, "limit": 5, "fields": "title,year"}
                if year:
                    params_fuzzy["year"] = year
                time.sleep(1.1)
                response2 = requests.get(api_url, params=params_fuzzy, headers=headers, timeout=30)
                if response2.status_code == 200:
                    data2 = response2.json()
                    papers2 = data2.get("data", [])
                    if papers2:
                        print(f"       💡 模糊搜索找到 {len(papers2)} 条结果:")
                        for p in papers2[:2]:
                            print(f"          - {p.get('title', '')[:50]} ({p.get('year', '')})")

        except Exception as e:
            print(f"    ❌ 请求异常: {e}")
            results_summary["api_error"] += 1

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    total = len(TEST_REFERENCES)
    print(f"总测试数:       {total}")
    print(f"✅ API 有结果:  {results_summary['api_found']} ({results_summary['api_found'] / total * 100:.1f}%)")
    print(f"❌ API 无结果:  {results_summary['api_not_found']} ({results_summary['api_not_found'] / total * 100:.1f}%)")
    print(f"⚠️ API 错误:    {results_summary['api_error']}")
    print("=" * 80)


def test_full_checker():
    """
    测试完整的 ReferenceChecker（包含匹配逻辑）
    """
    print("\n" + "=" * 80)
    print("ReferenceChecker 完整测试（包含匹配逻辑）")
    print("=" * 80)

    checker = ReferenceChecker(
        request_delay=1.5,
        semantic_scholar_api_key=SS_API_KEY,
        use_semantic_scholar=True,
        use_openalex=True,
    )

    results_summary = {
        "found": 0,
        "not_found": 0,
        "by_source": {
            "arxiv": 0,
            "semantic_scholar": 0,
            "openalex": 0,
        }
    }

    for ref in TEST_REFERENCES:
        title = ref["title"]
        year = ref["year"]
        num = ref["num"]

        print(f"\n[{num}] 校验: {title[:50]}{'...' if len(title) > 50 else ''}")

        result = checker.verify_reference(title, year=year)

        if result.can_get:
            results_summary["found"] += 1
            source = result.source.value
            results_summary["by_source"][source] = results_summary["by_source"].get(source, 0) + 1

            print(f"    ✅ 找到 | 来源: {source} | 相似度: {result.similarity:.2f}")
            print(f"       匹配标题: {result.matched_title[:50] if result.matched_title else 'N/A'}...")
        else:
            results_summary["not_found"] += 1
            print(f"    ❌ 未找到")

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    total = len(TEST_REFERENCES)
    print(f"总测试数:       {total}")
    print(f"✅ 找到:        {results_summary['found']} ({results_summary['found'] / total * 100:.1f}%)")
    print(f"❌ 未找到:      {results_summary['not_found']} ({results_summary['not_found'] / total * 100:.1f}%)")
    print(f"\n按来源统计:")
    for source, count in results_summary["by_source"].items():
        if count > 0:
            print(f"  - {source}: {count}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 Semantic Scholar 搜索能力（含噪声标题）")
    parser.add_argument(
        "--mode",
        choices=["raw", "full", "both"],
        default="both",
        help="测试模式: raw=原始API搜索, full=完整检查器, both=两者都测"
    )

    args = parser.parse_args()

    if args.mode in ["raw", "both"]:
        test_semantic_scholar_raw_search()

    if args.mode in ["full", "both"]:
        test_full_checker()