# -*- coding: utf-8 -*-
"""
测试：参考文献与引用文本匹配

测试数字型和作者年份型的匹配功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.models.reference import ReferenceEntry, ListingStyle, CitationFormat
from citeverify.extractor.citation_extractor import (
    CitationExtractor,
    CitationFormat as CitationFormatEnum,
    NumericCitation,
    AuthorYearCitation,
    CitationPosition,
)
from citeverify.checkers.citation_matcher import (
    CitationMatcher,
    MatchedCitation,
    UnmatchedCitation,
    MatchType,
    match_citations,
    match_citations_to_list,
)


def create_sample_references_numeric():
    """创建数字型参考文献样本"""
    references = [
        ReferenceEntry(
            full_content="[1] Smith J. Deep Learning. Nature. 2020.",
            title="Deep Learning",
            authors=["Smith J"],
            year="2020",
            number=1,
        ),
        ReferenceEntry(
            full_content="[2] Brown A, Lee B. Machine Learning Methods. ICML. 2019.",
            title="Machine Learning Methods",
            authors=["Brown A", "Lee B"],
            year="2019",
            number=2,
        ),
        ReferenceEntry(
            full_content="[3] Zhang W et al. Natural Language Processing. ACL. 2021.",
            title="Natural Language Processing",
            authors=["Zhang W", "Wang X", "Liu Y"],
            year="2021",
            number=3,
        ),
        ReferenceEntry(
            full_content="[4] Chen et al. Transformer Models. NeurIPS. 2022.",
            title="Transformer Models",
            authors=["Chen H", "Li M"],
            year="2022",
            number=4,
        ),
        ReferenceEntry(
            full_content="[5] 无标题参考文献",
            title=None,  # 无标题
            authors=[],
            year="2023",
            number=5,
        ),
    ]
    return references


def create_sample_references_author_year():
    """创建作者年份型参考文献样本"""
    references = [
        ReferenceEntry(
            full_content="Smith, J. (2020). Deep Learning. Nature.",
            title="Deep Learning",
            authors=["Smith J"],
            year="2020",
        ),
        ReferenceEntry(
            full_content="Brown, A., & Lee, B. (2019). Machine Learning Methods. ICML.",
            title="Machine Learning Methods",
            authors=["Brown A", "Lee B"],
            year="2019",
        ),
        ReferenceEntry(
            full_content="Zhang, W., Wang, X., & Liu, Y. (2021). Natural Language Processing. ACL.",
            title="Natural Language Processing",
            authors=["Zhang W", "Wang X", "Liu Y"],
            year="2021",
        ),
        ReferenceEntry(
            full_content="Chen, H., & Li, M. (2022). Transformer Models. NeurIPS.",
            title="Transformer Models",
            authors=["Chen H", "Li M"],
            year="2022",
        ),
    ]
    return references


def create_sample_numeric_citations():
    """创建数字型引用样本"""
    citations = [
        NumericCitation(
            position=CitationPosition("Introduction", 0, (50, 53)),
            number=1,
            citation_anchor="Deep learning has achieved great success [1].",
            context="Background. Deep learning has achieved great success [1]. Many applications.",
        ),
        NumericCitation(
            position=CitationPosition("Introduction", 1, (30, 33)),
            number=2,
            citation_anchor="Machine learning methods [2] are widely used.",
            context="Deep learning. Machine learning methods [2] are widely used. In this paper.",
        ),
        NumericCitation(
            position=CitationPosition("Methods", 0, (20, 23)),
            number=1,  # 同一文献在不同位置被引用
            citation_anchor="Following Smith [1], we propose.",
            context="Methods section. Following Smith [1], we propose. Our approach.",
        ),
        NumericCitation(
            position=CitationPosition("Results", 0, (40, 43)),
            number=5,  # 对应无标题的参考文献
            citation_anchor="As shown in previous work [5].",
            context="Results section. As shown in previous work [5]. This confirms.",
        ),
        NumericCitation(
            position=CitationPosition("Conclusion", 0, (25, 29)),
            number=10,  # 不存在的参考文献编号
            citation_anchor="According to [10], future work.",
            context="Conclusion. According to [10], future work. End.",
        ),
    ]
    return citations


def create_sample_author_year_citations():
    """创建作者年份型引用样本"""
    citations = [
        AuthorYearCitation(
            position=CitationPosition("Introduction", 0, (30, 43)),
            authors="Smith",
            year="2020",
            citation_anchor="Deep learning (Smith, 2020) has achieved success.",
            context="Background. Deep learning (Smith, 2020) has achieved success. Many applications.",
        ),
        AuthorYearCitation(
            position=CitationPosition("Introduction", 1, (20, 40)),
            authors="Brown & Lee",
            year="2019",
            citation_anchor="Brown & Lee (2019) proposed methods.",
            context="Introduction. Brown & Lee (2019) proposed methods. These methods.",
        ),
        AuthorYearCitation(
            position=CitationPosition("Methods", 0, (15, 35)),
            authors="Zhang et al.",
            year="2021",
            citation_anchor="Zhang et al. (2021) showed that.",
            context="Methods section. Zhang et al. (2021) showed that. We follow.",
        ),
        AuthorYearCitation(
            position=CitationPosition("Results", 0, (10, 25)),
            authors="Smith",
            year="2020",  # 同一文献在不同位置被引用
            citation_anchor="As Smith (2020) noted.",
            context="Results. As Smith (2020) noted. This confirms.",
        ),
        AuthorYearCitation(
            position=CitationPosition("Discussion", 0, (20, 35)),
            authors="Unknown",
            year="2025",  # 不存在的作者和年份
            citation_anchor="Unknown (2025) suggested.",
            context="Discussion. Unknown (2025) suggested. Future work.",
        ),
    ]
    return citations


def test_numeric_matching():
    """测试数字型匹配"""
    print("=" * 70)
    print("数字型匹配测试")
    print("=" * 70)
    
    references = create_sample_references_numeric()
    citations = create_sample_numeric_citations()
    
    matcher = CitationMatcher()
    matched, unmatched = matcher.match_numeric(references, citations)
    
    print(f"\n参考文献数量: {len(references)}")
    print(f"引用数量: {len(citations)}")
    print(f"匹配成功: {len(matched)}")
    print(f"匹配失败: {len(unmatched)}")
    
    print("\n--- 匹配成功的引用 ---")
    for m in matched:
        print(f"  [{m.reference_number}] {m.title[:30]}...")
        print(f"      引用句: {m.citation_anchor[:50]}...")
        print(f"      位置: {m.citation_position.title}, 句子{m.citation_position.sentence_id}")
        print(f"      匹配度: {m.match_score}")
        print()
    
    print("--- 匹配失败的引用 ---")
    for u in unmatched:
        print(f"  编号 [{u.number}]: {u.reason}")
        print(f"      引用句: {u.citation_anchor[:50]}...")
        print()
    
    # 验证结果
    assert len(matched) == 3, f"期望3个匹配，实际{len(matched)}"
    assert len(unmatched) == 2, f"期望2个未匹配，实际{len(unmatched)}"
    print("✅ 数字型匹配测试通过")


def test_author_year_matching():
    """测试作者年份型匹配"""
    print("\n" + "=" * 70)
    print("作者年份型匹配测试")
    print("=" * 70)
    
    references = create_sample_references_author_year()
    citations = create_sample_author_year_citations()
    
    matcher = CitationMatcher()
    matched, unmatched = matcher.match_author_year(references, citations, min_score=0.3)
    
    print(f"\n参考文献数量: {len(references)}")
    print(f"引用数量: {len(citations)}")
    print(f"匹配成功: {len(matched)}")
    print(f"匹配失败: {len(unmatched)}")
    
    print("\n--- 匹配成功的引用 ---")
    for m in matched:
        print(f"  [{m.authors}] ({m.year}) -> {m.title[:30]}...")
        print(f"      引用句: {m.citation_anchor[:50]}...")
        print(f"      位置: {m.citation_position.title}, 句子{m.citation_position.sentence_id}")
        print(f"      匹配度: {m.match_score}")
        print()
    
    print("--- 匹配失败的引用 ---")
    for u in unmatched:
        print(f"  {u.authors} ({u.year}): {u.reason}")
        print(f"      引用句: {u.citation_anchor[:50]}...")
        print()
    
    # 验证结果
    assert len(matched) >= 3, f"期望至少3个匹配，实际{len(matched)}"
    print("✅ 作者年份型匹配测试通过")


def test_author_normalization():
    """测试作者名称标准化"""
    print("\n" + "=" * 70)
    print("作者名称标准化测试")
    print("=" * 70)
    
    matcher = CitationMatcher()
    
    test_cases = [
        ("Smith et al.", "smith"),
        ("Brown & Lee", "brown lee"),
        ("Zhang, Wang, and Liu", "zhang wang liu"),
        ("Chen H., Li M.", "chen h li m"),
        ("张三等", "张三"),
        ("李四、王五", "李四 王五"),
    ]
    
    print("\n作者标准化结果:")
    for original, expected in test_cases:
        normalized = matcher.normalize_author(original)
        status = "✅" if normalized == expected else "❌"
        print(f"  {status} '{original}' -> '{normalized}' (期望: '{expected}')")
    
    print("\n✅ 作者标准化测试完成")


def test_author_similarity():
    """测试作者相似度计算"""
    print("\n" + "=" * 70)
    print("作者相似度测试")
    print("=" * 70)
    
    matcher = CitationMatcher()
    
    test_cases = [
        # (引用作者, 参考文献作者, 期望相似度范围)
        ("Smith", ["Smith J"], (0.5, 1.0)),
        ("Smith et al.", ["Smith J", "Brown A", "Lee B"], (0.3, 1.0)),
        ("Brown & Lee", ["Brown A", "Lee B"], (0.5, 1.0)),
        ("Zhang", ["Zhang W", "Wang X"], (0.3, 1.0)),
        ("Unknown", ["Smith J"], (0.0, 0.1)),
    ]
    
    print("\n作者相似度结果:")
    for cite_author, ref_authors, (min_sim, max_sim) in test_cases:
        similarity = matcher.calculate_author_similarity(cite_author, ref_authors)
        status = "✅" if min_sim <= similarity <= max_sim else "❌"
        print(f"  {status} '{cite_author}' vs {ref_authors}")
        print(f"      相似度: {similarity:.3f} (期望范围: {min_sim}-{max_sim})")
    
    print("\n✅ 作者相似度测试完成")


def test_match_to_list():
    """测试列表格式输出"""
    print("\n" + "=" * 70)
    print("列表格式输出测试")
    print("=" * 70)
    
    references = create_sample_references_numeric()
    citations = create_sample_numeric_citations()
    
    result = match_citations_to_list(references, citations)
    
    print(f"\n输出列表长度: {len(result)}")
    print("\n前3条结果:")
    for i, row in enumerate(result[:3], 1):
        title, authors, year, abstract, pdf_url, anchor, context, score = row
        print(f"  [{i}] {title[:30]}...")
        print(f"      作者: {authors}")
        print(f"      年份: {year}")
        print(f"      匹配度: {score}")
        print()
    
    print("✅ 列表格式输出测试完成")


def test_full_pipeline():
    """测试完整流程：文本 -> 引用提取 -> 匹配"""
    print("\n" + "=" * 70)
    print("完整流程测试")
    print("=" * 70)
    
    # 模拟 Markdown 文本
    markdown_text = """
# Introduction

Deep learning has revolutionized many fields [1]. Recent work by Smith (2020) 
showed promising results. Machine learning methods [2] are widely adopted.

# Methods

Following the approach of Zhang et al. (2021), we propose a new method [3].
Our implementation builds on [1][2].

# Results

As shown in previous studies [4], transformer models achieve state-of-the-art.
Brown & Lee (2019) confirmed these findings.

# Conclusion

In conclusion, deep learning [1] continues to advance rapidly.
"""
    
    # 提取引用
    extractor = CitationExtractor()
    
    # 数字型引用
    numeric_citations = extractor.extract_citations(
        markdown_text, 
        CitationFormatEnum.NUMERIC
    )
    
    # 作者年份型引用
    author_year_citations = extractor.extract_citations(
        markdown_text,
        CitationFormatEnum.AUTHOR_YEAR
    )
    
    print(f"\n提取到的数字型引用: {len(numeric_citations)}")
    print(f"提取到的作者年份型引用: {len(author_year_citations)}")
    
    # 创建参考文献
    references = create_sample_references_numeric()
    
    # 匹配数字型
    matcher = CitationMatcher()
    matched_numeric, unmatched_numeric = matcher.match_numeric(
        references, numeric_citations
    )
    
    print(f"\n数字型匹配结果:")
    print(f"  成功: {len(matched_numeric)}")
    print(f"  失败: {len(unmatched_numeric)}")
    
    # 匹配作者年份型
    references_ay = create_sample_references_author_year()
    matched_ay, unmatched_ay = matcher.match_author_year(
        references_ay, author_year_citations, min_score=0.3
    )
    
    print(f"\n作者年份型匹配结果:")
    print(f"  成功: {len(matched_ay)}")
    print(f"  失败: {len(unmatched_ay)}")
    
    print("\n✅ 完整流程测试完成")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试引用匹配功能")
    parser.add_argument(
        "--test",
        choices=["numeric", "author_year", "normalize", "similarity", "list", "pipeline", "all"],
        default="all",
        help="测试类型"
    )
    
    args = parser.parse_args()
    
    if args.test in ["numeric", "all"]:
        test_numeric_matching()
    
    if args.test in ["author_year", "all"]:
        test_author_year_matching()
    
    if args.test in ["normalize", "all"]:
        test_author_normalization()
    
    if args.test in ["similarity", "all"]:
        test_author_similarity()
    
    if args.test in ["list", "all"]:
        test_match_to_list()
    
    if args.test in ["pipeline", "all"]:
        test_full_pipeline()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
