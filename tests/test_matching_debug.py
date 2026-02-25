# -*- coding: utf-8 -*-
"""
测试匹配逻辑 - 调试脚本
"""
import sys
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logging.getLogger('citeverify').setLevel(logging.INFO)

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citeverify.extractor import CitationExtractor, CitationFormat
from citeverify.extractor import extract_references
from citeverify.models import ListingStyle, CitationFormat as RefCitationFormat
from citeverify.models.reference import ReferenceEntry
from citeverify.checkers.citation_matcher import CitationMatcher


def test_numeric_matching():
    """测试数字型匹配"""
    print("=" * 60)
    print("测试数字型匹配")
    print("=" * 60)
    
    # 模拟参考文献文本
    ref_text = """
[1] Smith, J. (2020). Introduction to Machine Learning. Nature, 100, 1-10.
[2] Brown, A., & Jones, B. (2019). Deep Learning Fundamentals. Science, 50, 20-30.
[3] Wang, L., et al. (2021). Neural Networks in Practice. IEEE, 200, 100-110.
    """
    
    # 模拟正文文本（包含引用）
    main_text = """
# Introduction

Machine learning has become increasingly important [1]. Many researchers have explored deep learning [2].
Recent advances in neural networks [3] have shown promising results.

The study by Smith [1] demonstrated that...
    """
    
    # 1. 提取参考文献
    print("\n1. 提取参考文献:")
    refs_raw = extract_references(ref_text, ListingStyle.NUMBERED, RefCitationFormat.APA)
    print(f"   提取到 {len(refs_raw)} 条参考文献")
    for ref in refs_raw:
        print(f"   {ref}")
    
    # 转换为 ReferenceEntry 对象
    refs = []
    for ref in refs_raw:
        entry = ReferenceEntry(
            full_content=ref[1],
            title=ref[2],
            title_normalized=ref[2],
            authors=ref[3].split("; ") if ref[3] else [],
            year=ref[4],
            number=ref[0],
        )
        refs.append(entry)
    
    print(f"\n   转换为 ReferenceEntry 对象: {len(refs)} 条")
    for ref in refs:
        print(f"   number={ref.number}, title={ref.title}")
    
    # 2. 提取引用
    print("\n2. 提取引用:")
    extractor = CitationExtractor()
    citations = extractor.extract_citations(main_text, CitationFormat.NUMERIC)
    print(f"   提取到 {len(citations)} 条引用")
    for c in citations:
        print(f"   number={c.number}, anchor={c.citation_anchor[:50]}...")
    
    # 3. 匹配
    print("\n3. 进行匹配:")
    matcher = CitationMatcher()
    matched, unmatched = matcher.match(refs, citations)
    
    print(f"\n   匹配结果: {len(matched)} 匹配, {len(unmatched)} 未匹配")
    
    print("\n   匹配成功的:")
    for m in matched:
        print(f"   [{m.reference_number}] {m.title[:40]}... -> anchor: {m.citation_anchor[:30]}...")
    
    print("\n   未匹配的:")
    for u in unmatched:
        print(f"   [{u.number}] {u.reason}")


def test_author_year_matching():
    """测试作者年份型匹配"""
    print("\n" + "=" * 60)
    print("测试作者年份型匹配")
    print("=" * 60)
    
    # 模拟参考文献文本 (APA格式，作者年份型)
    ref_text = """
Smith, J. (2020). Introduction to Machine Learning. Nature, 100, 1-10.
Brown, A., & Jones, B. (2019). Deep Learning Fundamentals. Science, 50, 20-30.
Wang, L., Zhang, W., & Chen, X. (2021). Neural Networks in Practice. IEEE, 200, 100-110.
    """
    
    # 模拟正文文本（包含作者年份引用）
    main_text = """
# Introduction

Machine learning has become increasingly important (Smith, 2020). Many researchers have explored deep learning (Brown & Jones, 2019).
Recent advances in neural networks (Wang et al., 2021) have shown promising results.

The study by Smith (2020) demonstrated that...
    """
    
    # 1. 提取参考文献
    print("\n1. 提取参考文献:")
    refs_raw = extract_references(ref_text, ListingStyle.AUTHOR_YEAR, RefCitationFormat.APA)
    print(f"   提取到 {len(refs_raw)} 条参考文献")
    for ref in refs_raw:
        print(f"   {ref}")
    
    # 转换为 ReferenceEntry 对象
    refs = []
    for ref in refs_raw:
        # 无编号: [全内容, 题目, 作者, 年份]
        entry = ReferenceEntry(
            full_content=ref[0],
            title=ref[1],
            title_normalized=ref[1],
            authors=ref[2].split("; ") if ref[2] else [],
            year=ref[3],
        )
        refs.append(entry)
    
    print(f"\n   转换为 ReferenceEntry 对象: {len(refs)} 条")
    for ref in refs:
        print(f"   authors={ref.authors}, year={ref.year}, title={ref.title[:30]}...")
    
    # 2. 提取引用
    print("\n2. 提取引用:")
    extractor = CitationExtractor()
    citations = extractor.extract_citations(main_text, CitationFormat.AUTHOR_YEAR)
    print(f"   提取到 {len(citations)} 条引用")
    for c in citations:
        print(f"   authors={c.authors}, year={c.year}, anchor={c.citation_anchor[:50]}...")
    
    # 3. 匹配
    print("\n3. 进行匹配:")
    matcher = CitationMatcher()
    matched, unmatched = matcher.match(refs, citations)
    
    print(f"\n   匹配结果: {len(matched)} 匹配, {len(unmatched)} 未匹配")
    
    print("\n   匹配成功的:")
    for m in matched:
        print(f"   {m.authors} ({m.year}) -> {m.title[:30]}... (score={m.match_score})")
    
    print("\n   未匹配的:")
    for u in unmatched:
        print(f"   ({u.authors}, {u.year}): {u.reason}")


if __name__ == "__main__":
    test_numeric_matching()
    test_author_year_matching()
