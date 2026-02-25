# -*- coding: utf-8 -*-
"""
测试 API 能否正常获取摘要
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citeverify.checkers.reference_checker import ReferenceChecker


def test_arxiv():
    """测试 arXiv API"""
    print("=" * 60)
    print("测试 arXiv API")
    print("=" * 60)
    
    checker = ReferenceChecker(request_delay=0.5)
    
    # 使用一个知名的 arXiv 论文
    title = "Attention Is All You Need"
    print(f"\n搜索标题: {title}")
    
    result = checker.search_arxiv(title)
    
    if result:
        print(f"  找到: {result.get('title')}")
        print(f"  相似度: {result.get('similarity')}")
        print(f"  PDF URL: {result.get('pdf_url')}")
        abstract = result.get('abstract', '')
        if abstract:
            print(f"  摘要 (前200字): {abstract[:200]}...")
        else:
            print("  摘要: 未获取到!")
    else:
        print("  未找到!")


def test_semantic_scholar():
    """测试 Semantic Scholar API"""
    print("\n" + "=" * 60)
    print("测试 Semantic Scholar API")
    print("=" * 60)
    
    checker = ReferenceChecker(
        request_delay=1.0,
        semantic_scholar_api_key="k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV"  # 用户提供的 key
    )
    
    title = "Attention Is All You Need"
    print(f"\n搜索标题: {title}")
    
    result = checker.search_semantic_scholar(title, year="2017")
    
    if result:
        print(f"  找到: {result.get('title')}")
        print(f"  相似度: {result.get('similarity')}")
        print(f"  PDF URL: {result.get('pdf_url')}")
        abstract = result.get('abstract', '')
        if abstract:
            print(f"  摘要 (前200字): {abstract[:200]}...")
        else:
            print("  摘要: 未获取到!")
    else:
        print("  未找到!")


def test_openalex():
    """测试 OpenAlex API"""
    print("\n" + "=" * 60)
    print("测试 OpenAlex API")
    print("=" * 60)
    
    checker = ReferenceChecker(request_delay=0.5)
    
    title = "Attention Is All You Need"
    
    # 测试不带年份
    print(f"\n搜索标题 (不带年份): {title}")
    result = checker.search_openalex(title)
    
    if result:
        print(f"  找到: {result.get('title')}")
        print(f"  相似度: {result.get('similarity')}")
        print(f"  PDF URL: {result.get('pdf_url')}")
        abstract = result.get('abstract', '')
        if abstract:
            print(f"  摘要 (前200字): {abstract[:200]}...")
        else:
            print("  摘要: 未获取到!")
    else:
        print("  未找到!")
    
    # 测试带年份
    print(f"\n搜索标题 (带年份 2017): {title}")
    result2 = checker.search_openalex(title, year="2017")
    
    if result2:
        print(f"  找到: {result2.get('title')}")
        print(f"  相似度: {result2.get('similarity')}")
        print(f"  PDF URL: {result2.get('pdf_url')}")
        abstract = result2.get('abstract', '')
        if abstract:
            print(f"  摘要 (前200字): {abstract[:200]}...")
        else:
            print("  摘要: 未获取到!")
    else:
        print("  未找到 (带年份过滤)!")


def test_verify_reference():
    """测试完整的校验流程"""
    print("\n" + "=" * 60)
    print("测试完整校验流程")
    print("=" * 60)
    
    checker = ReferenceChecker(
        request_delay=1.0,
        semantic_scholar_api_key="k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV"
    )
    
    titles = [
        ("Attention Is All You Need", "2017"),
        ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "2019"),
        ("Deep Residual Learning for Image Recognition", "2016"),
    ]
    
    for title, year in titles:
        print(f"\n校验: {title}")
        result = checker.verify_reference(title, year=year)
        print(f"  can_get: {result.can_get}")
        print(f"  source: {result.source.value}")
        print(f"  matched_title: {result.matched_title}")
        print(f"  similarity: {result.similarity}")
        print(f"  pdf_url: {result.pdf_url}")
        if result.abstract:
            print(f"  abstract (前100字): {result.abstract[:100]}...")
        else:
            print(f"  abstract: 未获取到!")


if __name__ == "__main__":
    test_arxiv()
    test_semantic_scholar()
    test_openalex()
    test_verify_reference()
