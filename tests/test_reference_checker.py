# -*- coding: utf-8 -*-
"""
测试：参考文献真伪校验

使用 arXiv、Semantic Scholar 和 OpenAlex API 验证参考文献真实性。
搜索优先级：arXiv -> Semantic Scholar -> OpenAlex
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.checkers import (
    ReferenceChecker,
    verify_references,
    verify_single_reference,
    SearchSource,
)

# Semantic Scholar API Key
SS_API_KEY = "k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV"


def test_single_reference():
    """测试单条参考文献校验"""
    print("=" * 60)
    print("单条参考文献校验测试")
    print("=" * 60)
    
    # 测试标题列表
    test_titles = [
        "Attention is all you need",
        "BERT: Pre-training of deep bidirectional transformers",
        "Language models are few-shot learners",
        "这是一个不存在的论文标题12345",
    ]
    
    for title in test_titles:
        print(f"\n🔍 搜索: {title}")
        result = verify_single_reference(title, semantic_scholar_api_key=SS_API_KEY)
        
        if result.can_get:
            print(f"  ✅ 找到!")
            print(f"     来源: {result.source.value}")
            print(f"     匹配标题: {result.matched_title}")
            print(f"     相似度: {result.similarity:.2f}")
            print(f"     PDF: {result.pdf_url or '无'}")
            if result.abstract:
                abstract_preview = result.abstract[:150] + "..." if len(result.abstract) > 150 else result.abstract
                print(f"     摘要: {abstract_preview}")
        else:
            print(f"  ❌ 未找到")
            if result.error:
                print(f"     错误: {result.error}")
    
    print("\n" + "=" * 60)


def test_batch_references():
    """测试批量参考文献校验"""
    print("=" * 60)
    print("批量参考文献校验测试")
    print("=" * 60)
    
    # 模拟提取的参考文献列表（有编号格式）
    # [编号, 全文, 标题, 作者, 年份]
    references = [
        [1, "Vaswani A. et al. Attention is all you need...", "Attention is all you need", "Vaswani A", "2017"],
        [2, "Devlin J. et al. BERT...", "BERT: Pre-training of deep bidirectional transformers for language understanding", "Devlin J", "2019"],
        [3, "Brown T. et al. GPT-3...", "Language models are few-shot learners", "Brown T", "2020"],
        [4, "Fake Paper...", "This is a completely fake paper title that does not exist", "Nobody", "2099"],
    ]
    
    print(f"\n输入 {len(references)} 条参考文献")
    print("-" * 60)
    
    results = verify_references(
        references,
        has_number=True,
        request_delay=1.5,
        semantic_scholar_api_key=SS_API_KEY,
        verbose=True,
    )
    
    print("\n校验结果:")
    print("-" * 60)
    
    for ref in results:
        # [编号, 全文, 标题, 作者, 年份, can_get, abstract, pdf_url]
        num = ref[0]
        title = ref[2]
        can_get = ref[-3]
        abstract = ref[-2]
        pdf_url = ref[-1]
        
        status = "✅" if can_get else "❌"
        print(f"  [{num}] {status} {title[:50]}...")
        if can_get:
            print(f"      PDF: {pdf_url or '无'}")
    
    print("\n" + "=" * 60)
    return results


def test_no_number_references():
    """测试无编号格式的参考文献"""
    print("=" * 60)
    print("无编号参考文献校验测试")
    print("=" * 60)
    
    # 无编号格式：[全文, 标题, 作者, 年份]
    references = [
        ["Vaswani A. et al. Attention is all you need...", "Attention is all you need", "Vaswani A", "2017"],
        ["LeCun Y. et al. Deep learning...", "Deep learning", "LeCun Y", "2015"],
    ]
    
    results = verify_references(
        references,
        has_number=False,
        request_delay=1.5,
        semantic_scholar_api_key=SS_API_KEY,
        verbose=True,
    )
    
    print("\n校验结果:")
    for ref in results:
        # [全文, 标题, 作者, 年份, can_get, abstract, pdf_url]
        title = ref[1]
        can_get = ref[-3]
        status = "✅" if can_get else "❌"
        print(f"  {status} {title}")
    
    print("=" * 60)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="参考文献真伪校验测试")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "no_number", "all"],
        default="all",
        help="测试模式"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="单条测试时的标题"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single" or args.title:
        if args.title:
            result = verify_single_reference(args.title)
            print(f"标题: {args.title}")
            print(f"找到: {result.can_get}")
            print(f"来源: {result.source.value}")
            print(f"PDF: {result.pdf_url}")
        else:
            test_single_reference()
    elif args.mode == "batch":
        test_batch_references()
    elif args.mode == "no_number":
        test_no_number_references()
    else:
        test_single_reference()
        print("\n")
        test_batch_references()
