# -*- coding: utf-8 -*-
"""
测试引文相关性分析器

使用方法:
    python tests/test_relevance_analyzer.py --api-key YOUR_API_KEY [--base-url URL] [--model MODEL]
"""
import sys
import os
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citeverify.checkers import (
    RelevanceAnalyzer,
    RelevanceResult,
    RelevanceJudgment,
    analyze_relevance,
    generate_relevance_report,
)


def create_sample_citations():
    """创建测试用的引用样例"""
    return [
        {
            # 示例1：强支持 - 引用与参考文献主题高度相关
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.",
            "citation_anchor": "Recent advances in pre-trained language models, such as BERT [1], have significantly improved performance on various NLP tasks.",
            "context": "Natural language processing has undergone a paradigm shift with the introduction of large pre-trained models. Recent advances in pre-trained language models, such as BERT [1], have significantly improved performance on various NLP tasks. These models leverage transfer learning to achieve state-of-the-art results.",
        },
        {
            # 示例2：弱支持 - 引用相关但不是直接支持
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art.",
            "citation_anchor": "Deep learning techniques have been applied to various domains including natural language processing [2].",
            "context": "The field of artificial intelligence has seen remarkable progress in recent years. Deep learning techniques have been applied to various domains including natural language processing [2]. Our work builds upon these foundations to develop novel approaches.",
        },
        {
            # 示例3：不支持 - 引用与论点不匹配
            "title": "A Survey on Knowledge Graphs: Representation, Acquisition and Applications",
            "abstract": "Knowledge graphs have attracted increasing attention from both academia and industry. This survey provides a comprehensive review of knowledge graph covering three aspects: representation, acquisition, and applications. We present a systematic overview of knowledge representation learning, including translation-based models and tensor decomposition methods.",
            "citation_anchor": "Large language models have demonstrated emergent reasoning capabilities [3].",
            "context": "Recent research has focused on understanding the capabilities of AI systems. Large language models have demonstrated emergent reasoning capabilities [3]. These findings have important implications for the development of more advanced AI.",
        },
        {
            # 示例4：不确定 - 摘要信息不足以判断
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "citation_anchor": "Transformer architectures have enabled significant advances in machine translation [4].",
            "context": "Neural machine translation has evolved rapidly in recent years. Transformer architectures have enabled significant advances in machine translation [4]. However, challenges remain in low-resource language pairs.",
        },
    ]


def test_single_analysis(analyzer: RelevanceAnalyzer):
    """测试单个引用分析"""
    print("\n" + "=" * 60)
    print("测试单个引用分析")
    print("=" * 60)
    
    sample = create_sample_citations()[0]
    
    print(f"\n标题: {sample['title'][:50]}...")
    print(f"引用句: {sample['citation_anchor'][:60]}...")
    
    result = analyzer.analyze(
        title=sample["title"],
        abstract=sample["abstract"],
        citation_anchor=sample["citation_anchor"],
        context=sample["context"],
    )
    
    print(f"\n分析结果:")
    print(f"  成功: {result.success}")
    print(f"  推断论点: {result.claim}")
    print(f"  判断: {result.judgment.value}")
    print(f"  理由: {result.reason}")
    
    if not result.success:
        print(f"  错误: {result.error_message}")
    
    return result


def test_batch_analysis(analyzer: RelevanceAnalyzer):
    """测试批量分析"""
    print("\n" + "=" * 60)
    print("测试批量分析")
    print("=" * 60)
    
    samples = create_sample_citations()
    
    def progress_callback(current, total):
        print(f"  进度: {current}/{total}")
    
    results = analyzer.analyze_batch(
        samples,
        progress_callback=progress_callback,
    )
    
    print(f"\n批量分析完成，共 {len(results)} 条结果")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.title[:40]}...")
        print(f"    判断: {result.judgment.value}")
        print(f"    理由: {result.reason[:80]}..." if result.reason else "    理由: (无)")
    
    return results


def test_report_generation(results):
    """测试报告生成"""
    print("\n" + "=" * 60)
    print("测试报告生成")
    print("=" * 60)
    
    report = generate_relevance_report(results)
    print(report)


def test_response_parsing():
    """测试响应解析（不需要 API）"""
    print("\n" + "=" * 60)
    print("测试响应解析（离线测试）")
    print("=" * 60)
    
    # 创建一个临时的分析器来测试解析
    class MockAnalyzer(RelevanceAnalyzer):
        def __init__(self):
            # 跳过 OpenAI 客户端初始化
            self.model_name = "test"
            self.client = None
    
    try:
        analyzer = MockAnalyzer()
    except Exception:
        print("  跳过（需要 openai 包）")
        return
    
    # 测试 JSON 格式响应解析
    test_responses = [
        # 标准 JSON 代码块格式
        '''```json
{
    "claim": "The authors claim that BERT has improved NLP task performance.",
    "judge": "Strongly supports",
    "reason": "The abstract explicitly states that BERT achieves state-of-the-art results on various NLP tasks."
}
```''',
        
        # 纯 JSON 格式（无代码块）
        '''{
    "claim": "Deep learning is used in NLP.",
    "judge": "Weakly supports",
    "reason": "While the paper discusses CNNs for image classification, deep learning is indeed applicable to NLP."
}''',
        
        # JSON 格式 - 不支持
        '''```json
{
    "claim": "The paper discusses emergent reasoning.",
    "judge": "Does not support",
    "reason": "The abstract focuses on knowledge graphs, not reasoning capabilities."
}
```''',
        
        # JSON 格式 - 不确定
        '''{"claim": "unclear claim", "judge": "Unclear", "reason": "The information provided is insufficient to make a determination."}''',
        
        # 带额外文本的 JSON（测试提取能力）
        '''Here is my analysis:

```json
{
    "claim": "LLMs enable process automation",
    "judge": "Strongly supports",
    "reason": "The abstract directly discusses using LLMs for business process modeling automation."
}
```

This concludes my analysis.''',
    ]
    
    for i, response in enumerate(test_responses, 1):
        claim, judgment, reason = analyzer._parse_response(response)
        print(f"\n测试 {i}:")
        print(f"  解析的论点: {claim[:50]}..." if len(claim) > 50 else f"  解析的论点: {claim}")
        print(f"  解析的判断: {judgment.value}")
        print(f"  解析的理由: {reason[:50]}..." if len(reason) > 50 else f"  解析的理由: {reason}")


def test_judgment_from_string():
    """测试判断字符串解析"""
    print("\n" + "=" * 60)
    print("测试判断字符串解析")
    print("=" * 60)
    
    test_cases = [
        ("Strongly supports", RelevanceJudgment.STRONGLY_SUPPORTS),
        ("strongly support", RelevanceJudgment.STRONGLY_SUPPORTS),
        ("Weakly supports", RelevanceJudgment.WEAKLY_SUPPORTS),
        ("weakly support", RelevanceJudgment.WEAKLY_SUPPORTS),
        ("Does not support", RelevanceJudgment.DOES_NOT_SUPPORT),
        ("not support", RelevanceJudgment.DOES_NOT_SUPPORT),
        ("Unclear", RelevanceJudgment.UNCLEAR),
        ("UNCLEAR", RelevanceJudgment.UNCLEAR),
        ("something else", RelevanceJudgment.UNCLEAR),  # 默认
    ]
    
    for text, expected in test_cases:
        result = RelevanceJudgment.from_string(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result.value} (期望: {expected.value})")


def main():
    parser = argparse.ArgumentParser(description="测试引文相关性分析器")
    parser.add_argument("--api-key", type=str, help="API 密钥")
    parser.add_argument("--base-url", type=str, help="API 基础 URL")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="模型名称")
    parser.add_argument("--offline", action="store_true", help="仅运行离线测试")
    args = parser.parse_args()
    
    print("=" * 60)
    print("引文相关性分析器测试")
    print("=" * 60)
    
    # 离线测试（不需要 API）
    test_judgment_from_string()
    test_response_parsing()
    
    if args.offline:
        print("\n离线测试完成")
        return
    
    if not args.api_key:
        print("\n⚠️ 未提供 API 密钥，跳过在线测试")
        print("使用方法: python tests/test_relevance_analyzer.py --api-key YOUR_API_KEY")
        return
    
    # 在线测试（需要 API）
    print(f"\n使用模型: {args.model}")
    print(f"API URL: {args.base_url or '默认'}")
    
    try:
        analyzer = RelevanceAnalyzer(
            model_name=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    except ImportError as e:
        print(f"\n错误: {e}")
        print("请安装 openai 包: pip install openai")
        return
    
    # 单个分析测试
    test_single_analysis(analyzer)
    
    # 批量分析测试
    results = test_batch_analysis(analyzer)
    
    # 报告生成测试
    test_report_generation(results)
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
