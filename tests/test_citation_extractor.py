# -*- coding: utf-8 -*-
"""
测试：引用文本定位与提取

测试各种引用格式的提取功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.extractor import (
    CitationExtractor,
    CitationFormat,
    extract_numeric_citations,
    extract_author_year_citations,
)


def test_numeric_citations():
    """测试数字型引用提取"""
    print("=" * 70)
    print("数字型引用提取测试")
    print("=" * 70)
    
    extractor = CitationExtractor()
    
    # 测试用例
    test_cases = [
        # (句子, 预期结果描述)
        ("[1] shows that...", "单一引用"),
        ("As shown in [2-5], the results...", "范围引用"),
        ("Several studies [2,3,5] have shown...", "列表引用"),
        ("According to [1,3-5,8], we can...", "混合引用"),
        ("The work [1][2][3] demonstrates...", "连续单一引用"),
        ("Studies [10–15] and [20] confirm...", "en-dash 范围 + 单一"),
    ]
    
    for sentence, desc in test_cases:
        print(f"\n📝 测试: {desc}")
        print(f"   句子: {sentence}")
        
        results = extractor.extract_all_numeric(sentence)
        print(f"   结果: {len(results)} 个引用")
        for num, raw, type_, span in results:
            print(f"      - 编号 {num}, 类型: {type_}, 原文: {raw}, 位置: {span}")
    
    print("\n" + "-" * 70)
    
    # 测试完整文本
    test_text = """
# Introduction

This is the first paragraph. Recent studies [1] have shown significant progress. 
The work by [2-5] demonstrates the importance of this field. Multiple authors [6,7,8] agree on this point.

# Background

Previous research [1,3-5,8] has established the foundation. 
As noted in [10], there are still challenges. The combination of [11,12] and [15-18] provides insights.

# Conclusion

In conclusion, the evidence [1][2][3] strongly supports our hypothesis.
"""
    
    print("\n📄 完整文本测试:")
    citations = extract_numeric_citations(test_text)
    print(f"   共提取 {len(citations)} 个引用")
    
    for c in citations[:5]:  # 只显示前5个
        # c = [position, number, anchor, context]
        pos, num, anchor, context = c
        print(f"\n   [{num}] 位置: {pos}")
        print(f"      句子: {anchor[:60]}...")


def test_author_year_citations():
    """测试作者年份型引用提取"""
    print("\n" + "=" * 70)
    print("作者年份型引用提取测试")
    print("=" * 70)
    
    extractor = CitationExtractor()
    
    # 英文测试用例
    en_test_cases = [
        ("Smith (2020) proposed a new method.", "单作者"),
        ("Smith & Brown (2021) extended the work.", "多作者 &"),
        ("Smith and Brown (2021) found similar results.", "多作者 and"),
        ("Smith et al. (2019) conducted experiments.", "et al."),
        ("The method (Smith, 2020) is widely used.", "括号内作者"),
        ("Recent work (Smith et al., 2019) confirms this.", "括号内 et al."),
        ("Studies (Smith & Brown, 2021) show that...", "括号内多作者"),
        ("Smith (2020a) and Smith (2020b) differ in...", "年份后缀"),
        ("Multiple studies (Smith, 2019; Brown, 2020) agree.", "多引用分号分隔"),
    ]
    
    print("\n--- 英文作者年份引用 ---")
    for sentence, desc in en_test_cases:
        print(f"\n📝 测试: {desc}")
        print(f"   句子: {sentence}")
        
        results = extractor.extract_all_author_year(sentence)
        print(f"   结果: {len(results)} 个引用")
        for author, year, raw, type_, span in results:
            print(f"      - 作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    # 中文测试用例
    cn_test_cases = [
        ("张三（2021）提出了新方法。", "中文单作者"),
        ("李四、王五（2020）扩展了该研究。", "中文多作者"),
        ("张三等（2021）进行了实验。", "中文等"),
        ("该方法（张三，2019）被广泛使用。", "中文括号内作者"),
        ("多项研究（张三，2019；李四，2020）表明。", "中文括号内多引用"),
    ]
    
    print("\n--- 中文作者年份引用 ---")
    for sentence, desc in cn_test_cases:
        print(f"\n📝 测试: {desc}")
        print(f"   句子: {sentence}")
        
        results = extractor.extract_all_chinese(sentence)
        print(f"   结果: {len(results)} 个引用")
        for author, year, raw, type_, span in results:
            print(f"      - 作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")


def test_context_extraction():
    """测试上下文提取"""
    print("\n" + "=" * 70)
    print("上下文提取测试")
    print("=" * 70)
    
    test_text = """
# Introduction

This is the first sentence. This is the second sentence. The third sentence mentions [1]. The fourth sentence continues. The fifth sentence ends.

Another paragraph starts here. It has a citation [2] in the middle. And it continues with more text.
"""
    
    citations = extract_numeric_citations(test_text, context_window=2)
    
    for c in citations:
        # c = [position, number, anchor, context]
        pos, num, anchor, context = c
        print(f"\n📌 引用 [{num}]")
        print(f"   📍 位置: {pos}")
        print(f"   📍 定位句子: {anchor}")
        print(f"   📖 上下文: {context}")


def test_real_paper_sample():
    """测试真实论文样本"""
    print("\n" + "=" * 70)
    print("真实论文样本测试")
    print("=" * 70)
    
    # 模拟真实论文段落
    sample_text = """
# Introduction

Process modeling is a fundamental activity in business process management [1]. 
The challenge of creating accurate process models has been studied extensively [2-5].
Recent work by Smith et al. (2019) and Brown (2020) has focused on automation.
The combination of natural language processing (Smith & Brown, 2021) and machine learning (Liu et al., 2022) 
shows promising results.

# Related Work

Previous studies [6,7,8] have explored various approaches. 
Zhang et al. (2023) proposed a novel framework. 
Multiple researchers (Wang, 2019; Li, 2020; Chen et al., 2021) have contributed to this field.
中文研究方面，张三（2021）和李四等（2020）也进行了相关探索。

# Methodology

Our approach builds on [1,3-5,8] and extends the work of Smith (2020a).
We follow the methodology proposed by (Brown et al., 2022).
"""
    
    print("\n--- 数字型引用 ---")
    numeric_citations = extract_numeric_citations(sample_text)
    print(f"共提取 {len(numeric_citations)} 个数字型引用")
    for c in numeric_citations:
        # c = [position, number, anchor, context]
        pos, num, anchor, context = c
        print(f"   [{num}] 位置: {pos[0]}, 句子ID: {pos[1]}, span: {pos[2]}")
        print(f"       句子: {anchor[:50]}...")
    
    print("\n--- 作者年份型引用 ---")
    author_citations = extract_author_year_citations(sample_text)
    print(f"共提取 {len(author_citations)} 个作者年份型引用")
    for c in author_citations:
        # c = [position, authors, year, anchor, context]
        pos, authors, year, anchor, context = c
        print(f"   {authors} ({year}) 位置: {pos[0]}, 句子ID: {pos[1]}, span: {pos[2]}")
        print(f"       句子: {anchor[:50]}...")


def test_each_subtype():
    """测试每个子类型的提取函数"""
    print("\n" + "=" * 70)
    print("子类型提取函数测试")
    print("=" * 70)
    
    extractor = CitationExtractor()
    
    # 数字型子类型
    print("\n--- 数字型子类型 ---")
    
    print("\n1. extract_numeric_single:")
    results = extractor.extract_numeric_single("The study [1] and [2] show...")
    for num, raw, type_, span in results:
        print(f"   编号: {num}, 原文: {raw}, 类型: {type_}, 位置: {span}")
    
    print("\n2. extract_numeric_range:")
    results = extractor.extract_numeric_range("Studies [2-5] and [10–15] show...")
    for num, raw, type_, span in results:
        print(f"   编号: {num}, 原文: {raw}, 类型: {type_}, 位置: {span}")
    
    print("\n3. extract_numeric_list:")
    results = extractor.extract_numeric_list("According to [2,3,5] and [7,8,9]...")
    for num, raw, type_, span in results:
        print(f"   编号: {num}, 原文: {raw}, 类型: {type_}, 位置: {span}")
    
    print("\n4. extract_numeric_mixed:")
    results = extractor.extract_numeric_mixed("Combined [1,3-5,8] and [10,12-14,20]...")
    for num, raw, type_, span in results:
        print(f"   编号: {num}, 原文: {raw}, 类型: {type_}, 位置: {span}")
    
    # 作者年份型子类型
    print("\n--- 作者年份型子类型 ---")
    
    print("\n1. extract_author_single:")
    results = extractor.extract_author_single("Smith (2020) proposed...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n2. extract_author_multiple_and:")
    results = extractor.extract_author_multiple_and("Smith & Brown (2021) and Lee and Kim (2022)...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n3. extract_author_et_al:")
    results = extractor.extract_author_et_al("Smith et al. (2019) conducted...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n4. extract_author_parenthetical:")
    results = extractor.extract_author_parenthetical("The method (Smith, 2020) and (Brown et al., 2021)...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n5. extract_author_multiple_citations:")
    results = extractor.extract_author_multiple_citations("Studies (Smith, 2019; Brown, 2020; Lee et al., 2021)...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    # 中文子类型
    print("\n--- 中文作者年份型子类型 ---")
    
    print("\n1. extract_chinese_single:")
    results = extractor.extract_chinese_single("张三（2021）提出...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n2. extract_chinese_multiple:")
    results = extractor.extract_chinese_multiple("李四、王五（2020）研究...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n3. extract_chinese_et_al:")
    results = extractor.extract_chinese_et_al("张三等（2021）发现...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n4. extract_chinese_parenthetical:")
    results = extractor.extract_chinese_parenthetical("该方法（张三，2019）...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")
    
    print("\n5. extract_chinese_multiple_parenthetical:")
    results = extractor.extract_chinese_multiple_parenthetical("研究表明（张三，2019；李四，2020）...")
    for author, year, raw, type_, span in results:
        print(f"   作者: {author}, 年份: {year}, 类型: {type_}, 位置: {span}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试引用提取功能")
    parser.add_argument(
        "--test",
        choices=["numeric", "author_year", "context", "real", "subtype", "all"],
        default="all",
        help="测试类型"
    )
    
    args = parser.parse_args()
    
    if args.test in ["numeric", "all"]:
        test_numeric_citations()
    
    if args.test in ["author_year", "all"]:
        test_author_year_citations()
    
    if args.test in ["context", "all"]:
        test_context_extraction()
    
    if args.test in ["real", "all"]:
        test_real_paper_sample()
    
    if args.test in ["subtype", "all"]:
        test_each_subtype()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
