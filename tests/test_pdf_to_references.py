# -*- coding: utf-8 -*-
"""
测试：PDF URL -> Markdown -> 参考文献列表
测试：本地 MD 文件 -> 参考文献列表

使用默认配置，只打印关键信息。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.pipeline import extract_references_from_url, extract_references_from_markdown
from citeverify.models import YaYiDocParserConfig

# 支持的参数
LISTING_STYLES = ("numbered", "author_year")
CITATION_FORMATS = (
    "apa", "mla", "ieee", "gb_t_7714",
    "chicago", "harvard", "vancouver",
    "other",  # 其他格式：将整段参考文献交给 LLM 提取，需配置 LLM
)



def test_pdf_to_references(
    pdf_url: str,
    listing_style: str = "numbered",
    citation_format: str = "apa",
):
    """
    从 PDF URL 转换到 MD，再提取参考文献列表，只打印关键信息。

    Args:
        pdf_url: PDF 文件的 URL
        listing_style: 条目列举方式，"numbered" 或 "author_year"
        citation_format: 引用格式，可选:
            "apa", "mla", "ieee", "gb_t_7714", "chicago", "harvard", "vancouver"
    """
    if listing_style not in LISTING_STYLES:
        raise ValueError(
            f"listing_style 必须是 {LISTING_STYLES} 之一，当前: {listing_style}"
        )
    if citation_format not in CITATION_FORMATS:
        raise ValueError(
            f"citation_format 必须是 {CITATION_FORMATS} 之一，当前: {citation_format}"
        )

    # 默认配置：雅意文档解析
    yayi_config = YaYiDocParserConfig()

    print("=" * 60)
    print("PDF -> Markdown -> 参考文献列表")
    print("=" * 60)
    print(f"PDF URL: {pdf_url}")
    print(f"条目列举: {listing_style}")
    print(f"引用格式: {citation_format}")
    print("-" * 60)

    result = extract_references_from_url(
        pdf_url,
        citation_format=citation_format,
        listing_style=listing_style,
        yayi_config=yayi_config,
        download_timeout=600,
    )

    # 只打印关键信息
    if not result.success:
        print(f"失败: {result.error}")
        return result

    print(f"成功 | 参考文献条数: {len(result.references)}")
    print("-" * 60)

    for i, ref in enumerate(result.references, 1):
        # ref: 数字标号型 [编号, 全文, 题目, 作者, 年份]
        #      作者年份型 [全文, 题目, 作者, 年份]
        if listing_style == "numbered":
            num, full, title, authors, year = ref
            print(f"[{num}] {title or '(无题目)'}")
        else:
            full, title, authors, year = ref
            print(f"[{i}] {title or '(无题目)'}")
        print(f"    作者: {authors or '-'}")
        print(f"    年份: {year or '-'}")
        print()

    print("=" * 60)
    return result


def test_md_to_references(
    md_path: str,
    listing_style: str = "numbered",
    citation_format: str = "apa",
):
    """
    从本地 MD 文件提取参考文献列表，只打印关键信息。

    Args:
        md_path: MD 文件路径
        listing_style: 条目列举方式，"numbered" 或 "author_year"
        citation_format: 引用格式，可选:
            "apa", "mla", "ieee", "gb_t_7714", "chicago", "harvard", "vancouver"
    """
    if listing_style not in LISTING_STYLES:
        raise ValueError(
            f"listing_style 必须是 {LISTING_STYLES} 之一，当前: {listing_style}"
        )
    if citation_format not in CITATION_FORMATS:
        raise ValueError(
            f"citation_format 必须是 {CITATION_FORMATS} 之一，当前: {citation_format}"
        )

    md_file = Path(md_path)
    if not md_file.exists():
        raise FileNotFoundError(f"MD 文件不存在: {md_path}")

    print("=" * 60)
    print("MD 文件 -> 参考文献列表")
    print("=" * 60)
    print(f"MD 文件: {md_path}")
    print(f"条目列举: {listing_style}")
    print(f"引用格式: {citation_format}")
    print("-" * 60)

    # 读取 MD 文件内容
    with open(md_file, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    print(f"MD 文件大小: {len(markdown_content)} 字符")

    result = extract_references_from_markdown(
        markdown_content,
        citation_format=citation_format,
        listing_style=listing_style,
    )

    # 只打印关键信息
    if not result.success:
        print(f"失败: {result.error}")
        return result

    print(f"成功 | 参考文献条数: {len(result.references)}")
    print("-" * 60)

    for i, ref in enumerate(result.references, 1):
        if listing_style == "numbered":
            num, full, title, authors, year = ref
            print(f"[{num}] {title or '(无题目)'}")
        else:
            full, title, authors, year = ref
            print(f"[{i}] {title or '(无题目)'}")
        print(f"    作者: {authors or '-'}")
        print(f"    年份: {year or '-'}")
        print()

    print("=" * 60)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF/MD -> 参考文献列表（仅打印关键信息）"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # pdf 子命令
    pdf_parser = subparsers.add_parser("pdf", help="从 PDF URL 提取参考文献")
    pdf_parser.add_argument("pdf_url", help="PDF 文件的 URL")
    pdf_parser.add_argument(
        "-l", "--listing-style",
        choices=LISTING_STYLES,
        default="numbered",
        help="条目列举方式 (默认: numbered)",
    )
    pdf_parser.add_argument(
        "-c", "--citation-format",
        choices=CITATION_FORMATS,
        default="apa",
        help="引用格式 (默认: apa)",
    )

    # md 子命令
    md_parser = subparsers.add_parser("md", help="从本地 MD 文件提取参考文献")
    md_parser.add_argument("md_path", help="MD 文件路径")
    md_parser.add_argument(
        "-l", "--listing-style",
        choices=LISTING_STYLES,
        default="numbered",
        help="条目列举方式 (默认: numbered)",
    )
    md_parser.add_argument(
        "-c", "--citation-format",
        choices=CITATION_FORMATS,
        default="apa",
        help="引用格式 (默认: apa)",
    )

    args = parser.parse_args()

    if args.command == "pdf":
        test_pdf_to_references(
            pdf_url=args.pdf_url,
            listing_style=args.listing_style,
            citation_format=args.citation_format,
        )
    elif args.command == "md":
        test_md_to_references(
            md_path=args.md_path,
            listing_style=args.listing_style,
            citation_format=args.citation_format,
        )
    else:
        parser.print_help()
