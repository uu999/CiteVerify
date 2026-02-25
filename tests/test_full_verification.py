# -*- coding: utf-8 -*-
"""
端到端测试：PDF URL -> 提取参考文献 -> 校验真伪 -> 生成报告

完整流程：
1. 输入论文 PDF URL
2. 转换为 Markdown
3. 提取参考文献列表
4. 逐条校验真伪（arXiv -> Semantic Scholar -> OpenAlex）
5. 生成校验报告
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.pipeline import extract_references_from_url
from citeverify.checkers import verify_references, SearchSource
from citeverify.models import YaYiDocParserConfig

# Semantic Scholar API Key（可设为环境变量或直接填写）
DEFAULT_SS_API_KEY = "k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV"


def verify_paper_references(
    pdf_url: str,
    citation_format: str = "ieee",
    listing_style: str = "numbered",
    output_dir: str = "output",
    request_delay: float = 1.5,
    semantic_scholar_api_key: Optional[str] = None,
    use_semantic_scholar: bool = True,
    use_openalex: bool = True,
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    从 PDF URL 提取参考文献并校验真伪
    
    搜索优先级：arXiv -> Semantic Scholar -> OpenAlex
    
    Args:
        pdf_url: 论文 PDF 的 URL
        citation_format: 引用格式（apa, mla, ieee, gb_t_7714, chicago, harvard, vancouver）
        listing_style: 列举方式（numbered 或 author_year）
        output_dir: 输出目录
        request_delay: API 请求间隔（秒）
        semantic_scholar_api_key: Semantic Scholar API Key
        use_semantic_scholar: 是否使用 Semantic Scholar
        use_openalex: 是否使用 OpenAlex
        save_report: 是否保存报告文件
        
    Returns:
        校验报告字典
    """
    # 使用默认 API Key（如果未提供）
    if semantic_scholar_api_key is None:
        semantic_scholar_api_key = DEFAULT_SS_API_KEY
    print("=" * 70)
    print("参考文献真伪校验系统")
    print("=" * 70)
    print(f"论文 URL: {pdf_url}")
    print(f"引用格式: {citation_format}")
    print(f"列举方式: {listing_style}")
    print("-" * 70)
    
    # 1. 提取参考文献
    print("\n📄 Step 1: 提取参考文献...")
    
    yayi_config = YaYiDocParserConfig()
    
    extraction_result = extract_references_from_url(
        pdf_url,
        citation_format=citation_format,
        listing_style=listing_style,
        yayi_config=yayi_config,
        download_timeout=600,
    )
    
    if not extraction_result.success:
        print(f"❌ 提取失败: {extraction_result.error}")
        return {"success": False, "error": extraction_result.error}
    
    references = extraction_result.references
    print(f"✅ 提取完成，共 {len(references)} 条参考文献")
    
    if not references:
        print("⚠️ 未提取到参考文献")
        return {"success": False, "error": "未提取到参考文献"}
    
    # 2. 校验参考文献
    sources = ["arXiv"]
    if use_semantic_scholar:
        sources.append("Semantic Scholar")
    if use_openalex:
        sources.append("OpenAlex")
    print(f"\n🔍 Step 2: 校验参考文献真伪（使用 {' -> '.join(sources)}）...")
    print(f"   请求间隔: {request_delay}s")
    print("-" * 70)
    
    has_number = (listing_style == "numbered")
    
    verified_refs = verify_references(
        references,
        has_number=has_number,
        request_delay=request_delay,
        semantic_scholar_api_key=semantic_scholar_api_key,
        use_semantic_scholar=use_semantic_scholar,
        use_openalex=use_openalex,
        verbose=True,
    )
    
    # 3. 生成报告
    print(f"\n📊 Step 3: 生成校验报告...")
    
    report = generate_report(
        pdf_url=pdf_url,
        verified_refs=verified_refs,
        has_number=has_number,
        citation_format=citation_format,
    )
    
    # 4. 打印报告
    print_report(report)
    
    # 5. 保存报告
    if save_report:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 从 URL 提取文件名
        url_name = pdf_url.split("/")[-1].split("?")[0]
        if url_name.endswith(".pdf"):
            url_name = url_name[:-4]
        report_name = f"verification_report_{url_name[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_file = output_path / report_name
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 报告已保存: {report_file}")
    
    return report


def generate_report(
    pdf_url: str,
    verified_refs: List[List],
    has_number: bool,
    citation_format: str,
) -> Dict[str, Any]:
    """
    生成校验报告
    
    Args:
        pdf_url: 论文 URL
        verified_refs: 校验后的参考文献列表
        has_number: 是否有编号
        citation_format: 引用格式
        
    Returns:
        报告字典
    """
    total = len(verified_refs)
    verified_count = sum(1 for r in verified_refs if r[-3])  # can_get 字段
    unverified_count = total - verified_count
    
    # 按来源统计（通过 URL 特征判断）
    arxiv_count = 0
    ss_count = 0
    openalex_count = 0
    has_pdf_count = 0
    
    details = []
    
    for i, ref in enumerate(verified_refs):
        if has_number:
            # [编号, 全文, 标题, 作者, 年份, can_get, abstract, pdf_url, source]
            num = ref[0]
            title = ref[2]
            authors = ref[3]
            year = ref[4]
            can_get = ref[-4]
            abstract = ref[-3]
            pdf_url_ref = ref[-2]
            source = ref[-1]
        else:
            # [全文, 标题, 作者, 年份, can_get, abstract, pdf_url, source]
            num = i + 1
            title = ref[1]
            authors = ref[2]
            year = ref[3]
            can_get = ref[-4]
            abstract = ref[-3]
            pdf_url_ref = ref[-2]
            source = ref[-1]
        
        detail = {
            "number": num,
            "title": title,
            "authors": authors,
            "year": year,
            "verified": can_get,
            "has_pdf": bool(pdf_url_ref),
            "pdf_url": pdf_url_ref,
            "source": source,
            "abstract_preview": abstract[:200] + "..." if abstract and len(abstract) > 200 else abstract,
        }
        details.append(detail)
        
        if can_get:
            if pdf_url_ref:
                has_pdf_count += 1
            # 使用实际来源字段统计
            if source == "arxiv":
                arxiv_count += 1
            elif source == "semantic_scholar":
                ss_count += 1
            elif source == "openalex":
                openalex_count += 1
    
    report = {
        "meta": {
            "pdf_url": pdf_url,
            "citation_format": citation_format,
            "has_number": has_number,
            "generated_at": datetime.now().isoformat(),
        },
        "summary": {
            "total": total,
            "verified": verified_count,
            "unverified": unverified_count,
            "verification_rate": f"{verified_count / total * 100:.1f}%" if total > 0 else "N/A",
            "arxiv_found": arxiv_count,
            "semantic_scholar_found": ss_count,
            "openalex_found": openalex_count,
            "has_pdf": has_pdf_count,
        },
        "details": details,
    }
    
    return report


def print_report(report: Dict[str, Any]) -> None:
    """打印校验报告"""
    summary = report["summary"]
    details = report["details"]
    
    print("\n" + "=" * 70)
    print("📋 校验报告")
    print("=" * 70)
    
    print(f"\n📊 统计摘要:")
    print(f"   总参考文献数:     {summary['total']}")
    print(f"   ✅ 可验证:        {summary['verified']} ({summary['verification_rate']})")
    print(f"   ❌ 无法验证:      {summary['unverified']}")
    print(f"   📚 来自 arXiv:    {summary['arxiv_found']}")
    print(f"   📖 来自 Semantic Scholar: {summary['semantic_scholar_found']}")
    print(f"   🔬 来自 OpenAlex: {summary.get('openalex_found', 0)}")
    print(f"   📄 有 PDF 链接:   {summary['has_pdf']}")
    
    print(f"\n📝 详细结果:")
    print("-" * 70)
    
    for d in details:
        status = "✅" if d["verified"] else "❌"
        pdf_status = "📄" if d["has_pdf"] else "  "
        title_display = d["title"][:50] + "..." if d["title"] and len(d["title"]) > 50 else (d["title"] or "(无标题)")
        
        print(f"  [{d['number']:2}] {status} {pdf_status} {title_display}")
        if d["verified"] and d["pdf_url"]:
            print(f"       └─ PDF: {d['pdf_url'][:60]}...")
    
    print("\n" + "=" * 70)
    
    # 列出无法验证的文献
    unverified = [d for d in details if not d["verified"]]
    if unverified:
        print("\n⚠️ 无法验证的文献（可能需要人工核实）:")
        for d in unverified:
            title = d["title"] if d["title"] else "(无标题)"
            print(f"   [{d['number']}] {title}")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="从论文 PDF 提取参考文献并校验真伪（arXiv -> Semantic Scholar -> OpenAlex）"
    )
    parser.add_argument("pdf_url", help="论文 PDF 的 URL")
    parser.add_argument(
        "-c", "--citation-format",
        choices=["apa", "mla", "ieee", "gb_t_7714", "chicago", "harvard", "vancouver"],
        default="ieee",
        help="引用格式 (默认: ieee)"
    )
    parser.add_argument(
        "-l", "--listing-style",
        choices=["numbered", "author_year"],
        default="numbered",
        help="列举方式 (默认: numbered)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="输出目录 (默认: output)"
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=1.5,
        help="API 请求间隔秒数 (默认: 1.5)"
    )
    parser.add_argument(
        "--ss-api-key",
        default=None,
        help="Semantic Scholar API Key（默认使用内置 Key）"
    )
    parser.add_argument(
        "--no-semantic-scholar",
        action="store_true",
        help="禁用 Semantic Scholar"
    )
    parser.add_argument(
        "--no-openalex",
        action="store_true",
        help="禁用 OpenAlex"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存报告文件"
    )
    
    args = parser.parse_args()
    
    # 获取 API Key（优先命令行参数，其次环境变量，最后默认值）
    ss_api_key = args.ss_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or DEFAULT_SS_API_KEY
    
    verify_paper_references(
        pdf_url=args.pdf_url,
        citation_format=args.citation_format,
        listing_style=args.listing_style,
        output_dir=args.output_dir,
        request_delay=args.delay,
        semantic_scholar_api_key=ss_api_key,
        use_semantic_scholar=not args.no_semantic_scholar,
        use_openalex=not args.no_openalex,
        save_report=not args.no_save,
    )
