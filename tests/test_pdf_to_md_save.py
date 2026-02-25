# -*- coding: utf-8 -*-
"""
测试：PDF URL -> Markdown 文件（按 title 聚合，以第一个 title 命名）
"""
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from citeverify.converter.pdf_converter import PDFConverter
from citeverify.models.document import YaYiDocParserConfig


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    清理文件名，移除非法字符
    
    Args:
        name: 原始文件名
        max_length: 最大长度
        
    Returns:
        清理后的文件名
    """
    if not name:
        return "untitled"
    
    # 移除非法字符
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '', name)
    # 替换空格为下划线
    name = re.sub(r'\s+', '_', name)
    # 移除首尾的点和空格
    name = name.strip('. ')
    # 截断
    if len(name) > max_length:
        name = name[:max_length]
    
    return name or "untitled"


def test_pdf_to_md_save(
    pdf_url: str,
    output_dir: str = "output",
) -> Path:
    """
    将 PDF URL 转化为 MD 文件（按 title 聚合），以第一个 title 命名并保存。
    
    Args:
        pdf_url: PDF 文件的 URL
        output_dir: 输出目录（默认: output）
        
    Returns:
        保存的 MD 文件路径
    """
    print("=" * 60)
    print("PDF URL -> Markdown 文件")
    print("=" * 60)
    print(f"PDF URL: {pdf_url}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 使用默认配置
    yayi_config = YaYiDocParserConfig()
    
    print(f"\n配置:")
    print(f"  - API: {yayi_config.api_url}")
    print(f"  - Mode: {yayi_config.mode}")
    print(f"  - 页眉页码过滤: {yayi_config.remove_headers_footers}")
    
    # 创建转换器
    converter = PDFConverter(yayi_config=yayi_config)
    
    print("\n" + "-" * 60)
    print("开始转换...")
    print("-" * 60)
    
    # 执行转换
    result = converter.convert(pdf_url)
    
    print(f"\n转换完成!")
    print(f"  - 章节数量: {len(result.sections)}")
    print(f"  - 内容长度: {len(result.full_markdown)} 字符")
    print(f"  - 分离成功: {result.separation_success}")
    
    # 获取第一个 title 作为文件名
    first_title = None
    if result.sections:
        for section in result.sections:
            if section.title:
                first_title = section.title
                break
    
    if first_title:
        filename = sanitize_filename(first_title)
    else:
        # 从 URL 提取文件名
        url_name = pdf_url.split("/")[-1].split("?")[0]
        if url_name.endswith(".pdf"):
            url_name = url_name[:-4]
        filename = sanitize_filename(url_name) or "document"
    
    # 保存文件
    output_file = output_path / f"{filename}.md"
    
    # 避免覆盖，添加序号
    counter = 1
    while output_file.exists():
        output_file = output_path / f"{filename}_{counter}.md"
        counter += 1
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.full_markdown)
    
    print(f"\n" + "=" * 60)
    print(f"文件已保存: {output_file}")
    print(f"第一个标题: {first_title or '(无)'}")
    print("=" * 60)
    
    # 显示章节列表
    if result.sections:
        print(f"\n章节列表 (共 {len(result.sections)} 个):")
        for i, section in enumerate(result.sections[:15], 1):
            title = section.title[:40] + "..." if len(section.title) > 40 else section.title
            print(f"  {i}. [{section.level}] {title or '(无标题)'}")
        if len(result.sections) > 15:
            print(f"  ... 还有 {len(result.sections) - 15} 个章节")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PDF URL -> Markdown 文件（按 title 聚合，以第一个 title 命名）"
    )
    parser.add_argument("pdf_url", help="PDF 文件的 URL")
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="输出目录 (默认: output)"
    )
    
    args = parser.parse_args()
    
    test_pdf_to_md_save(
        pdf_url=args.pdf_url,
        output_dir=args.output_dir,
    )
