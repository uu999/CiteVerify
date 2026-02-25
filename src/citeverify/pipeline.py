# -*- coding: utf-8 -*-
"""
参考文献提取流水线

一站式完成：PDF URL -> Markdown -> 分离参考文献 -> 提取参考文献列表
"""
from typing import List, Optional, Union
from dataclasses import dataclass, field

from .converter import PDFConverter, separate_markdown
from .extractor import extract_references
from .models import MinerUConfig, LLMConfig, ListingStyle, CitationFormat, YaYiDocParserConfig


@dataclass
class PipelineResult:
    """
    流水线处理结果
    
    Attributes:
        success: 是否成功
        references: 提取的参考文献列表
        references_text: 参考文献原文
        main_text: 正文内容
        error: 错误信息（如果失败）
    """
    success: bool = False
    references: List[list] = field(default_factory=list)
    references_text: str = ""
    main_text: str = ""
    error: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.success:
            return f"PipelineResult(success=True, references_count={len(self.references)})"
        return f"PipelineResult(success=False, error='{self.error}')"


def extract_references_from_url(
    pdf_url: str,
    citation_format: Union[str, CitationFormat],
    listing_style: Union[str, ListingStyle],
    mineru_config: Optional[MinerUConfig] = None,
    yayi_config: Optional[YaYiDocParserConfig] = None,
    download_timeout: int = 60,
) -> PipelineResult:
    """
    从 PDF URL 提取参考文献列表
    
    完整流水线：PDF URL -> 转换 Markdown -> 分离参考文献 -> 提取列表
    
    解析服务优先级：
    1. 雅意文档解析服务（如果配置了 yayi_config）
    2. MinerU API 服务（如果配置了 mineru_config.api_url）
    3. MinerU 本地模式
    
    Args:
        pdf_url: PDF 文件的 URL
        citation_format: 引用格式，支持:
            - "apa": APA 格式
            - "mla": MLA 格式
            - "ieee": IEEE 格式
            - "gb_t_7714": GB/T 7714 中国国标格式
            - "chicago": Chicago 格式
            - "harvard": Harvard 格式
            - "vancouver": Vancouver 格式
        listing_style: 条目列举方式，支持:
            - "numbered": 数字标号型 ([1], (1), 1. 等)
            - "author_year": 作者年份型（无数字标号）
        mineru_config: MinerU 配置（可选）
        yayi_config: 雅意文档解析服务配置（可选，优先使用）
        download_timeout: 下载超时时间（秒）
        
    Returns:
        PipelineResult 对象，包含:
        - success: 是否成功
        - references: 参考文献列表，每项为 [编号, 全文, 标题, 作者, 年份]（数字标号型）
                     或 [全文, 标题, 作者, 年份]（作者年份型）
        - references_text: 参考文献原文
        - main_text: 正文内容
        - error: 错误信息
        
    Example:
        # 使用雅意文档解析服务
        >>> from citeverify.models import YaYiDocParserConfig
        >>> yayi_config = YaYiDocParserConfig()
        >>> result = extract_references_from_url(
        ...     "https://example.com/paper.pdf",
        ...     citation_format="apa",
        ...     listing_style="numbered",
        ...     yayi_config=yayi_config,
        ... )
        
        # 使用 MinerU 服务
        >>> result = extract_references_from_url(
        ...     "https://arxiv.org/pdf/1706.03762.pdf",
        ...     citation_format="apa",
        ...     listing_style="numbered"
        ... )
        >>> if result.success:
        ...     for ref in result.references:
        ...         print(f"[{ref[0]}] {ref[2]} ({ref[4]})")
    """
    result = PipelineResult()
    
    try:
        # 1. 转换 PDF 为 Markdown
        converter = PDFConverter(
            mineru_config=mineru_config or MinerUConfig(),
            yayi_config=yayi_config,
        )
        
        conversion_result = converter.convert(
            pdf_url,
            yayi_config=yayi_config,
            download_timeout=download_timeout,
        )
        
        result.main_text = conversion_result.main_text
        result.references_text = conversion_result.references_text
        
        # 2. 检查是否成功分离参考文献
        if not conversion_result.separation_success:
            result.error = "无法从文档中分离出参考文献部分"
            return result
        
        if not conversion_result.references_text.strip():
            result.error = "参考文献部分为空"
            return result
        
        # 3. 提取参考文献列表
        refs = extract_references(
            conversion_result.references_text,
            listing_style=listing_style,
            citation_format=citation_format,
        )
        
        result.references = refs
        result.success = True
        
    except ImportError as e:
        result.error = f"缺少依赖: {e}"
    except ValueError as e:
        result.error = f"参数错误: {e}"
    except Exception as e:
        result.error = f"处理失败: {e}"
    
    return result


def extract_references_from_markdown(
    markdown_content: str,
    citation_format: Union[str, CitationFormat],
    listing_style: Union[str, ListingStyle],
) -> PipelineResult:
    """
    从 Markdown 内容提取参考文献列表
    
    流水线：Markdown -> 分离参考文献 -> 提取列表
    
    Args:
        markdown_content: Markdown 内容
        citation_format: 引用格式
        listing_style: 条目列举方式
        
    Returns:
        PipelineResult 对象
        
    Example:
        >>> result = extract_references_from_markdown(
        ...     markdown_text,
        ...     citation_format="gb_t_7714",
        ...     listing_style="numbered"
        ... )
    """
    result = PipelineResult()
    
    try:
        # 1. 分离参考文献
        conversion_result = separate_markdown(markdown_content)
        
        result.main_text = conversion_result.main_text
        result.references_text = conversion_result.references_text
        
        # 2. 检查是否成功分离
        if not conversion_result.separation_success:
            result.error = "无法从文档中分离出参考文献部分"
            return result
        
        # 3. 提取参考文献列表
        refs = extract_references(
            conversion_result.references_text,
            listing_style=listing_style,
            citation_format=citation_format,
        )
        
        result.references = refs
        result.success = True
        
    except Exception as e:
        result.error = f"处理失败: {e}"
    
    return result
