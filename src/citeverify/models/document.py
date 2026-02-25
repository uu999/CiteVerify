# -*- coding: utf-8 -*-
"""
文档数据模型
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class SeparationMethod(Enum):
    """参考文献分离方法"""
    KEYWORD = "keyword"  # 关键字定位
    LLM = "llm"  # 大模型定位
    MANUAL = "manual"  # 手动指定
    FAILED = "failed"  # 分离失败


@dataclass
class Section:
    """
    文档章节
    
    Attributes:
        title: 章节标题
        content: 章节内容
        level: 标题级别 (1-6 对应 # - ######)
        start_pos: 在原文中的起始位置
        end_pos: 在原文中的结束位置
    """
    title: str
    content: str
    level: int = 1
    start_pos: int = 0
    end_pos: int = 0
    
    def __repr__(self) -> str:
        return f"Section(title='{self.title[:30]}...', level={self.level})"


@dataclass
class ConversionResult:
    """
    PDF 转换结果
    
    Attributes:
        full_markdown: 完整的 Markdown 内容
        main_text: 正文部分（不含参考文献）
        references_text: 参考文献部分
        sections: 按标题分段的章节列表
        separation_method: 分离方法
        separation_success: 是否成功分离正文与参考文献
        reference_start_pos: 参考文献在原文中的起始位置
        metadata: 其他元数据
    """
    full_markdown: str
    main_text: str = ""
    references_text: str = ""
    sections: List[Section] = field(default_factory=list)
    separation_method: SeparationMethod = SeparationMethod.KEYWORD
    separation_success: bool = False
    reference_start_pos: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"ConversionResult(separation_success={self.separation_success}, "
            f"main_text_len={len(self.main_text)}, "
            f"references_len={len(self.references_text)})"
        )


@dataclass
class MinerUConfig:
    """
    MinerU 配置
    
    支持两种模式：
    1. API 模式（推荐）：通过 HTTP 调用 MinerU 服务
    2. 本地模式：直接使用 magic_pdf 库（需要安装大量依赖）
    
    Attributes:
        api_url: MinerU API 服务地址（如 "http://localhost:8000"）
                 如果设置了此参数，则使用 API 模式
        output_dir: 输出目录（本地模式使用）
        image_dir: 图片输出目录（本地模式使用）
        use_ocr: 是否强制使用 OCR（None 表示自动检测）
        include_images: 是否在 markdown 中包含图片（API 模式使用）
        timeout: API 请求超时时间（秒）
    """
    api_url: Optional[str] = None  # 设置后使用 API 模式
    output_dir: str = "output"
    image_dir: str = "output/images"
    use_ocr: Optional[bool] = None
    include_images: bool = False  # API 模式下是否包含 base64 图片
    timeout: int = 300  # API 超时时间（PDF 转换可能较慢）


@dataclass 
class LLMConfig:
    """
    LLM 配置（用于备用方案）
    
    Attributes:
        api_key: API 密钥
        api_base: API 基础 URL
        model: 模型名称
        timeout: 请求超时时间（秒）
    """
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: int = 60


@dataclass
class YaYiDocParserConfig:
    """
    雅意文档解析服务配置
    
    公司内部部署的文档解析服务，支持多种文档格式（PDF、Word、PPT等）
    
    工作流程：
    1. 下载原始 PDF URL 到本地临时文件
    2. 上传临时文件到云端存储获取新的 URL
    3. 使用新 URL 调用分析 API
    4. 基于 bbox 过滤页眉页码
    5. 将返回的内容按标题聚合生成 Markdown
    
    Attributes:
        api_url: 文档分析 API 服务地址
        upload_url: 文件上传服务地址（获取云端 URL）
        bucket_key: 文件上传服务的 bucket key
        mode: 输出模式
            - 0: 段落模式（输出带 bbox、type、text 等字段的 list）
            - 1: 全文模式（输出 string）
            - 2: 分页模式
            - 3: PDF 全文解析（无版面分析无 OCR）
        webmode: 网页解析输出模式（仅针对网页解析）
            - "txt": 纯文本
            - "html": HTML 格式
            - "markdown": Markdown 格式（默认）
        watermark: 是否去除水印（0: 否，1: 是）
        formula: 是否识别公式（0: 否，1: 是）
        finetable: 是否采用精细表格识别（0: 否，1: 是）
        charttext: 是否加入图表解析（0: 否，1: 是）
        timeout: 请求超时时间（秒）
        remove_headers_footers: 是否去除页眉页码
        header_ratio: 页眉检测区域（页面顶部占比）
        footer_ratio: 页脚检测区域（页面底部占比）
    """
    api_url: str = "http://172.16.10.33:31017/analysis"  # 文档分析 API 地址,备用：http://172.16.10.69:3009/analysis
    upload_url: str = "http://hongqiplus.wengegroup.com/mam/api/file/getUrl"  # 文件上传服务地址
    bucket_key: str = "zs-a3efde7e"  # 文件上传 bucket key
    mode: int = 0  # 段落模式，获取结构化内容
    webmode: str = "markdown"  # 网页解析输出模式
    watermark: int = 0  # 不去除水印
    formula: int = 0  # 不识别公式
    finetable: int = 0  # 不采用精细表格识别
    charttext: int = 0  # 不加入图表解析
    timeout: int = 300  # 超时时间（秒）
    # 页眉页码过滤配置
    remove_headers_footers: bool = True  # 是否去除页眉页码
    header_ratio: float = 0.08  # 页眉检测区域（页面顶部 8%）
    footer_ratio: float = 0.08  # 页脚检测区域（页面底部 8%）
