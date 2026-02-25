# -*- coding: utf-8 -*-
"""
PDF 转 Markdown 转换器

功能：
1. 使用 MinerU 将 PDF 转换为 Markdown
2. 分离正文与参考文献
3. 按标题分段，保证引用内容不割裂
4. 支持 URL 和本地文件路径
"""
import os
import re
import tempfile
import hashlib
from typing import Optional, List, Tuple, Union, Dict, Any
from pathlib import Path
from urllib.parse import urlparse

from ..models.document import (
    ConversionResult,
    Section,
    SeparationMethod,
    MinerUConfig,
    LLMConfig,
    YaYiDocParserConfig,
)


class PDFConverter:
    """PDF 转 Markdown 转换器
    
    支持多种 PDF 解析服务：
    1. 雅意文档解析服务（YaYi）：公司内部服务，需要文件公网 URL
    2. MinerU API 服务：开源方案，可 Docker 部署
    3. MinerU 本地模式：需要完整依赖
    """
    
    # 标题匹配关键词（宽松，用于章节标题匹配）
    REFERENCE_TITLE_KEYWORDS = [
        '参考文献', 'references', 'bibliography',
        '引用文献', '文献', 'works cited', 'citations',
    ]
    
    # 全文匹配关键词（严格，避免误匹配正文中的自然语言表达）
    # 只使用明确的参考文献标题词，不使用"文献"、"引用文献"等容易误匹配的词
    REFERENCE_KEYWORDS_STRICT = [
        r'^#{1,3}\s*参考文献\s*$',
        r'^#{1,3}\s*References\s*$',
        r'^#{1,3}\s*REFERENCES\s*$',
        r'^#{1,3}\s*Bibliography\s*$',
        r'^#{1,3}\s*BIBLIOGRAPHY\s*$',
        r'^#{1,3}\s*Works\s+Cited\s*$',
        # 无 markdown 标题标记的情况
        r'^\s*参考文献\s*$',
        r'^\s*References\s*$',
        r'^\s*REFERENCES\s*$',
        r'^\s*Bibliography\s*$',
        r'^\s*BIBLIOGRAPHY\s*$',
        r'^\s*Works\s+Cited\s*$',
    ]
    
    # 标题正则模式
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(
        self, 
        mineru_config: Optional[MinerUConfig] = None,
        yayi_config: Optional[YaYiDocParserConfig] = None,
    ):
        """
        初始化转换器
        
        Args:
            mineru_config: MinerU 配置
            yayi_config: 雅意文档解析服务配置（优先使用）
        """
        self.mineru_config = mineru_config or MinerUConfig()
        self.yayi_config = yayi_config
        # 编译严格的全文匹配正则（用于备选方案）
        self._reference_patterns_strict = [
            re.compile(p, re.MULTILINE | re.IGNORECASE) 
            for p in self.REFERENCE_KEYWORDS_STRICT
        ]
    
    @staticmethod
    def is_url(path: str) -> bool:
        """
        判断是否是 URL
        
        Args:
            path: 路径字符串
            
        Returns:
            是否是 URL
        """
        try:
            result = urlparse(path)
            return result.scheme in ('http', 'https')
        except Exception:
            return False
    
    @staticmethod
    def download_pdf(
        url: str,
        output_dir: Optional[str] = None,
        timeout: int = 60,
    ) -> Path:
        """
        从 URL 下载 PDF 文件
        
        Args:
            url: PDF 文件的 URL
            output_dir: 输出目录（默认使用临时目录）
            timeout: 下载超时时间（秒）
            
        Returns:
            下载后的本地文件路径
            
        Raises:
            ImportError: requests 未安装
            ValueError: URL 无效或下载失败
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests 未安装。请运行: pip install requests"
            )
        
        # 验证 URL
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            raise ValueError(f"无效的 URL: {url}")
        
        # 确定输出目录
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(tempfile.gettempdir()) / "citeverify_pdf"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名（使用 URL 的哈希值避免冲突）
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        # 尝试从 URL 中提取文件名
        url_path = parsed.path
        if url_path.endswith('.pdf'):
            filename = Path(url_path).name
            # 添加哈希避免重名
            filename = f"{Path(filename).stem}_{url_hash}.pdf"
        else:
            filename = f"document_{url_hash}.pdf"
        
        local_path = output_path / filename
        
        # 如果文件已存在，直接返回
        if local_path.exists():
            return local_path
        
        # 下载文件
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # 检查 Content-Type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and not url.lower().endswith('.pdf'):
                # 尝试检查文件头
                first_bytes = response.content[:8]
                if not first_bytes.startswith(b'%PDF'):
                    raise ValueError(f"URL 不是有效的 PDF 文件: {url}")
            
            # 写入文件
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return local_path
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"下载 PDF 失败: {url}\n错误: {e}")
    
    def convert(
        self,
        pdf_source: Union[str, Path],
        mineru_config: Optional[MinerUConfig] = None,
        yayi_config: Optional[YaYiDocParserConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        use_llm_fallback: bool = True,
        download_timeout: int = 60,
    ) -> ConversionResult:
        """
        转换 PDF 为 Markdown 并分离正文与参考文献
        
        解析服务优先级：
        1. 雅意文档解析服务（如果配置了 yayi_config）
        2. MinerU API 服务（如果配置了 mineru_config.api_url）
        3. MinerU 本地模式
        
        Args:
            pdf_source: PDF 文件路径或 URL
            mineru_config: MinerU 配置（覆盖初始化时的配置）
            yayi_config: 雅意文档解析服务配置（覆盖初始化时的配置）
            llm_config: LLM 配置（用于备用方案）
            use_llm_fallback: 是否启用 LLM 备用方案
            download_timeout: URL 下载超时时间（秒）
            
        Returns:
            ConversionResult 对象
        """
        mineru_cfg = mineru_config or self.mineru_config
        yayi_cfg = yayi_config or self.yayi_config
        
        # 优先使用雅意文档解析服务
        if yayi_cfg:
            # 雅意服务现在支持 URL 和本地文件
            # URL 会先下载到本地，然后上传到云端获取新 URL
            markdown_content = self._convert_via_yayi(pdf_source, yayi_cfg)
        else:
            # 使用 MinerU 服务
            pdf_path = pdf_source
            if isinstance(pdf_source, str) and self.is_url(pdf_source):
                pdf_path = self.download_pdf(
                    pdf_source, 
                    output_dir=mineru_cfg.output_dir,
                    timeout=download_timeout
                )
            
            markdown_content = self._convert_pdf_to_markdown(pdf_path, mineru_cfg)
        
        # 2. 按标题分段
        sections = self._split_by_headings(markdown_content)
        
        # 3. 尝试按关键字分离参考文献
        result = self._separate_by_keyword(markdown_content, sections)
        
        # 4. 如果关键字分离失败，尝试 LLM 备用方案
        if not result.separation_success and use_llm_fallback and llm_config:
            result = self._separate_by_llm(markdown_content, sections, llm_config)
        
        return result
    
    def convert_from_markdown(
        self,
        markdown_content: str,
        llm_config: Optional[LLMConfig] = None,
        use_llm_fallback: bool = True,
    ) -> ConversionResult:
        """
        从已有的 Markdown 内容分离正文与参考文献
        
        Args:
            markdown_content: Markdown 内容
            llm_config: LLM 配置（用于备用方案）
            use_llm_fallback: 是否启用 LLM 备用方案
            
        Returns:
            ConversionResult 对象
        """
        # 1. 按标题分段
        sections = self._split_by_headings(markdown_content)
        
        # 2. 尝试按关键字分离参考文献
        result = self._separate_by_keyword(markdown_content, sections)
        
        # 3. 如果关键字分离失败，尝试 LLM 备用方案
        if not result.separation_success and use_llm_fallback and llm_config:
            result = self._separate_by_llm(markdown_content, sections, llm_config)
        
        return result
    
    def _convert_pdf_to_markdown(
        self,
        pdf_path: Union[str, Path],
        config: MinerUConfig,
    ) -> str:
        """
        使用 MinerU 将 PDF 转换为 Markdown
        
        支持两种模式：
        1. API 模式：通过 HTTP 调用 MinerU 服务（推荐）
        2. 本地模式：直接使用 magic_pdf 库
        
        Args:
            pdf_path: PDF 文件路径
            config: MinerU 配置
            
        Returns:
            Markdown 内容
        """
        if config.api_url:
            # 使用 API 模式
            return self._convert_via_api(pdf_path, config)
        else:
            # 使用本地模式
            return self._convert_via_local(pdf_path, config)
    
    def _convert_via_api(
        self,
        pdf_path: Union[str, Path],
        config: MinerUConfig,
    ) -> str:
        """
        通过 MinerU API 服务转换 PDF
        
        API 服务地址示例: http://localhost:8000
        端点: /convert/markdown
        
        Args:
            pdf_path: PDF 文件路径
            config: MinerU 配置
            
        Returns:
            Markdown 内容
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests 未安装。请运行: pip install requests"
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        # 构建 API URL
        api_url = config.api_url.rstrip('/')
        endpoint = f"{api_url}/convert/markdown"
        
        # 准备请求
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path.name, f, 'application/pdf')}
            data = {
                'include_images': str(config.include_images).lower(),
            }
            
            try:
                response = requests.post(
                    endpoint,
                    files=files,
                    data=data,
                    timeout=config.timeout,
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise TimeoutError(
                    f"MinerU API 请求超时（{config.timeout}秒）。\n"
                    f"PDF 转换可能需要较长时间，可以尝试增加 timeout 配置。"
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"无法连接到 MinerU API 服务: {api_url}\n"
                    "请确保 MinerU API 服务已启动。\n"
                    "可以使用 Docker 启动: docker run --rm -p 8000:8000 erikvullings/mineru-api:gpu"
                )
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"MinerU API 返回错误: {response.status_code}\n"
                    f"响应内容: {response.text}\n"
                    f"原始错误: {e}"
                )
        
        # 解析响应
        content_type = response.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            # JSON 响应
            result = response.json()
            if isinstance(result, dict):
                # 可能包含 markdown 字段
                return result.get('markdown', result.get('content', str(result)))
            return str(result)
        else:
            # 纯文本响应
            return response.text
    
    def _convert_via_local(
        self,
        pdf_path: Union[str, Path],
        config: MinerUConfig,
    ) -> str:
        """
        使用本地 magic_pdf 库转换 PDF
        
        注意：此方法需要安装大量依赖，包括：
        - magic-pdf
        - rapid_table
        - doclayout_yolo
        - ultralytics
        - opencv-python
        等
        
        推荐使用 API 模式以避免依赖问题。
        
        Args:
            pdf_path: PDF 文件路径
            config: MinerU 配置
            
        Returns:
            Markdown 内容
        """
        try:
            # 动态导入 MinerU 模块
            from magic_pdf.data.data_reader_writer import (
                FileBasedDataWriter,
                FileBasedDataReader,
            )
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod
        except ImportError as e:
            missing_module = str(e).split("'")[-2] if "'" in str(e) else str(e)
            raise ImportError(
                f"缺少依赖: {e}\n\n"
                f"本地模式需要安装完整的 MinerU 依赖，包括模型文件。\n"
                f"推荐使用 API 模式来避免依赖问题：\n"
                f"  1. 启动 MinerU API 服务（Docker）:\n"
                f"     docker run --rm -p 8000:8000 erikvullings/mineru-api:gpu\n"
                f"  2. 配置 MinerUConfig(api_url='http://localhost:8000')\n"
            )
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        # 准备输出目录
        output_dir = Path(config.output_dir)
        image_dir = Path(config.image_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        image_dir_name = image_dir.name
        
        # 创建 writer
        image_writer = FileBasedDataWriter(str(image_dir))
        
        # 读取 PDF
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(str(pdf_path))
        
        # 创建数据集并处理
        ds = PymuDocDataset(pdf_bytes)
        
        # 根据配置决定是否使用 OCR
        if config.use_ocr is True:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        elif config.use_ocr is False:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        else:
            # 自动检测
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)
        
        # 获取 Markdown 内容
        md_content = pipe_result.get_markdown(image_dir_name)
        
        return md_content
    
    def _upload_file_to_cloud(
        self,
        local_file_path: Union[str, Path],
        config: YaYiDocParserConfig,
    ) -> str:
        """
        上传本地文件到云端存储，获取公网可访问的 URL
        
        Args:
            local_file_path: 本地文件路径
            config: 雅意文档解析服务配置（包含上传服务信息）
            
        Returns:
            上传后的公网 URL
            
        Raises:
            ImportError: 依赖未安装
            RuntimeError: 上传失败
        """
        try:
            import requests
            from requests_toolbelt import multipart
        except ImportError as e:
            raise ImportError(
                f"缺少依赖: {e}\n"
                "请运行: pip install requests requests-toolbelt"
            )
        
        local_file_path = Path(local_file_path)
        if not local_file_path.exists():
            raise FileNotFoundError(f"文件不存在: {local_file_path}")
        
        # 获取文件名
        file_name = local_file_path.name
        
        # 构建 multipart 请求
        with open(local_file_path, 'rb') as f:
            data = multipart.MultipartEncoder(
                fields={
                    'files': (file_name, f, 'multipart/form-data')
                }
            )
            
            headers = {
                "STORE_BUCKET_KEY": config.bucket_key,
                "Content-Type": data.content_type,
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"
            }
            
            try:
                response = requests.post(
                    config.upload_url,
                    data=data,
                    headers=headers,
                    timeout=config.timeout,
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise TimeoutError(
                    f"文件上传超时（{config.timeout}秒）。"
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"无法连接到文件上传服务: {config.upload_url}"
                )
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"文件上传服务返回 HTTP 错误: {response.status_code}\n"
                    f"响应内容: {response.text}\n"
                    f"原始错误: {e}"
                )
        
        # 解析响应
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(
                f"文件上传服务返回无效的 JSON 响应:\n{response.text}"
            )
        
        # 提取上传后的 URL
        data = result.get("data", [])
        if not data or len(data) == 0:
            raise RuntimeError(
                f"文件上传服务未返回有效的 URL。\n"
                f"完整响应: {result}"
            )
        
        uploaded_url = data[0]
        return uploaded_url
    
    def _remove_headers_footers(
        self,
        content_list: List[Dict[str, Any]],
        header_ratio: float = 0.08,  # 页面顶部 8% 区域
        footer_ratio: float = 0.08,  # 页面底部 8% 区域
        min_pages_for_detection: int = 3,  # 至少 3 页才进行检测
        y_tolerance: float = 5.0,  # y 坐标容差（像素）
        similarity_threshold: float = 0.6,  # 内容相似度阈值
    ) -> List[Dict[str, Any]]:
        """
        基于 bbox 和页面统计规则去除页眉页码
        
        检测规则：
        1. 页眉：页顶部区域内，跨页 y 位置一致、内容重复
        2. 页码：页底部区域内，跨页 y 位置一致、文本短且包含数字
        
        Args:
            content_list: API 返回的内容列表
            header_ratio: 页面顶部区域占比（默认 8%）
            footer_ratio: 页面底部区域占比（默认 8%）
            min_pages_for_detection: 最少页数才进行检测
            y_tolerance: y 坐标判定为"一致"的容差（像素）
            similarity_threshold: 内容判定为"重复"的相似度阈值
            
        Returns:
            过滤后的内容列表
        """
        import re
        from collections import defaultdict
        
        if not content_list or len(content_list) == 0:
            return content_list
        
        # 获取总页数
        page_nums = set(item.get("page_num", 1) for item in content_list)
        total_pages = max(page_nums) if page_nums else 1
        
        # 页数太少，不进行检测
        if total_pages < min_pages_for_detection:
            return content_list
        
        # 获取页面尺寸（从第一个有 shape 的元素获取）
        page_height = 791  # 默认值
        for item in content_list:
            shape = item.get("shape")
            if shape and len(shape) >= 2:
                page_height = shape[1]
                break
        
        # 计算页眉页脚区域边界
        header_boundary = page_height * header_ratio
        footer_boundary = page_height * (1 - footer_ratio)
        
        # 分析每页的顶部和底部文本块
        header_candidates = defaultdict(list)  # y_position -> [(page_num, text, item)]
        footer_candidates = defaultdict(list)  # y_position -> [(page_num, text, item)]
        
        for item in content_list:
            bbox = item.get("bbox", [])
            page_num = item.get("page_num", 1)
            text = item.get("text", "").strip()
            
            if not bbox or len(bbox) < 4 or not text:
                continue
            
            # 获取文本块的 y 坐标范围
            # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            y_coords = [point[1] for point in bbox if len(point) >= 2]
            if not y_coords:
                continue
            
            y_min = min(y_coords)
            y_max = max(y_coords)
            y_center = (y_min + y_max) / 2
            
            # 检查是否在页眉区域（顶部）
            if y_max <= header_boundary:
                # 将 y 坐标四舍五入到容差范围
                y_key = round(y_center / y_tolerance) * y_tolerance
                header_candidates[y_key].append((page_num, text, item))
            
            # 检查是否在页脚区域（底部）
            elif y_min >= footer_boundary:
                y_key = round(y_center / y_tolerance) * y_tolerance
                footer_candidates[y_key].append((page_num, text, item))
        
        # 识别要删除的项
        items_to_remove = set()
        
        # 分析页眉候选
        for y_pos, candidates in header_candidates.items():
            if len(candidates) < min_pages_for_detection:
                continue
            
            # 检查是否跨多页出现在相似位置
            pages_with_content = set(c[0] for c in candidates)
            
            # 如果超过 50% 的页面在此位置有内容，可能是页眉
            if len(pages_with_content) >= total_pages * 0.5:
                # 进一步检查内容是否重复/相似
                texts = [c[1] for c in candidates]
                if self._check_content_repetition(texts, similarity_threshold):
                    # 标记为页眉，需要删除
                    for _, _, item in candidates:
                        items_to_remove.add(id(item))
        
        # 分析页脚/页码候选
        for y_pos, candidates in footer_candidates.items():
            if len(candidates) < min_pages_for_detection:
                continue
            
            pages_with_content = set(c[0] for c in candidates)
            
            # 如果超过 50% 的页面在此位置有内容
            if len(pages_with_content) >= total_pages * 0.5:
                texts = [c[1] for c in candidates]
                
                # 页码特征：文本短且主要包含数字
                is_page_number = self._check_page_number_pattern(texts)
                
                # 或者内容重复（页脚文本）
                is_repeated = self._check_content_repetition(texts, similarity_threshold)
                
                if is_page_number or is_repeated:
                    for _, _, item in candidates:
                        items_to_remove.add(id(item))
        
        # 过滤结果
        filtered_content = [
            item for item in content_list 
            if id(item) not in items_to_remove
        ]
        
        # 输出过滤统计
        removed_count = len(content_list) - len(filtered_content)
        if removed_count > 0:
            print(f"页眉页码过滤: 移除了 {removed_count} 个文本块")
        
        return filtered_content
    
    def _check_content_repetition(
        self,
        texts: List[str],
        similarity_threshold: float = 0.6,
    ) -> bool:
        """
        检查文本列表是否存在重复/相似内容
        
        Args:
            texts: 文本列表
            similarity_threshold: 相似度阈值
            
        Returns:
            是否存在重复
        """
        if len(texts) < 2:
            return False
        
        # 方法1：完全相同
        unique_texts = set(texts)
        if len(unique_texts) == 1:
            return True
        
        # 方法2：去除数字后相同（处理带页码的页眉如 "Title - Page 1"）
        def normalize_text(t):
            import re
            # 移除数字、空白、标点
            return re.sub(r'[\d\s\-–—·•.,，。、]', '', t.lower())
        
        normalized = [normalize_text(t) for t in texts]
        unique_normalized = set(normalized)
        
        # 如果标准化后只有 1-2 种变体，认为是重复
        if len(unique_normalized) <= 2 and len(normalized) >= 3:
            return True
        
        # 方法3：计算相似度（简单的 Jaccard 相似度）
        def jaccard_similarity(s1, s2):
            set1 = set(s1)
            set2 = set(s2)
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0
        
        # 检查是否大多数文本相似
        similar_pairs = 0
        total_pairs = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                total_pairs += 1
                if jaccard_similarity(texts[i], texts[j]) >= similarity_threshold:
                    similar_pairs += 1
        
        if total_pairs > 0 and similar_pairs / total_pairs >= 0.5:
            return True
        
        return False
    
    def _check_page_number_pattern(
        self,
        texts: List[str],
        max_length: int = 20,  # 页码文本最大长度
    ) -> bool:
        """
        检查文本列表是否符合页码模式
        
        页码特征：
        - 文本较短（通常 < 20 字符）
        - 主要包含数字
        - 数字递增或为连续序列
        
        Args:
            texts: 文本列表
            max_length: 页码文本最大长度
            
        Returns:
            是否是页码
        """
        import re
        
        if not texts:
            return False
        
        # 检查文本长度
        avg_length = sum(len(t) for t in texts) / len(texts)
        if avg_length > max_length:
            return False
        
        # 提取所有数字
        numbers = []
        digit_ratio_list = []
        
        for text in texts:
            # 计算数字占比
            digits = re.findall(r'\d+', text)
            total_digits = sum(len(d) for d in digits)
            text_len = len(text.replace(' ', ''))
            
            if text_len > 0:
                digit_ratio = total_digits / text_len
                digit_ratio_list.append(digit_ratio)
            
            # 提取主要数字（通常是页码）
            if digits:
                # 取最大的数字（可能是页码）
                numbers.append(max(int(d) for d in digits))
        
        # 如果大多数文本的数字占比较高，可能是页码
        if digit_ratio_list:
            avg_digit_ratio = sum(digit_ratio_list) / len(digit_ratio_list)
            if avg_digit_ratio >= 0.3:  # 30% 以上是数字
                return True
        
        # 检查数字是否递增（页码通常是连续的）
        if len(numbers) >= 3:
            # 排序后检查是否接近连续
            sorted_nums = sorted(numbers)
            diffs = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
            
            # 如果差值主要是 1，说明是连续页码
            if diffs and sum(1 for d in diffs if d == 1) / len(diffs) >= 0.5:
                return True
        
        # 常见页码模式
        page_patterns = [
            r'^\d+$',  # 纯数字
            r'^-\s*\d+\s*-$',  # - 1 -
            r'^page\s*\d+',  # Page 1
            r'^\d+\s*/\s*\d+$',  # 1/10
            r'^第\s*\d+\s*页',  # 第 1 页
            r'^\[\d+\]$',  # [1]
        ]
        
        pattern_matches = 0
        for text in texts:
            text_lower = text.lower().strip()
            for pattern in page_patterns:
                if re.match(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1
                    break
        
        # 如果超过一半的文本匹配页码模式
        if pattern_matches / len(texts) >= 0.5:
            return True
        
        return False

    def _aggregate_content_by_title(
        self,
        content_list: List[Dict[str, Any]],
        remove_headers_footers: bool = True,
        header_ratio: float = 0.08,
        footer_ratio: float = 0.08,
    ) -> str:
        """
        将解析结果按标题聚合生成 Markdown
        
        API 返回的 content 结构示例：
        {
            "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            "index": 0.5,
            "type": "title" | "text" | "table" | "figure" | ...,
            "text": "内容文本",
            "title": "所属标题（如果 type 是 title，则为标题本身）",
            "para_num": 1,
            "shape": [612, 791],
            "page_num": 1
        }
        
        Args:
            content_list: API 返回的内容列表
            remove_headers_footers: 是否去除页眉页码
            header_ratio: 页眉检测区域（页面顶部占比）
            footer_ratio: 页脚检测区域（页面底部占比）
            
        Returns:
            聚合后的 Markdown 内容
        """
        from collections import OrderedDict
        
        if not content_list:
            return ""
        
        # 先过滤页眉页码
        if remove_headers_footers:
            content_list = self._remove_headers_footers(
                content_list,
                header_ratio=header_ratio,
                footer_ratio=footer_ratio,
            )
        
        # 按页码和索引排序
        sorted_content = sorted(
            content_list,
            key=lambda x: (x.get("page_num", 0), x.get("index", 0))
        )
        
        # 按标题聚合内容
        # 使用 OrderedDict 保持标题出现的顺序
        sections = OrderedDict()
        current_title = None
        
        for item in sorted_content:
            item_type = item.get("type", "text")
            text = item.get("text", "").strip()
            title = item.get("title", "").strip()
            
            if not text:
                continue
            
            # 如果是标题类型，设置当前标题
            if item_type == "title":
                current_title = text
                # 如果标题还没加入 sections，初始化
                if current_title not in sections:
                    sections[current_title] = []
            else:
                # 非标题内容
                # 如果有关联的 title 字段且不是当前标题，使用该 title
                if title and title != current_title:
                    current_title = title
                    if current_title not in sections:
                        sections[current_title] = []
                
                # 如果还没有标题，使用默认标题
                if current_title is None:
                    current_title = ""
                    if current_title not in sections:
                        sections[current_title] = []
                
                # 根据类型添加内容
                if item_type == "table":
                    # 表格内容，保持原样
                    sections[current_title].append(text)
                elif item_type == "figure":
                    # 图片描述
                    sections[current_title].append(f"[图片: {text}]")
                else:
                    # 普通文本
                    sections[current_title].append(text)
        
        # 生成 Markdown
        markdown_lines = []
        
        for title, contents in sections.items():
            if title:
                # 判断标题级别（简单启发式：包含数字前缀的可能是子标题）
                level = self._detect_heading_level(title)
                markdown_lines.append(f"{'#' * level} {title}")
                markdown_lines.append("")  # 标题后空行
            
            # 添加内容
            for content in contents:
                if content:
                    markdown_lines.append(content)
                    markdown_lines.append("")  # 段落间空行
        
        return "\n".join(markdown_lines)
    
    def _detect_heading_level(self, title: str) -> int:
        """
        检测标题级别
        
        简单启发式规则：
        - 以大写字母开头且全大写：一级标题
        - 以数字开头（如 "1.", "1.1"）：根据层级判断
        - 其他：二级标题
        
        Args:
            title: 标题文本
            
        Returns:
            标题级别 (1-6)
        """
        import re
        
        title = title.strip()
        
        # 全大写标题通常是一级标题
        if title.isupper() and len(title) > 3:
            return 1
        
        # 检测数字前缀 (如 "1.", "1.1", "1.1.1")
        match = re.match(r'^(\d+(?:\.\d+)*)\s*[.、]?\s*', title)
        if match:
            num_parts = match.group(1).split('.')
            level = len(num_parts)
            return min(level + 1, 6)  # 限制最大为 6
        
        # 一些常见的一级标题关键词
        level_1_keywords = [
            'abstract', 'introduction', 'conclusion', 'references',
            'acknowledgement', 'acknowledgments', 'appendix',
            '摘要', '引言', '结论', '参考文献', '致谢', '附录',
        ]
        
        if title.lower() in level_1_keywords:
            return 1
        
        # 默认二级标题
        return 2

    def _convert_via_yayi(
        self,
        pdf_source: Union[str, Path],
        config: YaYiDocParserConfig,
    ) -> str:
        """
        通过雅意文档解析服务转换 PDF
        
        工作流程：
        1. 如果是 URL，先下载到本地
        2. 上传本地文件到云端存储获取公网 URL
        3. 使用公网 URL 调用分析 API
        4. 将返回的内容按标题聚合生成 Markdown
        5. 清理临时文件
        
        Args:
            pdf_source: PDF 文件路径或 URL
            config: 雅意文档解析服务配置
            
        Returns:
            解析后的 Markdown 内容
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests 未安装。请运行: pip install requests"
            )
        
        import uuid
        
        local_file_path = None
        temp_file_created = False
        
        try:
            # 1. 确保有本地文件
            if isinstance(pdf_source, str) and self.is_url(pdf_source):
                # 从 URL 下载到本地临时文件
                local_file_path = self.download_pdf(pdf_source)
                temp_file_created = True
            else:
                local_file_path = Path(pdf_source)
                if not local_file_path.exists():
                    raise FileNotFoundError(f"PDF 文件不存在: {local_file_path}")
            
            # 2. 上传文件到云端获取公网 URL
            uploaded_url = self._upload_file_to_cloud(local_file_path, config)
            
            # 3. 调用分析 API
            task_id = str(uuid.uuid4())
            
            request_body = {
                "id": task_id,
                "url": uploaded_url,
                "content": {
                    "mode": config.mode,
                }
            }
            
            headers = {
                "Content-Type": "application/json",
            }
            
            try:
                response = requests.post(
                    config.api_url,
                    json=request_body,
                    headers=headers,
                    timeout=config.timeout,
                )
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise TimeoutError(
                    f"雅意文档解析服务请求超时（{config.timeout}秒）。\n"
                    f"PDF 解析可能需要较长时间，可以尝试增加 timeout 配置。"
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"无法连接到雅意文档解析服务: {config.api_url}\n"
                    "请检查网络连接或服务地址是否正确。"
                )
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"雅意文档解析服务返回 HTTP 错误: {response.status_code}\n"
                    f"响应内容: {response.text}\n"
                    f"原始错误: {e}"
                )
            
            # 4. 解析响应
            try:
                result = response.json()
            except ValueError:
                raise RuntimeError(
                    f"雅意文档解析服务返回无效的 JSON 响应:\n{response.text}"
                )
            
            # 检查返回码
            code = result.get("code")
            if code != "200" and code != 200:
                msg = result.get("msg", "未知错误")
                raise RuntimeError(
                    f"雅意文档解析服务返回错误 (code={code}): {msg}\n"
                    f"完整响应: {result}"
                )
            
            # 5. 提取并处理内容
            data = result.get("data", {})
            file_content = data.get("file_content", {})
            content = file_content.get("content", [])
            
            # 检查是否有超时截断
            if data.get("timebreak"):
                print(f"警告: 雅意文档解析服务因超时截断了输出内容")
            
            if not content:
                raise RuntimeError(
                    f"雅意文档解析服务返回空内容。\n"
                    f"请检查 PDF 是否有效。\n"
                    f"完整响应: {result}"
                )
            
            # 6. 根据 mode 处理返回内容
            if config.mode == 0:
                # 段落模式：返回的是带结构化信息的列表，需要按标题聚合
                if isinstance(content, list):
                    markdown_content = self._aggregate_content_by_title(
                        content,
                        remove_headers_footers=config.remove_headers_footers,
                        header_ratio=config.header_ratio,
                        footer_ratio=config.footer_ratio,
                    )
                else:
                    # 兼容处理：如果返回的是字符串
                    markdown_content = str(content)
            else:
                # 其他模式：mode=1 全文模式返回 string
                if isinstance(content, list):
                    markdown_content = "\n".join(str(item) for item in content)
                else:
                    markdown_content = str(content)
            
            return markdown_content
            
        finally:
            # 清理临时文件
            if temp_file_created and local_file_path and local_file_path.exists():
                try:
                    os.remove(local_file_path)
                except Exception:
                    pass  # 忽略清理失败
    
    def _split_by_headings(self, markdown_content: str) -> List[Section]:
        """
        按标题分段
        
        Args:
            markdown_content: Markdown 内容
            
        Returns:
            章节列表
        """
        sections = []
        lines = markdown_content.split('\n')
        
        current_section = None
        current_content_lines = []
        current_start = 0
        
        for i, line in enumerate(lines):
            heading_match = self.HEADING_PATTERN.match(line)
            
            if heading_match:
                # 保存之前的章节
                if current_section is not None:
                    current_section.content = '\n'.join(current_content_lines).strip()
                    current_section.end_pos = sum(len(l) + 1 for l in lines[:i])
                    sections.append(current_section)
                
                # 开始新章节
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_start = sum(len(l) + 1 for l in lines[:i])
                current_section = Section(
                    title=title,
                    content="",
                    level=level,
                    start_pos=current_start,
                )
                current_content_lines = []
            else:
                current_content_lines.append(line)
        
        # 保存最后一个章节
        if current_section is not None:
            current_section.content = '\n'.join(current_content_lines).strip()
            current_section.end_pos = len(markdown_content)
            sections.append(current_section)
        elif current_content_lines:
            # 没有标题的情况，整个内容作为一个章节
            sections.append(Section(
                title="",
                content='\n'.join(current_content_lines).strip(),
                level=0,
                start_pos=0,
                end_pos=len(markdown_content),
            ))
        
        return sections
    
    def _separate_by_keyword(
        self,
        markdown_content: str,
        sections: List[Section],
    ) -> ConversionResult:
        """
        按关键字分离正文与参考文献
        
        匹配策略（按优先级）：
        1. 优先：章节标题匹配（使用宽松关键词，标题匹配更可靠）
        2. 备选：全文正则匹配（使用严格关键词，避免误匹配正文自然语言）
        
        Args:
            markdown_content: Markdown 内容
            sections: 章节列表
            
        Returns:
            ConversionResult 对象
        """
        result = ConversionResult(
            full_markdown=markdown_content,
            sections=sections,
            separation_method=SeparationMethod.KEYWORD,
        )
        
        # 第一步：优先通过章节标题匹配（使用宽松关键词）
        for i, section in enumerate(sections):
            if self._is_reference_section(section.title):

                start_pos = section.start_pos

                # 新增：确定结束位置（下一个章节）
                if i + 1 < len(sections):
                    end_pos = sections[i + 1].start_pos
                else:
                    end_pos = len(markdown_content)

                result.reference_start_pos = start_pos
                result.main_text = markdown_content[:start_pos].strip()
                result.references_text = markdown_content[start_pos:end_pos].strip()
                result.separation_success = True
                return result

        # 第二步：备选全文正则匹配（使用严格关键词）
        reference_pos = -1
        
        for pattern in self._reference_patterns_strict:
            match = pattern.search(markdown_content)
            if match:
                pos = match.start()
                # 选择最靠后的位置（参考文献通常在文末）
                if reference_pos == -1 or pos > reference_pos:
                    reference_pos = pos

        if reference_pos != -1:
            result.reference_start_pos = reference_pos
            result.main_text = markdown_content[:reference_pos].strip()

            # 新增：寻找下一个章节标题
            next_section_pos = None
            for section in sections:
                if section.start_pos > reference_pos:
                    next_section_pos = section.start_pos
                    break

            if next_section_pos:
                result.references_text = markdown_content[
                                         reference_pos:next_section_pos
                                         ].strip()
            else:
                result.references_text = markdown_content[
                                         reference_pos:
                                         ].strip()

            result.separation_success = True
            return result
        
        # 分离失败，整个内容作为正文
        result.main_text = markdown_content
        result.references_text = ""
        
        return result
    
    def _is_reference_section(self, title: str) -> bool:
        """
        判断是否是参考文献章节（使用宽松关键词匹配）
        
        Args:
            title: 章节标题
            
        Returns:
            是否是参考文献章节
        """
        if not title:
            return False
        
        title_lower = title.lower().strip()
        return title_lower in self.REFERENCE_TITLE_KEYWORDS
    
    def _separate_by_llm(
        self,
        markdown_content: str,
        sections: List[Section],
        llm_config: LLMConfig,
    ) -> ConversionResult:
        """
        使用 LLM 分离正文与参考文献（备用方案）
        
        Args:
            markdown_content: Markdown 内容
            sections: 章节列表
            llm_config: LLM 配置
            
        Returns:
            ConversionResult 对象
        
        Note:
            当前为占位实现，实际功能待后续完善
        """
        result = ConversionResult(
            full_markdown=markdown_content,
            sections=sections,
            separation_method=SeparationMethod.LLM,
            separation_success=False,
        )
        
        # TODO: 实现 LLM 分离逻辑
        # 大致思路：
        # 1. 将 markdown 内容（或章节标题列表）发送给 LLM
        # 2. 让 LLM 识别参考文献的起始位置
        # 3. 根据 LLM 返回的位置进行分割
        
        # 占位实现：调用 LLM API 定位参考文献
        try:
            reference_pos = self._call_llm_for_reference_position(
                markdown_content, sections, llm_config
            )
            
            if reference_pos is not None and reference_pos > 0:
                result.reference_start_pos = reference_pos
                result.main_text = markdown_content[:reference_pos].strip()
                result.references_text = markdown_content[reference_pos:].strip()
                result.separation_success = True
            else:
                # LLM 也无法定位
                result.separation_method = SeparationMethod.FAILED
                result.main_text = markdown_content
                result.references_text = ""
                
        except Exception as e:
            # LLM 调用失败
            result.separation_method = SeparationMethod.FAILED
            result.main_text = markdown_content
            result.references_text = ""
            result.metadata['llm_error'] = str(e)
        
        return result
    
    def _call_llm_for_reference_position(
        self,
        markdown_content: str,
        sections: List[Section],
        llm_config: LLMConfig,
    ) -> Optional[int]:
        """
        调用 LLM 定位参考文献位置
        
        Args:
            markdown_content: Markdown 内容
            sections: 章节列表
            llm_config: LLM 配置
            
        Returns:
            参考文献起始位置，如果无法定位则返回 None
            
        Note:
            当前为占位实现
        """
        # TODO: 实现实际的 LLM 调用逻辑
        # 
        # 示例实现思路:
        # 
        # import openai
        # 
        # client = openai.OpenAI(
        #     api_key=llm_config.api_key,
        #     base_url=llm_config.api_base,
        # )
        # 
        # # 构建章节列表用于 LLM 分析
        # sections_info = []
        # for i, section in enumerate(sections):
        #     sections_info.append(f"{i}. [{section.start_pos}] {section.title}")
        # 
        # prompt = f"""
        # 以下是一篇学术论文的章节列表，请识别参考文献部分的起始位置。
        # 
        # 章节列表：
        # {chr(10).join(sections_info)}
        # 
        # 请返回参考文献章节的起始位置（数字），如果找不到参考文献章节，返回 -1。
        # 只返回数字，不要有其他内容。
        # """
        # 
        # response = client.chat.completions.create(
        #     model=llm_config.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     timeout=llm_config.timeout,
        # )
        # 
        # try:
        #     position = int(response.choices[0].message.content.strip())
        #     return position if position >= 0 else None
        # except ValueError:
        #     return None
        
        # 占位返回
        return None


def convert_pdf_to_markdown(
    pdf_source: Union[str, Path],
    mineru_config: Optional[MinerUConfig] = None,
    llm_config: Optional[LLMConfig] = None,
    use_llm_fallback: bool = True,
    download_timeout: int = 60,
) -> ConversionResult:
    """
    便捷函数：转换 PDF 为 Markdown 并分离正文与参考文献
    
    Args:
        pdf_source: PDF 文件路径或 URL
        mineru_config: MinerU 配置
        llm_config: LLM 配置（用于备用方案）
        use_llm_fallback: 是否启用 LLM 备用方案
        download_timeout: URL 下载超时时间（秒）
        
    Returns:
        ConversionResult 对象
    """
    converter = PDFConverter(mineru_config)
    return converter.convert(
        pdf_source, 
        llm_config=llm_config, 
        use_llm_fallback=use_llm_fallback,
        download_timeout=download_timeout
    )


def download_pdf_from_url(
    url: str,
    output_dir: Optional[str] = None,
    timeout: int = 60,
) -> Path:
    """
    便捷函数：从 URL 下载 PDF 文件
    
    Args:
        url: PDF 文件的 URL
        output_dir: 输出目录（默认使用临时目录）
        timeout: 下载超时时间（秒）
        
    Returns:
        下载后的本地文件路径
    """
    return PDFConverter.download_pdf(url, output_dir, timeout)


def separate_markdown(
    markdown_content: str,
    llm_config: Optional[LLMConfig] = None,
    use_llm_fallback: bool = True,
) -> ConversionResult:
    """
    便捷函数：从 Markdown 内容分离正文与参考文献
    
    Args:
        markdown_content: Markdown 内容
        llm_config: LLM 配置（用于备用方案）
        use_llm_fallback: 是否启用 LLM 备用方案
        
    Returns:
        ConversionResult 对象
    """
    converter = PDFConverter()
    return converter.convert_from_markdown(
        markdown_content, 
        llm_config=llm_config, 
        use_llm_fallback=use_llm_fallback
    )
