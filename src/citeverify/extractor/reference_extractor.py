# -*- coding: utf-8 -*-
"""
参考文献抽取器

支持的条目列举方式：
- 数字标号型: 1. xxx, [1] xxx, (1) xxx, （1）xxx
- 作者年份型（无数字标号）

支持的引用格式：
- MLA, APA, GB/T 7714, IEEE, Chicago, Harvard, Vancouver, other（其他格式由 LLM 全文提取）

支持 LLM 备选提取：当规则提取失败时，使用 LLM 提取元数据；格式为 other 时直接由 LLM 从全文提取
"""
import re
import json
import logging
from typing import List, Optional, Tuple, Dict, Any

from ..models.reference import (
    ReferenceEntry,
    ListingStyle,
    CitationFormat
)

logger = logging.getLogger(__name__)


class ReferenceExtractor:
    """参考文献抽取器"""
    
    # 数字标号正则模式（限制 1-3 位数字，避免误匹配）
    NUMBERED_PATTERNS = [
        r'^\s*\[(\d{1,3})\]\s*',        # [1] xxx
        r'^\s*（(\d{1,3})）\s*',         # （1）xxx (中文括号)
        r'^\s*\((\d{1,3})\)\s*',         # (1) xxx
        r'^\s*(\d{1,3})\.\s+',           # 1. xxx
        r'^\s*(\d{1,3})\s+',             # 1 xxx (数字后跟空格)
    ]
    
    # 年份正则
    YEAR_PATTERN = r'\b(19\d{2}|20\d{2})\b'
    
    # 无数字标号条目开头模式（增强版）
    AUTHOR_START_PATTERNS = [
        # 英文作者：姓, 名首字母（如 Smith, J. 或 Smith J.）
        r'^[A-Z][a-zA-Z\'\-]+,?\s*[A-Z]\.',
        # 英文作者：姓 名首字母（无逗号，如 Smith J）
        r'^[A-Z][a-zA-Z\'\-]+\s+[A-Z]\s',
        # 全大写作者（如 SMITH J.）
        r'^[A-Z]{2,}[,\s]+[A-Z]\.',
        # 机构作者（如 World Health Organization.）
        r'^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,5}\s*[.,]',
        # 带 van/von/de 等前缀的作者（如 van der Berg, A.）
        r'^(?:van|von|de|del|der|la|le)\s+[A-Z][a-zA-Z\'\-]+,?\s*[A-Z]\.',
        # 中文作者（2-4字 + 逗号或句号）
        r'^[\u4e00-\u9fa5]{2,4}[,，.。]',
        # 中文作者（2-4字 + 空格/等）
        r'^[\u4e00-\u9fa5]{2,4}(?:\s|等)',
        # APA 风格年份开头（如 Author, A. (2020).）- 检测年份括号
        r'^[A-Z][a-zA-Z\'\-]+.*?\(\d{4}\)',
    ]
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        use_llm_fallback: bool = True,
    ):
        """
        初始化抽取器
        
        Args:
            llm_model: LLM 模型名称（如 gpt-4, qwen-turbo 等）
            llm_api_key: LLM API 密钥
            llm_base_url: LLM API 基础 URL（OpenAI 兼容格式）
            use_llm_fallback: 规则提取失败时是否使用 LLM 备选提取
        """
        # 编译正则表达式
        self._numbered_regexes = [re.compile(p) for p in self.NUMBERED_PATTERNS]
        self._year_regex = re.compile(self.YEAR_PATTERN)
        self._author_start_regexes = [re.compile(p, re.IGNORECASE) for p in self.AUTHOR_START_PATTERNS]
        
        # LLM 配置
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._use_llm_fallback = use_llm_fallback
        self._llm_client = None
        
        # 初始化 LLM 客户端（如果配置了）
        if llm_model and llm_api_key:
            self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        try:
            from openai import OpenAI
            self._llm_client = OpenAI(
                api_key=self._llm_api_key,
                base_url=self._llm_base_url,
            )
            logger.info(f"LLM 客户端初始化成功: model={self._llm_model}")
        except ImportError:
            logger.warning("openai 未安装，LLM 备选提取不可用")
            self._llm_client = None
        except Exception as e:
            logger.warning(f"LLM 客户端初始化失败: {e}")
            self._llm_client = None
    
    def _extract_via_llm(self, content: str) -> Dict[str, Any]:
        """
        使用 LLM 提取参考文献元数据
        
        Args:
            content: 参考文献条目内容
            
        Returns:
            提取结果字典 {"title": ..., "authors": [...], "year": ...}
        """
        if not self._llm_client:
            return {}
        
        prompt = f"""请分析以下参考文献条目，提取其中的元数据信息。

参考文献条目：
{content}

请识别该参考文献的引用格式（如 APA, MLA, IEEE, GB/T 7714, Chicago, Harvard, Vancouver 等），
然后提取以下信息：
1. title: 论文/文章/书籍的标题（完整标题，不要截断）
2. authors: 作者列表（数组格式，每个作者一个元素）
3. year: 发表年份（4位数字）

请严格以 JSON 格式输出，不要包含其他内容：
{{"title": "论文标题", "authors": ["作者1", "作者2"], "year": "2020"}}

注意：
- 如果某个字段无法提取，设为 null
- authors 必须是数组格式
- year 必须是字符串格式的4位年份
- 只输出 JSON，不要有其他解释"""

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": "你是一个学术文献分析专家，擅长识别各种引用格式并提取元数据。请严格按照要求的 JSON 格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM 原始响应: {result_text}")
            
            # 尝试提取 JSON
            # 处理可能被 markdown 代码块包裹的情况
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            else:
                # 尝试直接找 JSON 对象
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(0)
            
            result = json.loads(result_text)
            logger.info(f"LLM 提取成功: title={result.get('title', '')[:50]}...")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM 响应 JSON 解析失败: {e}")
            return {}
        except Exception as e:
            logger.warning(f"LLM 提取失败: {e}")
            return {}

    def _extract_references_via_llm_full(
            self,
            text: str,
            listing_style: ListingStyle,
    ) -> List[ReferenceEntry]:
        """
        当 listing_style 为 other / 无法规则解析时：
        将整段参考文献文本交给 LLM，解析并返回 ReferenceEntry 列表。

        - NUMBERED：返回带 number 的 ReferenceEntry
        - AUTHOR_YEAR：number 恒为 None
        """
        if not self._llm_client:
            logger.warning("格式为 other 需要 LLM，未配置 LLM 客户端，返回空列表")
            return []

        # =========================
        # Prompt 构造
        # =========================
        if listing_style == ListingStyle.NUMBERED:
            prompt = """你是一位学术文献解析专家。

    下方是一整段「参考文献」原文，包含多条文献，每条带有数字编号
    （如 [1]、1.、(2) 等）。

    你的目标是：识别该参考文献的引用格式（如 APA, MLA, IEEE, GB/T 7714, Chicago, Harvard, Vancouver 等），然后提取以下信息：
    1. title: 论文/文章/书籍的标题（完整标题，不要截断）
    2. authors: 作者列表（数组格式，每个作者一个元素）
    3. year: 发表年份（4位数字）

    请严格执行以下要求：

    1. 按原文顺序逐条识别参考文献。
    2. 每一条输出一个 JSON 对象。
    3. 每个对象必须包含以下字段（字段名必须完全一致）：

    - number: 整数，对应原文中的文献编号,若缺失则根据上下文进行推测
    - full_content: 该条参考文献的完整原文字符串
    - title: 文献标题字符串；无法识别则使用 null
    - authors: 作者数组，如 ["Smith J", "Doe A"]；无法识别则使用 []
    - year: 4 位年份字符串，如 "2020"；无法识别则使用 null


    输出要求：
    - 仅输出一个合法 JSON 数组
    - 不要输出解释、注释、markdown 标记
    输出示例：
    [
      {
        "number": 1,
        "full_content": "[1] 张三, 李四. 基于深度学习的图像识别方法研究[J]. 计算机学报, 2021, 44(3): 512-525.",
        "title": "基于深度学习的图像识别方法研究",
        "authors": ["张三", "李四"],
        "year": "2021"
      },
      {
        "number": 2,
        "full_content": "[2] Wang L, Chen H. A survey on large language models[J]. Artificial Intelligence Review, 2023, 56(4): 3015–3048.",
        "title": "A survey on large language models",
        "authors": ["Wang L", "Chen H"],
        "year": "2023"
      }
    ]

    参考文献原文：
    """
        else:
            prompt = """你是一位学术文献解析专家。

    下方是一整段作者-年份型「参考文献」原文（无数字编号）。

    你的目标是：识别该参考文献的引用格式（如 APA, MLA, IEEE, GB/T 7714, Chicago, Harvard, Vancouver 等），然后提取以下信息：
    1. title: 论文/文章/书籍的标题（完整标题，不要截断）
    2. authors: 作者列表（数组格式，每个作者一个元素）
    3. year: 发表年份（4位数字）

    请严格执行以下要求：

    1. 按原文顺序逐条识别参考文献。
    2. 每一条输出一个 JSON 对象。
    3. 每个对象必须包含以下字段（不要包含 number）：

    - full_content: 该条参考文献的完整原文字符串
    - title: 文献标题字符串；无法识别则使用 null
    - authors: 作者数组，如 ["Smith J", "Doe A"]；无法识别则使用 []
    - year: 4 位年份字符串；无法识别则使用 null


    输出要求：
    - 仅输出一个合法 JSON 数组
    - 不要输出解释、注释、markdown 标记
    输出示例：
    [
      {
        "full_content": "[1] 张三, 李四. 基于深度学习的图像识别方法研究[J]. 计算机学报, 2021, 44(3): 512-525.",
        "title": "基于深度学习的图像识别方法研究",
        "authors": ["张三", "李四"],
        "year": "2021"
      },
      {
        "full_content": "[2] Wang L, Chen H. A survey on large language models[J]. Artificial Intelligence Review, 2023, 56(4): 3015–3048.",
        "title": "A survey on large language models",
        "authors": ["Wang L", "Chen H"],
        "year": "2023"
      }
    ]

    参考文献原文：
    """

        full_prompt = prompt + "\n\n" + text.strip()

        try:
            # =========================
            # LLM 调用
            # =========================
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个学术文献分析专家，擅长识别各种引用格式并提取元数据。请严格按照要求的 JSON 格式输出。"
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.1,
                max_tokens=8192,
                timeout=300
            )

            result_text = response.choices[0].message.content.strip()
            logger.debug(f"LLM 全文参考文献原始响应长度: {len(result_text)}")

            # =========================
            # JSON 数组提取（鲁棒）
            # =========================
            arr_match = re.search(r'\[\s*\{', result_text)
            if arr_match:
                start = arr_match.start()
                depth = 0
                for i in range(start, len(result_text)):
                    if result_text[i] == '[':
                        depth += 1
                    elif result_text[i] == ']':
                        depth -= 1
                        if depth == 0:
                            result_text = result_text[start: i + 1]
                            break
                else:
                    result_text = result_text[start:]
            else:
                fenced_match = re.search(
                    r'```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```',
                    result_text,
                    re.DOTALL,
                )
                if fenced_match:
                    result_text = fenced_match.group(1).strip()
            #解析模型返回
            decoder = json.JSONDecoder()

            try:
                data, _ = decoder.raw_decode(result_text)
            except json.JSONDecodeError:
                logger.warning("LLM JSON raw_decode 失败")
                return []
            if not isinstance(data, list):
                logger.warning("LLM 返回的不是 JSON 数组")
                return []

            # =========================
            # ReferenceEntry 构造
            # =========================
            entries: List[ReferenceEntry] = []

            for item in data:
                if not isinstance(item, dict):
                    continue

                full_content = str(item.get("full_content") or "").strip()

                title = item.get("title")
                title = str(title).strip() if title else None

                # authors：统一归一为 List[str]
                authors_raw = item.get("authors", [])
                if isinstance(authors_raw, list):
                    authors = [str(a).strip() for a in authors_raw if str(a).strip()]
                elif isinstance(authors_raw, str):
                    authors = [
                        a.strip()
                        for a in authors_raw.replace(";", ",").split(",")
                        if a.strip()
                    ]
                else:
                    authors = []

                # year：Optional[str]
                year = item.get("year")
                year = str(year).strip() if year and str(year).strip() else None

                # number：仅 NUMBERED
                number: Optional[int] = None
                if listing_style == ListingStyle.NUMBERED:
                    try:
                        number = int(item.get("number"))
                    except (TypeError, ValueError):
                        number = None

                # title_normalized：派生字段
                title_normalized = self._normalize_title(title) if title else None

                entries.append(
                    ReferenceEntry(
                        full_content=full_content,
                        title=title,
                        title_normalized=title_normalized,
                        authors=authors,
                        year=year,
                        number=number,
                    )
                )

            logger.info(f"LLM 全文提取参考文献条数: {len(entries)}")
            return entries

        except json.JSONDecodeError as e:
            logger.warning(f"LLM 全文参考文献 JSON 解析失败: {e}")
            return []
        except Exception as e:
            logger.warning(f"LLM 全文参考文献提取失败: {e}")
            return []

    def extract(
        self,
        text: str,
        listing_style: ListingStyle,
        citation_format: CitationFormat
    ) -> List[ReferenceEntry]:
        """
        抽取参考文献列表
        
        Args:
            text: 参考文献文本（md或纯文本）
            listing_style: 条目列举方式
            citation_format: 引用格式
            
        Returns:
            参考文献条目列表
        """
        # 1. 预处理文本
        text = self._preprocess_text(text)
        
        # 2. 格式为 other 时：全文交给 LLM 提取，不按规则拆分
        if citation_format == CitationFormat.OTHER:
            return self._extract_references_via_llm_full(text, listing_style)

        
        # 3. 根据列举方式拆分条目
        if listing_style == ListingStyle.NUMBERED:
            entries = self._split_numbered_entries(text)
        else:
            entries = self._split_author_year_entries(text, citation_format)
        
        # 4. 根据引用格式提取元数据
        results = []
        for number, content in entries:
            entry = self._extract_metadata(content, citation_format)
            if listing_style == ListingStyle.NUMBERED:
                entry.number = number
            results.append(entry)
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 规范化换行
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # ---- 统一 Unicode 引号为 ASCII 引号 ----
        # 双引号变体（PDF / Word 常用）
        for ch in '\u201c\u201d\u201e\u201f\u00ab\u00bb\uff02':
            text = text.replace(ch, '"')
        # 单引号变体
        for ch in '\u2018\u2019\u201a\u201b\uff07':
            text = text.replace(ch, "'")
        
        # ---- 处理 PDF 换行连字符 ----
        # 当一行以 "字母-" 结尾且下一行以小写字母开头时，合并并去除连字符
        # 例如: "se-\ncure" → "secure"，但 "large-\nscale" → "large-scale"（保留复合词连字符）
        # 策略：先移除换行，后续在拆分条目时再精细处理
        text = re.sub(r'(\w)-\n\s*([a-z])', r'\1\2', text)
        
        # 去除每行前后空白后再处理
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # 移除 markdown 标题（现在每行都已经 strip 过了）
        text = re.sub(r'^#+\s*.*$', '', text, flags=re.MULTILINE)
        # 再次分割并过滤
        lines = [line.strip() for line in text.split('\n')]
        # 过滤掉常见的节标题（如 "References", "参考文献" 等）
        filtered_lines = []
        for line in lines:
            # 跳过常见的参考文献节标题
            if re.match(r'^(References|Bibliography|参考文献|引用文献|文献|Works Cited)$', line, re.IGNORECASE):
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    
    def _split_numbered_entries(self, text: str) -> List[Tuple[Optional[int], str]]:
        """
        拆分数字标号型条目
        
        Returns:
            [(编号, 内容), ...]
        """
        entries = []
        lines = text.split('\n')
        
        current_number = None
        current_content = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # 尝试匹配数字标号
            matched = False
            for regex in self._numbered_regexes:
                match = regex.match(line)
                if match:
                    # 保存之前的条目
                    if current_content:
                        content = ' '.join(current_content).strip()
                        entries.append((current_number, content))
                    
                    # 开始新条目
                    current_number = int(match.group(1))
                    remaining = line[match.end():].strip()
                    current_content = [remaining] if remaining else []
                    matched = True
                    break
            
            if not matched and current_content is not None:
                # 续行
                current_content.append(line.strip())
        
        # 保存最后一个条目
        if current_content:
            content = ' '.join(current_content).strip()
            entries.append((current_number, content))
        
        # 过滤掉无效条目（内容太短或没有编号的非引用文本）
        valid_entries = []
        for number, content in entries:
            # 跳过空内容
            if not content or len(content.strip()) < 10:
                continue
            # 跳过看起来像标题的内容（没有编号且很短）
            if number is None and len(content) < 30:
                continue
            valid_entries.append((number, content))
        
        return valid_entries
    
    def _split_author_year_entries(
        self,
        text: str,
        citation_format: CitationFormat
    ) -> List[Tuple[Optional[int], str]]:
        """
        拆分作者年份型条目（无数字标号）
        
        Returns:
            [(None, 内容), ...]
        """
        entries = []
        lines = text.split('\n')
        
        # 根据不同格式使用不同的分隔策略
        if citation_format in [CitationFormat.APA, CitationFormat.HARVARD]:
            # APA/Harvard: 通常以作者姓氏开头，后面紧跟逗号和名字首字母
            # 每个条目通常独立一段或以空行分隔
            entries = self._split_by_paragraph_or_author(lines)
        elif citation_format == CitationFormat.MLA:
            # MLA: 以作者姓氏开头
            entries = self._split_by_paragraph_or_author(lines)
        elif citation_format == CitationFormat.CHICAGO:
            # Chicago: 类似 APA
            entries = self._split_by_paragraph_or_author(lines)
        elif citation_format == CitationFormat.VANCOUVER:
            # Vancouver: 通常有数字，但也支持无数字
            entries = self._split_by_paragraph_or_author(lines)
        elif citation_format == CitationFormat.GB_T_7714:
            # GB/T 7714: 中文格式，通常以作者开头
            entries = self._split_by_paragraph_or_author(lines)
        else:
            # 默认按段落分隔
            entries = self._split_by_paragraph_or_author(lines)
        
        return [(None, e) for e in entries if e.strip()]
    
    def _split_by_paragraph_or_author(self, lines: List[str]) -> List[str]:
        """按段落或作者模式分隔（增强版）"""
        entries = []
        current_entry = []
        
        def is_author_start(line: str) -> bool:
            """检查是否是作者开头（使用增强模式）"""
            for regex in self._author_start_regexes:
                if regex.match(line):
                    return True
            return False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                # 空行，保存当前条目
                if current_entry:
                    entries.append(' '.join(current_entry))
                    current_entry = []
                continue
            
            # 检查是否是新条目的开始
            if current_entry and is_author_start(line):
                # 可能是新条目，保存当前条目
                entries.append(' '.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        # 保存最后一个条目
        if current_entry:
            entries.append(' '.join(current_entry))
        
        return entries
    
    def _extract_metadata(
        self,
        content: str,
        citation_format: CitationFormat
    ) -> ReferenceEntry:
        """
        根据引用格式提取元数据
        
        Args:
            content: 参考文献条目内容
            citation_format: 引用格式
            
        Returns:
            ReferenceEntry 对象
        """
        entry = ReferenceEntry(full_content=content)
        
        # 提取年份
        entry.year = self._extract_year(content)
        
        # 根据格式提取作者和标题
        if citation_format == CitationFormat.APA:
            entry.authors, entry.title = self._extract_apa(content)
        elif citation_format == CitationFormat.MLA:
            entry.authors, entry.title = self._extract_mla(content)
        elif citation_format == CitationFormat.IEEE:
            entry.authors, entry.title = self._extract_ieee(content)
        elif citation_format == CitationFormat.GB_T_7714:
            entry.authors, entry.title = self._extract_gbt7714(content)
        elif citation_format == CitationFormat.CHICAGO:
            entry.authors, entry.title = self._extract_chicago(content)
        elif citation_format == CitationFormat.HARVARD:
            entry.authors, entry.title = self._extract_harvard(content)
        elif citation_format == CitationFormat.VANCOUVER:
            entry.authors, entry.title = self._extract_vancouver(content)
        else:
            entry.authors, entry.title = self._extract_generic(content)
        
        # 如果规则提取失败（标题为空），尝试使用 LLM 提取
        if not entry.title and self._use_llm_fallback and self._llm_client:
            logger.info(f"规则提取标题失败，尝试 LLM 提取: {content[:80]}...")
            llm_result = self._extract_via_llm(content)
            
            if llm_result:
                # 使用 LLM 提取的结果
                if llm_result.get("title"):
                    entry.title = llm_result["title"]
                    logger.info(f"LLM 提取标题成功: {entry.title[:50]}...")
                
                if llm_result.get("authors") and not entry.authors:
                    authors = llm_result["authors"]
                    if isinstance(authors, list):
                        entry.authors = authors
                    elif isinstance(authors, str):
                        entry.authors = [a.strip() for a in authors.split(",")]
                
                if llm_result.get("year") and not entry.year:
                    entry.year = str(llm_result["year"])
        
        # 标准化标题
        if entry.title:
            entry.title_normalized = self._normalize_title(entry.title)
        
        return entry
    
    def _extract_year(self, content: str) -> Optional[str]:
        """提取年份"""
        match = self._year_regex.search(content)
        return match.group(1) if match else None
    
    def _extract_apa(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 APA 格式
        格式: Author, A. A., & Author, B. B. (Year). Title. Journal...
        """
        authors = []
        title = None
        
        # 提取作者: 年份前的部分
        year_match = re.search(r'\((\d{4})\)', content)
        if year_match:
            author_part = content[:year_match.start()].strip().rstrip(',').rstrip('.')
            authors = self._parse_authors(author_part)
            
            # 提取标题: 年份后到句号或期刊名（斜体）
            remaining = content[year_match.end():].strip().lstrip('.')
            # 标题通常到第一个句号，或者斜体标记前
            title_match = re.match(r'\s*([^.]+?)(?:\.|(?=\s*[A-Z][a-z]+,?\s*\d))', remaining)
            if title_match:
                title = title_match.group(1).strip()
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_mla(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 MLA 格式
        格式: Author. "Title." Journal, vol. X, no. X, Year, pp. X-X.
        """
        authors = []
        title = None
        
        # MLA 标题通常在引号内
        title_match = re.search(r'"([^"]+)"', content)
        if title_match:
            title = title_match.group(1)
            # 作者在引号前
            author_part = content[:title_match.start()].strip().rstrip('.')
            authors = self._parse_authors(author_part)
        else:
            # 尝试提取斜体标题 *Title* 或 _Title_
            title_match = re.search(r'[*_]([^*_]+)[*_]', content)
            if title_match:
                title = title_match.group(1)
                author_part = content[:title_match.start()].strip().rstrip('.')
                authors = self._parse_authors(author_part)
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_ieee(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 IEEE 格式
        格式: A. Author, B. Author, "Title," Journal, vol. X, pp. X-X, Year.
        或: A. Author et al., "Title," Journal...
        或: A. Author et al. Title. Journal...（无引号变体）
        """
        authors = []
        title = None
        
        # ---- 方式1: 标题在引号内 ----
        # 预处理中已统一 Unicode 引号为 ASCII，这里用 ASCII " 匹配即可
        title_match = re.search(r'"([^"]+)"', content)
        if title_match:
            title = title_match.group(1).strip().rstrip(',').strip()
            # 作者在引号前
            author_part = content[:title_match.start()].strip().rstrip(',')
            authors = self._parse_authors_ieee(author_part)
            return authors, title
        
        # ---- 方式2: 有 "et al." 但无引号 ----
        et_al_match = re.search(r'(.*?et\s+al\.?)\s*[.,]?\s*(.+)', content, re.IGNORECASE)
        if et_al_match:
            author_part_raw = et_al_match.group(1).strip()
            remaining = et_al_match.group(2).strip()
            
            # 提取作者
            author_part = re.sub(r'^\[\d{1,3}\]\s*', '', author_part_raw)
            if author_part:
                authors = self._parse_authors_ieee(author_part)
            
            # 从 remaining 提取标题
            title = self._extract_title_from_remaining_ieee(remaining)
            
            if title:
                return authors, title
        
        # ---- 方式3: 有 "and" 连接作者但无 et al. ----
        # 格式: A. Author and B. Author, "Title," ...
        # 先找 "and" 后面的逗号/句号分界点
        and_match = re.search(
            r'(?:^|\s)([A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Za-z\-]+)\s+and\s+([A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Za-z\-]+)',
            content,
        )
        if and_match:
            # 找到作者部分结束位置（and 后的作者姓名后）
            after_authors_pos = and_match.end()
            after_part = content[after_authors_pos:].strip()
            # 跳过逗号/句号
            after_part = re.sub(r'^[,.\s]+', '', after_part)
            
            if after_part:
                title = self._extract_title_from_remaining_ieee(after_part)
                if title:
                    author_raw = content[:after_authors_pos].strip()
                    author_raw = re.sub(r'^\[\d{1,3}\]\s*', '', author_raw)
                    authors = self._parse_authors_ieee(author_raw)
                    return authors, title
        
        # ---- 方式4: 最终回退 ----
        title = self._extract_title_fallback_ieee(content)
        
        return authors, title
    
    def _extract_title_from_remaining_ieee(self, remaining: str) -> Optional[str]:
        """
        从作者部分之后的剩余文本中提取 IEEE 标题
        
        处理两种情况：
        1. 引号包裹: "Title," Journal...
        2. 无引号: Title. Journal...
        """
        if not remaining:
            return None
        
        # 先检查引号
        quote_match = re.search(r'"([^"]+)"', remaining)
        if quote_match:
            return quote_match.group(1).strip().rstrip(',').strip()
        
        # 无引号：按句号分割，但要避免在缩写处断开（如 J. Sel. Areas Commun.）
        # IEEE 期刊/会议名通常包含大写缩写词（如 IEEE, Proc., Trans., Commun.）
        # 标题通常是第一个"较长"的句子
        
        # 策略：找到标题终止标记
        # 终止标记：vol., pp., no., Proc., IEEE, Trans., DOI, arXiv, URL, In:, 年份等
        end_match = re.search(
            r',\s*(?:vol\b|pp\b|no\b|Proc\b|IEEE\b|Trans\b|ACM\b|Springer|Wiley|Elsevier)',
            remaining,
            re.IGNORECASE,
        )
        if end_match:
            title = remaining[:end_match.start()].strip().rstrip('.,').strip()
            if len(title) > 10:
                return title
        
        # 退而求其次：取到第一个句号但忽略短缩写
        # 按 ". " 分割（句号后有空格）
        parts = re.split(r'\.\s+', remaining)
        if parts and len(parts[0].strip()) > 10:
            return parts[0].strip().rstrip('.,').strip()
        
        return None
    
    def _extract_gbt7714(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 GB/T 7714 格式（中国国标）
        格式: 作者. 题名[文献类型标志]. 刊名, 年, 卷(期): 页码.
        或: 作者. 题名[文献类型标志]. 出版地: 出版者, 年.
        """
        authors = []
        title = None
        
        # GB/T 7714 标题后通常有文献类型标志 [J], [M], [C], [D], [R], [S], [P], [DB/OL] 等
        title_match = re.search(r'[\.\u3002]\s*([^.\[\]]+)\s*\[([JMCDRSPN]|DB/OL|EB/OL)\]', content)
        if title_match:
            title = title_match.group(1).strip()
            # 作者在标题前
            author_part = content[:title_match.start()].strip()
            # 移除可能的编号（限制 1-3 位）
            author_part = re.sub(r'^\[\d{1,3}\]\s*', '', author_part)
            authors = self._parse_authors_chinese(author_part)
        else:
            # 尝试匹配中文句号分隔
            parts = re.split(r'[.\u3002]', content, maxsplit=2)
            if len(parts) >= 2:
                author_part = re.sub(r'^\[\d{1,3}\]\s*', '', parts[0].strip())
                authors = self._parse_authors_chinese(author_part)
                title = parts[1].strip()
                # 移除文献类型标志
                title = re.sub(r'\s*\[[^\]]+\]\s*$', '', title)
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_chicago(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 Chicago 格式
        格式 (Author-Date): Author, First. Year. "Title." Journal Volume (Issue): Pages.
        格式 (Notes-Bibliography): Author, First. "Title." Journal Volume, no. Issue (Year): Pages.
        """
        authors = []
        title = None
        
        # 尝试引号内标题
        title_match = re.search(r'"([^"]+)"', content)
        if title_match:
            title = title_match.group(1)
            # 作者部分
            author_part = content[:title_match.start()]
            # 移除年份
            author_part = re.sub(r'\s*\d{4}\.\s*$', '', author_part).strip().rstrip('.')
            authors = self._parse_authors(author_part)
        else:
            # 尝试斜体标题
            title_match = re.search(r'[*_]([^*_]+)[*_]', content)
            if title_match:
                title = title_match.group(1)
                author_part = content[:title_match.start()].strip()
                author_part = re.sub(r'\s*\d{4}\.\s*$', '', author_part).strip().rstrip('.')
                authors = self._parse_authors(author_part)
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_harvard(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 Harvard 格式
        格式: Author, I. (Year) 'Title', Journal, Volume(Issue), pp. Pages.
        """
        authors = []
        title = None
        
        # Harvard 标题通常在单引号内
        title_match = re.search(r"'([^']+)'", content)
        if title_match:
            title = title_match.group(1)
            # 作者在年份前
            year_match = re.search(r'\((\d{4})\)', content)
            if year_match:
                author_part = content[:year_match.start()].strip().rstrip(',')
                authors = self._parse_authors(author_part)
        
        if not title:
            # 尝试双引号
            title_match = re.search(r'"([^"]+)"', content)
            if title_match:
                title = title_match.group(1)
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_vancouver(self, content: str) -> Tuple[List[str], Optional[str]]:
        """
        提取 Vancouver 格式
        格式: Author AA, Author BB, et al. Title. Journal. Year;Volume(Issue):Pages.
        """
        authors = []
        title = None
        
        # 预处理：合并连续句号（如 "et al.." -> "et al."）
        content_clean = re.sub(r'\.{2,}', '.', content)
        
        # 方式1：尝试识别 "et al" 标记来定位作者结束位置
        et_al_match = re.search(r'(.*?(?:et\s*al\.?|等))\s*[.,]\s*(.+)', content_clean, re.IGNORECASE)
        if et_al_match:
            author_part = et_al_match.group(1).strip()
            remaining = et_al_match.group(2).strip()
            
            # 移除编号
            author_part = re.sub(r'^\[\d{1,3}\]\s*', '', author_part)
            author_part = re.sub(r'^\d{1,3}\.\s*', '', author_part)
            authors = self._parse_authors_vancouver(author_part)
            
            # 剩余部分按句号分割：Title. Journal. Year;...
            remaining_parts = [p.strip() for p in remaining.split('.') if p.strip()]
            if remaining_parts:
                title = remaining_parts[0]
            
            if title:
                return authors, title
        
        # 方式2：按句号分割，过滤空部分
        parts = [p.strip() for p in content_clean.split('.') if p.strip()]
        
        if len(parts) >= 3:
            author_part = parts[0]
            # 移除可能的编号（限制 1-3 位）
            author_part = re.sub(r'^\[\d{1,3}\]\s*', '', author_part)
            author_part = re.sub(r'^\d{1,3}\.\s*', '', author_part)
            authors = self._parse_authors_vancouver(author_part)
            title = parts[1]
        elif len(parts) == 2:
            author_part = parts[0]
            author_part = re.sub(r'^\[\d{1,3}\]\s*', '', author_part)
            authors = self._parse_authors_vancouver(author_part)
            title = parts[1]
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_generic(self, content: str) -> Tuple[List[str], Optional[str]]:
        """通用提取方法"""
        # 尝试多种模式提取标题
        title = None
        authors = []
        
        # 尝试引号
        title_match = re.search(r'["\']([^"\']+)["\']', content)
        if title_match:
            title = title_match.group(1)
        
        if not title:
            title = self._extract_title_fallback(content)
        
        return authors, title
    
    def _extract_title_fallback_ieee(self, content: str) -> Optional[str]:
        """
        IEEE 格式专用的回退标题提取方法
        处理无引号无 et al. 的 IEEE 格式，如：A. Author and B. Author. Title. Journal...
        """
        # 移除编号
        content = re.sub(r'^\[\d{1,3}\]\s*', '', content)
        
        # 1. 尝试引号（含 Unicode 引号，作为最后保障）
        quote_match = re.search(r'["""\u201c\u201d\']([^"""\u201c\u201d\']+)["""\u201c\u201d\']', content)
        if quote_match:
            return quote_match.group(1).strip().rstrip(',').strip()
        
        # 2. 策略：跳过 IEEE 作者名部分，找到标题起始位置
        # IEEE 作者名格式: "A. Author" "A. B. Author" "A.-B. Author"
        # 关键特征: 作者名中的句号后面紧跟着 1-2 个大写字母或空格+大写字母
        # 标题开始: 句号/逗号后 + 多个小写字母组成的单词（至少 3 个字母）
        
        # 找到最后一个作者名的结束位置
        # 作者名模式: "X. Lastname" 或 "X. Y. Lastname"
        # 跳过所有 "X. " 或 "X. Y. " 模式
        
        # 更可靠的方法: 找到 ", " 或 ". " 后面跟着的第一个至少 3 个小写字母的单词
        # 且该单词不是常见的作者前缀
        title_start_match = re.search(
            r'[.,]\s+([A-Z][a-z]{2,}(?:\s+[A-Za-z]|\s*[:\-–—])[^,]*)',
            content
        )
        if title_start_match:
            potential_title = title_start_match.group(1).strip()
            
            # 截取到期刊/会议终止标记
            end_match = re.search(
                r',\s*(?:vol\b|pp\b|no\b|Proc\b|IEEE\b|Trans\b|ACM\b|Springer|Wiley|Elsevier|in\s+Proc)',
                potential_title,
                re.IGNORECASE
            )
            if end_match:
                potential_title = potential_title[:end_match.start()].strip().rstrip('.,')
            else:
                # 按 ". " 分割取第一段
                dot_parts = re.split(r'\.\s+', potential_title)
                if dot_parts:
                    potential_title = dot_parts[0].strip().rstrip('.,')
            
            if len(potential_title) > 10:
                return potential_title
        
        return None
    
    def _extract_title_fallback(self, content: str) -> Optional[str]:
        """
        通用回退标题提取方法
        尝试从内容中智能提取标题
        """
        # 移除编号
        content = re.sub(r'^\[\d{1,3}\]\s*', '', content)
        content = re.sub(r'^\d{1,3}\.\s*', '', content)
        
        # 尝试找到引号中的内容
        quote_match = re.search(r'["""\']([^"""\']+)["""\']', content)
        if quote_match:
            return quote_match.group(1).strip()
        
        # 改进的句号分割：避免在作者名缩写处分割（如 "A." "M.S."）
        # 只在 ". " + 大写字母开头的单词（非单字母）处分割
        # 匹配：句号 + 空格 + 大写字母开头且至少2个字母的单词
        title_start_match = re.search(
            r'\.\s+([A-Z][a-z]{2,}[^.]*)',
            content
        )
        if title_start_match:
            potential_title = title_start_match.group(1).strip()
            # 清理文献类型标志
            potential_title = re.sub(r'\s*\[[^\]]+\]', '', potential_title)
            # 截取到句号
            dot_pos = potential_title.find('.')
            if dot_pos > 10:
                potential_title = potential_title[:dot_pos]
            if len(potential_title) > 5:
                return potential_title
        
        # 最后尝试：简单句号分割（针对格式规范的情况）
        parts = re.split(r'(?<=[a-z])\.\s+(?=[A-Z])', content, maxsplit=1)
        if len(parts) >= 2:
            potential_title = parts[1].strip()
            # 清理
            potential_title = re.sub(r'\s*\[[^\]]+\]', '', potential_title)
            dot_pos = potential_title.find('.')
            if dot_pos > 10:
                potential_title = potential_title[:dot_pos]
            if len(potential_title) > 5:
                return potential_title
        
        return None
    
    def _parse_authors(self, author_str: str) -> List[str]:
        """解析西文作者字符串"""
        if not author_str:
            return []
        
        authors = []
        # 按 , and 或 & 分隔
        # 先替换 & 和 and 为统一分隔符
        author_str = re.sub(r'\s*&\s*', ', ', author_str)
        author_str = re.sub(r',\s*and\s+', ', ', author_str)
        author_str = re.sub(r'\s+and\s+', ', ', author_str)
        
        # 按逗号分隔，但要注意 "Last, First" 格式
        parts = author_str.split(',')
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue
            
            # 检查下一部分是否是名字首字母（如 "A." 或 "A. B."）
            if i + 1 < len(parts):
                next_part = parts[i + 1].strip()
                # 如果下一部分看起来像名字首字母
                if re.match(r'^[A-Z]\.?\s*[A-Z]?\.?\s*$', next_part):
                    authors.append(f"{part}, {next_part}")
                    i += 2
                    continue
            
            # 单独的名字
            if part and not re.match(r'^[A-Z]\.?\s*[A-Z]?\.?\s*$', part):
                authors.append(part)
            i += 1
        
        return authors
    
    def _parse_authors_ieee(self, author_str: str) -> List[str]:
        """解析 IEEE 格式作者（名在前）"""
        if not author_str:
            return []
        
        # IEEE: A. B. Author, C. D. Author, and E. F. Author
        author_str = re.sub(r',?\s*and\s+', ', ', author_str)
        author_str = re.sub(r'\s*&\s*', ', ', author_str)
        
        authors = []
        # 匹配 "A. B. LastName" 模式
        pattern = r'([A-Z]\.\s*)+[A-Za-z\-]+'
        matches = re.findall(pattern, author_str)
        
        # 简单按逗号分隔
        for part in author_str.split(','):
            part = part.strip()
            if part and re.search(r'[A-Za-z]', part):
                authors.append(part)
        
        return authors
    
    def _parse_authors_chinese(self, author_str: str) -> List[str]:
        """解析中文作者字符串"""
        if not author_str:
            return []
        
        # 中文作者通常用逗号、顿号分隔
        author_str = author_str.replace('、', ',').replace('，', ',')
        authors = [a.strip() for a in author_str.split(',') if a.strip()]
        return authors
    
    def _parse_authors_vancouver(self, author_str: str) -> List[str]:
        """解析 Vancouver 格式作者"""
        if not author_str:
            return []
        
        # Vancouver: LastName AB, LastName CD
        authors = [a.strip() for a in author_str.split(',') if a.strip()]
        return authors
    
    def _normalize_title(self, title: str) -> str:
        """
        标准化标题
        - 只去除首尾的标点符号
        - 保留中间的标点（如连字符）
        - 保留原始大小写
        """
        if not title:
            return ""
        
        # 定义首尾需要去除的标点符号
        punctuation_to_strip = r'[\s\.,;:!?\'\"，。；：！？、\[\]\(\)（）\{\}【】""\'\'`]'
        
        # 去除首部标点和空白
        normalized = re.sub(f'^{punctuation_to_strip}+', '', title)
        # 去除尾部标点和空白
        normalized = re.sub(f'{punctuation_to_strip}+$', '', normalized)
        
        # 合并多个空格为单个空格
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized


def extract_references(
    text: str,
    listing_style: str | ListingStyle,
    citation_format: str | CitationFormat,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    use_llm_fallback: bool = True,
) -> List[list]:
    """
    便捷函数：抽取参考文献列表
    
    Args:
        text: 参考文献文本
        listing_style: 条目列举方式 ("numbered" 或 "author_year")
        citation_format: 引用格式 ("apa", "mla", "ieee", "gb_t_7714", "chicago", "harvard", "vancouver", "other")
        llm_model: LLM 模型名称（规则提取失败时使用）
        llm_api_key: LLM API 密钥
        llm_base_url: LLM API 基础 URL
        use_llm_fallback: 规则提取失败时是否使用 LLM 备选提取
        
    Returns:
        参考文献列表，每个元素为:
        - 有数字标号型: [编号, 全内容, 题目, 作者, 年份]
        - 无数字标号型: [全内容, 题目, 作者, 年份]
    """
    # 转换字符串参数
    if isinstance(listing_style, str):
        listing_style = ListingStyle(listing_style)
    if isinstance(citation_format, str):
        citation_format = CitationFormat(citation_format)
    
    extractor = ReferenceExtractor(
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        use_llm_fallback=use_llm_fallback,
    )
    entries = extractor.extract(text, listing_style, citation_format)
    
    return [entry.to_list() for entry in entries]

# extracter=ReferenceExtractor(llm_model="ep-20250501131031-76kqz",llm_base_url="https://ark.cn-beijing.volces.com/api/v3",llm_api_key="30a70266-37d5-4210-b8a2-34d5fb629230",use_llm_fallback=True)
# text="""
# ARTEMYEVA（袁丽萌）M.基于共享经济的商业模式创新驱动因素研究[D].对外经济贸易大学, 2022.
# 陈军, 李庆瑞.数字经济下商业模式创新与盈余管理的探索[J].上海商业, 2023(01):29-31.
# 董成惠.共享经济:理论与现实[J].广东财经大学学报, 2016, 31(05):4-15.
# 樊丽丽.基于商业模式对ofo单车经营失败原因的分析[J].技术与市场, 2021, 28(01):187+189.
# 高岳, 王海智, 刘思茹.基于层次分析的共享单车用户满意度影响因素[J].科技和产业, 2023, 23(22):99-104.
# 葛文静.共享单车价值网模型的构建及盈利模式探讨——以摩拜和OFO共享单车为例[J].中国商论, 2017(15):174-176.
# 关钰桥, 孟韬.分享经济背景下企业商业模式比较分析——以美国Uber与中国滴滴为例[J].企业经济, 2018(04):27-35.
# 关钰桥, 孟韬.数字时代共享经济商业模式合法性获取机制研究——以滴滴出行、哈啰出行和闲鱼为例[J].财经问题研究, 2022(05):27-37.
# 郭晓琴. 分享经济时代知识付费平台的商业模式优化研究[D].苏州大学, 2019.
# 国家市场监督管理总局.共享经济指导原则与基础框架:GB/T41836-2022[S].2022.
# 纪淑平, 李振国.国外共享单车发展对我国的经验借鉴与启示[J].对外经贸实务, 2018(04):36-39.
# 李鸿磊.基于价值创造视角的商业模式分类研究——以三个典型企业的分类应用为例[J].管理评论, 2018, 30(04):257-272.
# 李牧南, 黄槿.我国当前共享经济发展障碍与相关政策启示[J].科技管理研究, 2020, 40(8):237-242.
# 李四海, 傅瑜佳.未来共享经济的主战场:从资本竞赛到创新竞赛[J].清华管理评论, 2018, 61(5):64-69.
#
# """
# result=extracter._extract_references_via_llm_full(text,ListingStyle.AUTHOR_YEAR)
# print(result)
