# -*- coding: utf-8 -*-
"""
引用文本定位与提取器

从按 title 聚合后的 Markdown 文本中定位和提取引用信息。

支持的引用格式：
1. 数字型 (numeric)：[1], [2-5], [2,3,5], [1,3-5,8]
2. 作者年份型 (author_year)：Smith (2020), Smith et al. (2019), (Smith, 2020)
3. 中文作者年份型 (chinese)：张三（2021）, 张三等（2021）, （张三，2019）

输出格式：
- 数字型: [[position, number, citation_anchor, context], ...]
- 作者年份型: [[position, authors, year, citation_anchor, context], ...]

position = (title, sentence_id, anchor_span)
- title: 段落标题
- sentence_id: 句子在段落中的索引（0-based）
- anchor_span: 引用在句子中的字符区间 [start, end)
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class CitationFormat(Enum):
    """引用格式类型"""
    NUMERIC = "numeric"
    AUTHOR_YEAR = "author_year"


@dataclass
class CitationPosition:
    """引用位置标识"""
    title: str              # 段落标题
    sentence_id: int        # 句子在段落中的索引（0-based）
    anchor_span: Tuple[int, int]  # 引用在句子中的字符区间 [start, end)
    
    def to_tuple(self) -> Tuple:
        return (self.title, self.sentence_id, self.anchor_span)
    
    def __hash__(self):
        return hash((self.title, self.sentence_id, self.anchor_span))
    
    def __eq__(self, other):
        if not isinstance(other, CitationPosition):
            return False
        return (self.title, self.sentence_id, self.anchor_span) == \
               (other.title, other.sentence_id, other.anchor_span)


@dataclass
class NumericCitation:
    """数字型引用结果"""
    position: CitationPosition     # 引用位置
    number: int                    # 数字标号
    citation_anchor: str           # 定位引用的句子
    context: str                   # 上下文（上两句 + 当前句 + 下两句）
    citation_type: str = ""        # 引用类型：single/range/list/mixed
    raw_citation: str = ""         # 原始引用文本，如 [1,3-5]
    
    def to_list(self) -> List:
        """转换为列表格式 [position, 数字标号, citation_anchor, context]"""
        return [self.position.to_tuple(), self.number, self.citation_anchor, self.context]


@dataclass
class AuthorYearCitation:
    """作者年份型引用结果"""
    position: CitationPosition     # 引用位置
    authors: str                   # 作者
    year: str                      # 年份（可能带后缀如 2020a）
    citation_anchor: str           # 定位引用的句子
    context: str                   # 上下文
    citation_type: str = ""        # 引用类型
    raw_citation: str = ""         # 原始引用文本
    
    def to_list(self) -> List:
        """转换为列表格式 [position, 作者, 年份, citation_anchor, context]"""
        return [self.position.to_tuple(), self.authors, self.year, self.citation_anchor, self.context]


class CitationExtractor:
    """
    引用文本提取器
    
    从 Markdown 文本中提取引用信息，包括引用标识、所在句子和上下文。
    """
    
    def __init__(self):
        """初始化提取器"""
        # 编译常用正则表达式
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译正则表达式"""
        # ================== 数字型引用模式 ==================
        
        # 单一引用: [1]
        self.pattern_numeric_single = re.compile(
            r'\[(\d{1,3})\]'
        )
        
        # 范围引用: [2-5] 或 [2–5]（注意 en-dash）
        self.pattern_numeric_range = re.compile(
            r'\[(\d{1,3})\s*[-–]\s*(\d{1,3})\]'
        )
        
        # 列表引用: [2,3,5]
        self.pattern_numeric_list = re.compile(
            r'\[(\d{1,3}(?:\s*,\s*\d{1,3})+)\]'
        )
        
        # 混合引用: [1,3-5,8] - 包含逗号和范围
        self.pattern_numeric_mixed = re.compile(
            r'\[(\d{1,3}(?:\s*[-–]\s*\d{1,3})?(?:\s*,\s*\d{1,3}(?:\s*[-–]\s*\d{1,3})?)+)\]'
        )
        
        # 通用数字引用（用于检测任何方括号数字引用）
        self.pattern_numeric_any = re.compile(
            r'\[[\d,\-–\s]+\]'
        )
        
        # ================== 作者年份型引用模式 ==================
        
        # 单作者: Smith (2020)
        self.pattern_author_single = re.compile(
            r'([A-Z][a-zA-Z\'\-]+)\s*\((\d{4}[a-z]?)\)'
        )
        
        # 多作者 (and/&): Smith & Brown (2021) 或 Smith and Brown (2021)
        self.pattern_author_multiple_and = re.compile(
            r'([A-Z][a-zA-Z\'\-]+)\s*(?:&|and)\s*([A-Z][a-zA-Z\'\-]+)\s*\((\d{4}[a-z]?)\)'
        )
        
        # et al.: Smith et al. (2019)
        self.pattern_author_et_al = re.compile(
            r'([A-Z][a-zA-Z\'\-]+)\s+et\s+al\.?\s*\((\d{4}[a-z]?)\)'
        )
        
        # 括号内作者: (Smith, 2020) 或 (Smith et al., 2019)
        self.pattern_author_parenthetical = re.compile(
            r'\(([A-Z][a-zA-Z\'\-]+(?:\s+et\s+al\.?)?)\s*,\s*(\d{4}[a-z]?)\)'
        )
        
        # 括号内多作者: (Smith & Brown, 2021)
        self.pattern_author_parenthetical_multi = re.compile(
            r'\(([A-Z][a-zA-Z\'\-]+\s*(?:&|and)\s*[A-Z][a-zA-Z\'\-]+)\s*,\s*(\d{4}[a-z]?)\)'
        )
        
        # 多引用（分号分隔）: (Smith, 2019; Brown, 2020)
        self.pattern_author_multiple_citations = re.compile(
            r'\(([^)]+;\s*[^)]+)\)'
        )
        
        # ================== 中文作者年份型引用模式 ==================
        
        # 中文单作者: 张三（2021）或 张三(2021)
        self.pattern_chinese_single = re.compile(
            r'([\u4e00-\u9fff]{2,4})\s*[（(](\d{4}[a-z]?)[）)]'
        )
        
        # 中文多作者: 李四、王五（2020）
        self.pattern_chinese_multiple = re.compile(
            r'([\u4e00-\u9fff]{2,4}(?:[、,][\u4e00-\u9fff]{2,4})+)\s*[（(](\d{4}[a-z]?)[）)]'
        )
        
        # 中文 et al.: 张三等（2021）
        self.pattern_chinese_et_al = re.compile(
            r'([\u4e00-\u9fff]{2,4})等\s*[（(](\d{4}[a-z]?)[）)]'
        )
        
        # 中文括号内作者: （张三，2019）
        self.pattern_chinese_parenthetical = re.compile(
            r'[（(]([\u4e00-\u9fff]{2,4}(?:等)?)\s*[,，]\s*(\d{4}[a-z]?)[）)]'
        )
        
        # 中文括号内多引用: （张三，2019；李四，2020）
        self.pattern_chinese_multiple_parenthetical = re.compile(
            r'[（(]([^）)]+[;；][^）)]+)[）)]'
        )
    
    # ================== 文本预处理 ==================
    
    def split_paragraphs(self, text: str) -> List[Tuple[str, str]]:
        """
        将文本按 Markdown 标题切分段落
        
        使用 # 标题行来划分段落，每个 # 开头的行开始一个新段落。
        
        Args:
            text: Markdown 文本
            
        Returns:
            [(段落标题, 段落内容), ...] 列表
        """
        result = []
        current_title = "Untitled"  # 默认标题（文档开头没有标题的情况）
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # 检查是否是标题行
            title_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if title_match:
                # 保存之前的段落（如果有内容）
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        result.append((current_title, content))
                
                # 开始新段落
                current_title = title_match.group(2).strip()
                current_content = []
            else:
                # 非标题行，添加到当前段落
                current_content.append(line)
        
        # 保存最后一个段落
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                result.append((current_title, content))
        
        return result
    
    def split_sentences(self, paragraph: str) -> List[str]:
        """
        将段落按句子切分
        
        支持：
        - 英文句号、问号、感叹号
        - 中文句号、问号、感叹号
        - 中文分号（；）
        
        Args:
            paragraph: 段落文本
            
        Returns:
            句子列表
        """
        # 处理常见缩写，避免错误切分
        text = paragraph
        # 保护常见缩写
        abbreviations = ['et al.', 'i.e.', 'e.g.', 'vs.', 'etc.', 'Fig.', 'Tab.', 'Eq.', 'Dr.', 'Mr.', 'Mrs.', 'Prof.']
        placeholders = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR{i}__"
            placeholders[placeholder] = abbr
            text = text.replace(abbr, placeholder)
        
        # 按句子结束符切分
        # 英文：句号、问号、感叹号（后面跟空格或换行）
        # 中文：句号、问号、感叹号、分号
        sentences = re.split(r'(?<=[.!?。！？；])\s*', text)
        
        # 恢复缩写
        result = []
        for s in sentences:
            for placeholder, abbr in placeholders.items():
                s = s.replace(placeholder, abbr)
            s = s.strip()
            if s:
                result.append(s)
        
        return result
    
    def get_context(self, sentences: List[str], index: int, window: int = 2) -> str:
        """
        获取指定句子的上下文
        
        Args:
            sentences: 句子列表
            index: 当前句子索引
            window: 上下文窗口大小（上下各多少句）
            
        Returns:
            上下文文本
        """
        start = max(0, index - window)
        end = min(len(sentences), index + window + 1)
        return ' '.join(sentences[start:end])
    
    # ================== 数字型引用提取 ==================
    
    def extract_numeric_single(self, sentence: str) -> List[Tuple[int, str, str, Tuple[int, int]]]:
        """
        提取单一数字引用 [1]
        
        注意：正则 \[(\d{1,3})\] 本身只匹配完整的单一引用，
        不会匹配 [1,2,3] 或 [1-3] 中的部分内容，因此无需额外过滤。
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(数字, 原始引用文本, 类型, span), ...]
            span: (start, end) 在句子中的字符位置
        """
        results = []
        for match in re.finditer(r'\[(\d{1,3})\]', sentence):
            full_match = match.group(0)
            span = match.span()
            num = int(match.group(1))
            results.append((num, full_match, 'single', span))
        
        return results
    
    def extract_numeric_range(self, sentence: str) -> List[Tuple[int, str, str, Tuple[int, int]]]:
        """
        提取范围数字引用 [2-5] 并展开为独立引用
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(数字, 原始引用文本, 类型, span), ...] 展开后的所有数字
        """
        results = []
        for match in self.pattern_numeric_range.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            start_num = int(match.group(1))
            end_num = int(match.group(2))
            
            # 展开范围，每个数字共享同一个 span
            for num in range(start_num, end_num + 1):
                results.append((num, full_match, 'range', span))
        
        return results
    
    def extract_numeric_list(self, sentence: str) -> List[Tuple[int, str, str, Tuple[int, int]]]:
        """
        提取列表数字引用 [2,3,5] 并拆分为独立引用
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(数字, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_numeric_list.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            nums_str = match.group(1)
            
            # 检查是否包含范围（如果包含，让 mixed 处理）
            if '-' in nums_str or '–' in nums_str:
                continue
            
            # 按逗号拆分
            nums = [int(n.strip()) for n in nums_str.split(',')]
            for num in nums:
                results.append((num, full_match, 'list', span))
        
        return results
    
    def extract_numeric_mixed(self, sentence: str) -> List[Tuple[int, str, str, Tuple[int, int]]]:
        """
        提取混合数字引用 [1,3-5,8] 并展开为独立引用
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(数字, 原始引用文本, 类型, span), ...]
        """
        results = []
        
        # 匹配任何方括号内的数字组合
        for match in re.finditer(r'\[([\d,\-–\s]+)\]', sentence):
            full_match = match.group(0)
            span = match.span()
            content = match.group(1)
            
            # 如果只是单个数字，跳过（让 single 处理）
            if re.match(r'^\d{1,3}$', content.strip()):
                continue
            
            # 如果只是范围（无逗号），跳过（让 range 处理）
            if ',' not in content and ('-' in content or '–' in content):
                continue
            
            # 如果不包含范围（只有逗号），跳过（让 list 处理）
            if '-' not in content and '–' not in content:
                continue
            
            # 混合类型：按逗号拆分，然后处理每个部分
            parts = content.split(',')
            for part in parts:
                part = part.strip()
                # 检查是否是范围
                range_match = re.match(r'(\d{1,3})\s*[-–]\s*(\d{1,3})', part)
                if range_match:
                    start_num = int(range_match.group(1))
                    end_num = int(range_match.group(2))
                    for num in range(start_num, end_num + 1):
                        results.append((num, full_match, 'mixed', span))
                else:
                    # 单个数字
                    try:
                        num = int(part)
                        results.append((num, full_match, 'mixed', span))
                    except ValueError:
                        continue
        
        return results
    
    def extract_all_numeric(self, sentence: str) -> List[Tuple[int, str, str, Tuple[int, int]]]:
        """
        提取句子中的所有数字型引用
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(数字, 原始引用文本, 类型, span), ...]
        """
        results = []
        # 使用 (span, number) 作为去重 key
        # 同一位置的同一文献不重复提取，但同一位置的不同文献可以
        seen = set()
        
        # 按优先级处理：mixed > range > list > single
        
        # 1. 混合引用
        for num, raw, type_, span in self.extract_numeric_mixed(sentence):
            key = (span, num)
            if key not in seen:
                results.append((num, raw, type_, span))
                seen.add(key)
        
        # 2. 范围引用
        for num, raw, type_, span in self.extract_numeric_range(sentence):
            key = (span, num)
            if key not in seen:
                results.append((num, raw, type_, span))
                seen.add(key)
        
        # 3. 列表引用
        for num, raw, type_, span in self.extract_numeric_list(sentence):
            key = (span, num)
            if key not in seen:
                results.append((num, raw, type_, span))
                seen.add(key)
        
        # 4. 单一引用
        for num, raw, type_, span in self.extract_numeric_single(sentence):
            key = (span, num)
            if key not in seen:
                results.append((num, raw, type_, span))
                seen.add(key)
        
        return results
    
    # ================== 作者年份型引用提取 ==================
    
    def extract_author_single(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取单作者引用 Smith (2020)
        
        注意：此函数不做复杂的过滤，由 extract_all_author_year 统一处理范围重叠问题。
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_author_single.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author = match.group(1)
            year = match.group(2)
            results.append((author, year, full_match, 'single_author', span))
        
        return results
    
    def extract_author_multiple_and(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取多作者引用 Smith & Brown (2021) 或 Smith and Brown (2021)
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_author_multiple_and.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author1 = match.group(1)
            author2 = match.group(2)
            year = match.group(3)
            
            authors = f"{author1} & {author2}"
            results.append((authors, year, full_match, 'multiple_authors_and', span))
        
        return results
    
    def extract_author_et_al(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取 et al. 引用 Smith et al. (2019)
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_author_et_al.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author = match.group(1)
            year = match.group(2)
            
            authors = f"{author} et al."
            results.append((authors, year, full_match, 'et_al', span))
        
        return results
    
    def extract_author_parenthetical(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取括号内作者引用 (Smith, 2020) 或 (Smith et al., 2019)
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_author_parenthetical.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author = match.group(1)
            year = match.group(2)
            
            results.append((author, year, full_match, 'parenthetical', span))
        
        return results
    
    def extract_author_parenthetical_multi(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取括号内多作者引用 (Smith & Brown, 2021)
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_author_parenthetical_multi.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            authors = match.group(1)
            year = match.group(2)
            
            results.append((authors, year, full_match, 'parenthetical_multi', span))
        
        return results
    
    def extract_author_year_suffix(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取年份后缀引用 Smith (2020a)
        
        注意：年份后缀在其他函数中已处理（\d{4}[a-z]?）
        """
        return []
    
    def extract_author_multiple_citations(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取多引用（分号分隔）(Smith, 2019; Brown, 2020)
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        
        for match in self.pattern_author_multiple_citations.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            content = match.group(1)
            
            # 按分号拆分
            parts = re.split(r'[;；]', content)
            
            for part in parts:
                part = part.strip()
                sub_match = re.match(r'([A-Za-z][a-zA-Z\'\-\s]+(?:et\s+al\.?)?)\s*,\s*(\d{4}[a-z]?)', part)
                if sub_match:
                    author = sub_match.group(1).strip()
                    year = sub_match.group(2)
                    results.append((author, year, full_match, 'multiple_citations', span))
        
        return results
    
    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """检查两个范围是否重叠"""
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])
    
    def _span_contained_in_any(self, span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
        """检查一个范围是否与任何已有范围重叠"""
        for existing_span in spans:
            if self._spans_overlap(span, existing_span):
                return True
        return False
    
    def extract_all_author_year(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取句子中的所有作者年份型引用
        
        使用范围重叠检查来避免重复提取：
        - 优先处理复杂模式（多引用、et al.、多作者等）
        - 如果简单模式的匹配范围与复杂模式重叠，则跳过
        - 同一个提取函数内的多个结果（如多引用分号分隔）都会被保留
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        matched_spans = []  # 记录已匹配的范围（用于跨函数去重）
        
        # 辅助函数：添加一批结果，返回新增的 spans
        def add_batch(extractions):
            new_spans = []
            for author, year, raw, type_, span in extractions:
                if not self._span_contained_in_any(span, matched_spans):
                    results.append((author, year, raw, type_, span))
                    if span not in new_spans:
                        new_spans.append(span)
            return new_spans
        
        # 按优先级处理（复杂模式优先）
        
        # 1. 多引用（分号分隔）- 最高优先级
        # 注意：同一个括号内的多个引用共享相同 span，都需要保留
        new_spans = add_batch(self.extract_author_multiple_citations(sentence))
        matched_spans.extend(new_spans)
        
        # 2. 括号内多作者
        new_spans = add_batch(self.extract_author_parenthetical_multi(sentence))
        matched_spans.extend(new_spans)
        
        # 3. 括号内作者
        new_spans = add_batch(self.extract_author_parenthetical(sentence))
        matched_spans.extend(new_spans)
        
        # 4. et al.
        new_spans = add_batch(self.extract_author_et_al(sentence))
        matched_spans.extend(new_spans)
        
        # 5. 多作者 and/&
        new_spans = add_batch(self.extract_author_multiple_and(sentence))
        matched_spans.extend(new_spans)
        
        # 6. 单作者 - 最低优先级
        new_spans = add_batch(self.extract_author_single(sentence))
        matched_spans.extend(new_spans)
        
        return results
    
    # ================== 中文作者年份型引用提取 ==================
    
    def extract_chinese_single(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取中文单作者引用 张三（2021）
        
        注意：此函数不做复杂的过滤，由 extract_all_chinese 统一处理范围重叠问题。
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        # 中文常见连接词（作为名字开头时需要去除）
        connectors = ('和', '与', '及', '或')
        
        results = []
        for match in self.pattern_chinese_single.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author = match.group(1)
            year = match.group(2)
            
            # 去除作者名开头的连接词
            for conn in connectors:
                if author.startswith(conn):
                    author = author[len(conn):]
                    break
            
            # 如果去除连接词后作者名为空或太短，跳过
            if len(author) < 2:
                continue
            
            results.append((author, year, full_match, 'chinese_single', span))
        
        return results
    
    def extract_chinese_multiple(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取中文多作者引用 李四、王五（2020）
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_chinese_multiple.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            authors = match.group(1)
            year = match.group(2)
            
            results.append((authors, year, full_match, 'chinese_multiple', span))
        
        return results
    
    def extract_chinese_et_al(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取中文"等"引用 张三等（2021）
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        # 中文常见连接词（作为名字开头时需要去除）
        connectors = ('和', '与', '及', '或')
        
        results = []
        for match in self.pattern_chinese_et_al.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author = match.group(1)
            year = match.group(2)
            
            # 去除作者名开头的连接词
            for conn in connectors:
                if author.startswith(conn):
                    author = author[len(conn):]
                    break
            
            authors = f"{author}等"
            results.append((authors, year, full_match, 'chinese_et_al', span))
        
        return results
    
    def extract_chinese_parenthetical(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取中文括号内作者引用 （张三，2019）
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        for match in self.pattern_chinese_parenthetical.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            author = match.group(1)
            year = match.group(2)
            
            results.append((author, year, full_match, 'chinese_parenthetical', span))
        
        return results
    
    def extract_chinese_multiple_parenthetical(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取中文括号内多引用 （张三，2019；李四，2020）
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        
        for match in self.pattern_chinese_multiple_parenthetical.finditer(sentence):
            full_match = match.group(0)
            span = match.span()
            content = match.group(1)
            
            # 按分号拆分
            parts = re.split(r'[;；]', content)
            
            for part in parts:
                part = part.strip()
                sub_match = re.match(r'([\u4e00-\u9fff]{2,4}(?:等)?)\s*[,，]\s*(\d{4}[a-z]?)', part)
                if sub_match:
                    author = sub_match.group(1)
                    year = sub_match.group(2)
                    results.append((author, year, full_match, 'chinese_multiple_parenthetical', span))
        
        return results
    
    def extract_all_chinese(self, sentence: str) -> List[Tuple[str, str, str, str, Tuple[int, int]]]:
        """
        提取句子中的所有中文作者年份型引用
        
        使用范围重叠检查来避免重复提取。
        同一个提取函数内的多个结果（如括号内多引用）都会被保留。
        
        Args:
            sentence: 句子文本
            
        Returns:
            [(作者, 年份, 原始引用文本, 类型, span), ...]
        """
        results = []
        matched_spans = []  # 记录已匹配的范围（用于跨函数去重）
        
        # 辅助函数：添加一批结果，返回新增的 spans
        def add_batch(extractions):
            new_spans = []
            for author, year, raw, type_, span in extractions:
                if not self._span_contained_in_any(span, matched_spans):
                    results.append((author, year, raw, type_, span))
                    if span not in new_spans:
                        new_spans.append(span)
            return new_spans
        
        # 按优先级处理（复杂模式优先）
        
        # 1. 括号内多引用（分号分隔）
        new_spans = add_batch(self.extract_chinese_multiple_parenthetical(sentence))
        matched_spans.extend(new_spans)
        
        # 2. 括号内作者
        new_spans = add_batch(self.extract_chinese_parenthetical(sentence))
        matched_spans.extend(new_spans)
        
        # 3. 中文"等"
        new_spans = add_batch(self.extract_chinese_et_al(sentence))
        matched_spans.extend(new_spans)
        
        # 4. 中文多作者
        new_spans = add_batch(self.extract_chinese_multiple(sentence))
        matched_spans.extend(new_spans)
        
        # 5. 中文单作者
        new_spans = add_batch(self.extract_chinese_single(sentence))
        matched_spans.extend(new_spans)
        
        return results
    
    # ================== 主提取函数 ==================
    
    def extract_citations(
        self,
        text: str,
        citation_format: CitationFormat,
        context_window: int = 1,
    ) -> List[Any]:
        """
        从文本中提取引用信息
        
        Args:
            text: 按 title 聚合后的 Markdown 文本正文
            citation_format: 引用格式（NUMERIC 或 AUTHOR_YEAR）
            context_window: 上下文窗口大小（上下各多少句）
            
        Returns:
            数字型: [NumericCitation, ...]
            作者年份型: [AuthorYearCitation, ...]
        """
        results = []
        
        # 1. 按段落切分（使用标题划分）
        paragraphs = self.split_paragraphs(text)
        
        for title, content in paragraphs:
            # 2. 按句子切分
            sentences = self.split_sentences(content)
            
            for sentence_id, sentence in enumerate(sentences):
                # 3. 获取上下文
                context = self.get_context(sentences, sentence_id, context_window)
                
                # 4. 根据引用格式提取
                if citation_format == CitationFormat.NUMERIC:
                    # 数字型引用
                    citations = self.extract_all_numeric(sentence)
                    for num, raw, type_, span in citations:
                        position = CitationPosition(
                            title=title,
                            sentence_id=sentence_id,
                            anchor_span=span,
                        )
                        citation = NumericCitation(
                            position=position,
                            number=num,
                            citation_anchor=sentence,
                            context=context,
                            citation_type=type_,
                            raw_citation=raw,
                        )
                        results.append(citation)
                
                else:  # AUTHOR_YEAR
                    # 作者年份型引用（包括英文和中文）
                    # 英文
                    en_citations = self.extract_all_author_year(sentence)
                    for author, year, raw, type_, span in en_citations:
                        position = CitationPosition(
                            title=title,
                            sentence_id=sentence_id,
                            anchor_span=span,
                        )
                        citation = AuthorYearCitation(
                            position=position,
                            authors=author,
                            year=year,
                            citation_anchor=sentence,
                            context=context,
                            citation_type=type_,
                            raw_citation=raw,
                        )
                        results.append(citation)
                    
                    # 中文
                    cn_citations = self.extract_all_chinese(sentence)
                    for author, year, raw, type_, span in cn_citations:
                        position = CitationPosition(
                            title=title,
                            sentence_id=sentence_id,
                            anchor_span=span,
                        )
                        citation = AuthorYearCitation(
                            position=position,
                            authors=author,
                            year=year,
                            citation_anchor=sentence,
                            context=context,
                            citation_type=type_,
                            raw_citation=raw,
                        )
                        results.append(citation)
        
        return results
    
    def extract_to_list(
        self,
        text: str,
        citation_format: CitationFormat,
        context_window: int = 1,
    ) -> List[List]:
        """
        从文本中提取引用信息并转换为列表格式
        
        Args:
            text: 按 title 聚合后的 Markdown 文本正文
            citation_format: 引用格式（NUMERIC 或 AUTHOR_YEAR）
            context_window: 上下文窗口大小
            
        Returns:
            数字型: [[position, 数字标号, citation_anchor, context], ...]
            作者年份型: [[position, 作者, 年份, citation_anchor, context], ...]
            
            position = (title, sentence_id, anchor_span)
        """
        citations = self.extract_citations(text, citation_format, context_window)
        return [c.to_list() for c in citations]


# ================== 便捷函数 ==================

def extract_numeric_citations(
    text: str,
    context_window: int = 1,
) -> List[List]:
    """
    便捷函数：提取数字型引用
    
    Args:
        text: Markdown 文本
        context_window: 上下文窗口大小
        
    Returns:
        [[position, 数字标号, citation_anchor, context], ...]
        position = (title, sentence_id, anchor_span)
    """
    extractor = CitationExtractor()
    return extractor.extract_to_list(text, CitationFormat.NUMERIC, context_window)


def extract_author_year_citations(
    text: str,
    context_window: int = 1,
) -> List[List]:
    """
    便捷函数：提取作者年份型引用
    
    Args:
        text: Markdown 文本
        context_window: 上下文窗口大小
        
    Returns:
        [[position, 作者, 年份, citation_anchor, context], ...]
        position = (title, sentence_id, anchor_span)
    """
    extractor = CitationExtractor()
    return extractor.extract_to_list(text, CitationFormat.AUTHOR_YEAR, context_window)
# with open("D:\projects\CiteVerify\output\converted_document.md","r",encoding="utf-8") as f:
#     text=f.read()
# cs=extract_numeric_citations(text,2)
# for c in cs:
#     print(c[1],c[2])