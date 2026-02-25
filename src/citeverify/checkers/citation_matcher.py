# -*- coding: utf-8 -*-
"""
参考文献与引用文本匹配器

将引用文本与参考文献列表进行匹配，建立引用与文献的对应关系。

支持两种匹配方式：
1. 数字型：根据 [数字] 直接匹配
2. 作者年份型：根据作者和年份进行模糊匹配
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any
from enum import Enum

# 配置日志
logger = logging.getLogger(__name__)

from ..models.reference import ReferenceEntry
from ..extractor.citation_extractor import (
    NumericCitation,
    AuthorYearCitation,
    CitationPosition,
)


class MatchType(Enum):
    """匹配类型"""
    NUMERIC = "numeric"
    AUTHOR_YEAR = "author_year"


@dataclass
class MatchedCitation:
    """匹配结果"""
    # 来自参考文献
    title: str                          # 文献标题
    authors: str                        # 文献作者
    year: str                           # 文献年份
    reference_number: Optional[int]     # 文献编号（数字型）
    
    # 来自引用文本
    citation_anchor: str                # 引用所在句子
    context: str                        # 引用上下文
    citation_position: CitationPosition # 引用位置
    
    # 匹配信息
    match_score: float                  # 匹配度得分 (0.0 - 1.0)
    match_type: MatchType               # 匹配类型
    
    # 可选信息（需要额外获取）
    abstract: str = ""                  # 文献摘要
    pdf_url: str = ""                   # 文献 PDF URL
    
    def to_list(self) -> List:
        """
        转换为列表格式
        
        Returns:
            [title, authors, year, abstract, pdf_url, citation_anchor, context, match_score]
        """
        return [
            self.title,
            self.authors,
            self.year,
            self.abstract,
            self.pdf_url,
            self.citation_anchor,
            self.context,
            self.match_score,
        ]


@dataclass
class UnmatchedCitation:
    """未匹配的引用"""
    citation_anchor: str
    context: str
    citation_position: CitationPosition
    reason: str  # 未匹配原因
    
    # 数字型特有
    number: Optional[int] = None
    
    # 作者年份型特有
    authors: Optional[str] = None
    year: Optional[str] = None


class CitationMatcher:
    """
    参考文献与引用文本匹配器
    
    将提取的引用文本与参考文献列表进行匹配，
    建立引用与文献的对应关系。
    """
    
    def __init__(self):
        """初始化匹配器"""
        # 作者标准化时需要去除的词
        self._author_stop_words = {
            'et', 'al', 'and', '&', 'etc', '等',
            'jr', 'sr', 'dr', 'mr', 'mrs', 'ms', 'prof',
        }
        # 标点符号正则
        self._punctuation_pattern = re.compile(r'[^\w\s\u4e00-\u9fff]')
    
    # ================== 作者标准化 ==================
    
    def normalize_author(self, author: str) -> str:
        """
        标准化作者名称
        
        处理步骤：
        1. 转小写
        2. 去除标点符号
        3. 去除 et al., and, & 等（包括中文的"等"）
        4. 合并多余空格
        
        Args:
            author: 原始作者字符串
            
        Returns:
            标准化后的作者字符串
        """
        if not author:
            return ""
        
        # 1. 转小写（对英文有效）
        normalized = author.lower()
        
        # 2. 特殊处理中文的"等"字（因为中文没有空格分隔）
        normalized = re.sub(r'等$', '', normalized)  # 去除末尾的"等"
        normalized = re.sub(r'等\s', ' ', normalized)  # 去除后面有空格的"等"
        
        # 3. 去除标点符号（保留中文字符和字母数字）
        normalized = self._punctuation_pattern.sub(' ', normalized)
        
        # 4. 分词并去除停用词
        words = normalized.split()
        words = [w for w in words if w not in self._author_stop_words]
        
        # 5. 合并空格
        normalized = ' '.join(words)
        
        return normalized.strip()
    
    def get_author_words(self, author: str) -> List[str]:
        """
        获取作者名称的单词列表
        
        Args:
            author: 原始作者字符串
            
        Returns:
            标准化后的单词列表
        """
        normalized = self.normalize_author(author)
        if not normalized:
            return []
        return normalized.split()
    
    # ================== 相似度计算 ==================
    
    def calculate_author_similarity(
        self,
        citation_author: str,
        reference_authors: Union[str, List[str]],
    ) -> float:
        """
        计算引用作者与参考文献作者的相似度
        
        匹配策略：
        1. 引用中通常只有第一作者（如 "Smith et al."）
        2. 参考文献中有完整作者列表（如 "Smith, Brown, Jones"）
        3. 需要检查引用作者是否出现在参考文献作者中
        
        计算方式：
        - 获取引用作者的单词集合
        - 获取参考文献所有作者的单词集合
        - 计算：引用单词在参考文献中出现的比例 + Jaccard 加权
        
        Args:
            citation_author: 引用中的作者（如 "Smith et al."）
            reference_authors: 参考文献中的作者（字符串或列表）
            
        Returns:
            相似度得分 (0.0 - 1.0)
        """
        # 获取引用作者单词
        cite_words = set(self.get_author_words(citation_author))
        if not cite_words:
            return 0.0
        
        # 获取参考文献作者单词
        if isinstance(reference_authors, list):
            ref_author_str = ' '.join(reference_authors)
        else:
            ref_author_str = reference_authors
        ref_words = set(self.get_author_words(ref_author_str))
        if not ref_words:
            return 0.0
        
        # 计算引用单词在参考文献中出现的比例
        matched_words = cite_words & ref_words
        if not matched_words:
            return 0.0
        
        # 覆盖率：引用中的单词有多少出现在参考文献中
        coverage = len(matched_words) / len(cite_words)
        
        # Jaccard 相似度作为辅助
        jaccard = len(matched_words) / len(cite_words | ref_words)
        
        # 综合得分：覆盖率权重更高（因为引用通常是简写形式）
        score = 0.7 * coverage + 0.3 * jaccard
        
        return score
    
    def calculate_year_match(
        self,
        citation_year: str,
        reference_year: str,
    ) -> Tuple[bool, float]:
        """
        计算年份匹配
        
        Args:
            citation_year: 引用中的年份（可能带后缀如 "2020a"）
            reference_year: 参考文献中的年份
            
        Returns:
            (是否匹配, 匹配得分)
        """
        if not citation_year or not reference_year:
            # 年份缺失，不进行年份过滤
            return True, 0.5
        
        # 提取纯数字年份
        cite_year_num = re.match(r'(\d{4})', citation_year)
        ref_year_num = re.match(r'(\d{4})', reference_year)
        
        if not cite_year_num or not ref_year_num:
            return True, 0.5
        
        cite_year = int(cite_year_num.group(1))
        ref_year = int(ref_year_num.group(1))
        
        # 精确匹配
        if cite_year == ref_year:
            return True, 1.0
        
        # 允许 ±1 年的误差（考虑到出版年份的差异）
        if abs(cite_year - ref_year) <= 1:
            return True, 0.8
        
        return False, 0.0
    
    # ================== 数字型匹配 ==================
    
    def match_numeric(
        self,
        references: List[ReferenceEntry],
        citations: List[NumericCitation],
        verbose: bool = True,
    ) -> Tuple[List[MatchedCitation], List[UnmatchedCitation]]:
        """
        数字型引用匹配
        
        根据引用中的数字编号与参考文献列表中的编号进行匹配。
        
        Args:
            references: 参考文献列表
            citations: 数字型引用列表
            verbose: 是否输出详细日志
            
        Returns:
            (匹配结果列表, 未匹配引用列表)
        """
        matched = []
        unmatched = []
        
        # 日志输出
        if verbose:
            logger.info("=" * 60)
            logger.info("[数字型匹配] 开始匹配")
            logger.info(f"  参考文献数量: {len(references)}")
            logger.info(f"  引用数量: {len(citations)}")
        
        # 构建编号到参考文献的映射
        ref_by_number: Dict[int, ReferenceEntry] = {}
        for ref in references:
            if ref.number is not None:
                ref_by_number[ref.number] = ref
        
        if verbose:
            logger.info(f"  参考文献编号映射: {sorted(ref_by_number.keys())}")
            # 打印前 5 个参考文献详情
            logger.info("  参考文献样例:")
            for i, ref in enumerate(references[:5]):
                logger.info(f"    [{ref.number}] title={ref.title[:50] if ref.title else 'None'}...")
            if len(references) > 5:
                logger.info(f"    ... 还有 {len(references) - 5} 条")
        
        # 提取引用中的编号
        if verbose:
            citation_numbers = [c.number for c in citations]
            logger.info(f"  引用中的编号: {sorted(set(citation_numbers))}")
            # 打印前 5 个引用详情
            logger.info("  引用样例:")
            for i, c in enumerate(citations[:5]):
                logger.info(f"    [{c.number}] anchor={c.citation_anchor[:50]}...")
            if len(citations) > 5:
                logger.info(f"    ... 还有 {len(citations) - 5} 条")
        
        # 匹配每个引用
        for citation in citations:
            number = citation.number
            
            if number in ref_by_number:
                ref = ref_by_number[number]
                
                # 检查参考文献是否有有效标题
                if not ref.title and not ref.title_normalized:
                    if verbose:
                        logger.warning(f"  [未匹配] 编号 [{number}] 参考文献无标题")
                    unmatched.append(UnmatchedCitation(
                        citation_anchor=citation.citation_anchor,
                        context=citation.context,
                        citation_position=citation.position,
                        reason="参考文献提取失败，无标题",
                        number=number,
                    ))
                    continue
                
                # 创建匹配结果
                match_result = MatchedCitation(
                    title=ref.title_normalized or ref.title or "",
                    authors="; ".join(ref.authors) if ref.authors else "",
                    year=ref.year or "",
                    reference_number=number,
                    citation_anchor=citation.citation_anchor,
                    context=citation.context,
                    citation_position=citation.position,
                    match_score=1.0,  # 数字型精确匹配
                    match_type=MatchType.NUMERIC,
                )
                matched.append(match_result)
                if verbose:
                    logger.debug(f"  [匹配成功] [{number}] -> {ref.title[:30]}...")
            else:
                # 引用编号不在参考文献列表中
                if verbose:
                    logger.warning(f"  [未匹配] 编号 [{number}] 不在参考文献列表中")
                unmatched.append(UnmatchedCitation(
                    citation_anchor=citation.citation_anchor,
                    context=citation.context,
                    citation_position=citation.position,
                    reason=f"引用编号 [{number}] 不在参考文献列表中",
                    number=number,
                ))
        
        if verbose:
            logger.info(f"[数字型匹配] 完成: 匹配 {len(matched)}, 未匹配 {len(unmatched)}")
            logger.info("=" * 60)
        
        return matched, unmatched
    
    # ================== 作者年份型匹配 ==================
    
    def match_author_year(
        self,
        references: List[ReferenceEntry],
        citations: List[AuthorYearCitation],
        min_score: float = 0.3,
        verbose: bool = True,
    ) -> Tuple[List[MatchedCitation], List[UnmatchedCitation]]:
        """
        作者年份型引用匹配
        
        匹配步骤：
        1. 年份过滤（如果引用有年份）
        2. 作者相似度计算
        3. 选择匹配度最高的参考文献
        
        Args:
            references: 参考文献列表
            citations: 作者年份型引用列表
            min_score: 最低匹配分数阈值
            verbose: 是否输出详细日志
            
        Returns:
            (匹配结果列表, 未匹配引用列表)
        """
        matched = []
        unmatched = []
        
        if verbose:
            logger.info("=" * 60)
            logger.info("[作者年份型匹配] 开始匹配")
            logger.info(f"  参考文献数量: {len(references)}")
            logger.info(f"  引用数量: {len(citations)}")
            logger.info(f"  最低匹配分数阈值: {min_score}")
            # 打印参考文献样例
            logger.info("  参考文献样例:")
            for i, ref in enumerate(references[:5]):
                authors_str = "; ".join(ref.authors[:2]) if ref.authors else "None"
                logger.info(f"    {authors_str} ({ref.year}) - {ref.title[:40] if ref.title else 'None'}...")
            if len(references) > 5:
                logger.info(f"    ... 还有 {len(references) - 5} 条")
            # 打印引用样例
            logger.info("  引用样例:")
            for i, c in enumerate(citations[:5]):
                logger.info(f"    ({c.authors}, {c.year}) anchor={c.citation_anchor[:40]}...")
            if len(citations) > 5:
                logger.info(f"    ... 还有 {len(citations) - 5} 条")
        
        for citation in citations:
            best_match: Optional[Tuple[ReferenceEntry, float]] = None
            
            for ref in references:
                # 1. 年份过滤
                year_match, year_score = self.calculate_year_match(
                    citation.year, ref.year
                )
                if not year_match:
                    continue
                
                # 2. 作者相似度
                author_score = self.calculate_author_similarity(
                    citation.authors,
                    ref.authors if ref.authors else "",
                )
                
                # 3. 综合得分
                # 年份权重 0.3，作者权重 0.7
                total_score = 0.3 * year_score + 0.7 * author_score
                
                # 更新最佳匹配
                if best_match is None or total_score > best_match[1]:
                    best_match = (ref, total_score)
            
            # 判断是否匹配成功
            if best_match and best_match[1] >= min_score:
                ref, score = best_match
                
                match_result = MatchedCitation(
                    title=ref.title_normalized or ref.title or "",
                    authors="; ".join(ref.authors) if ref.authors else "",
                    year=ref.year or "",
                    reference_number=ref.number,
                    citation_anchor=citation.citation_anchor,
                    context=citation.context,
                    citation_position=citation.position,
                    match_score=round(score, 3),
                    match_type=MatchType.AUTHOR_YEAR,
                )
                matched.append(match_result)
                if verbose:
                    logger.debug(f"  [匹配成功] ({citation.authors}, {citation.year}) -> {ref.title[:30]}... (score={score:.3f})")
            else:
                # 未找到匹配
                reason = "未找到匹配的参考文献"
                if best_match:
                    reason = f"最高匹配度 {best_match[1]:.2f} 低于阈值 {min_score}"
                
                if verbose:
                    logger.warning(f"  [未匹配] ({citation.authors}, {citation.year}): {reason}")
                
                unmatched.append(UnmatchedCitation(
                    citation_anchor=citation.citation_anchor,
                    context=citation.context,
                    citation_position=citation.position,
                    reason=reason,
                    authors=citation.authors,
                    year=citation.year,
                ))
        
        if verbose:
            logger.info(f"[作者年份型匹配] 完成: 匹配 {len(matched)}, 未匹配 {len(unmatched)}")
            logger.info("=" * 60)
        
        return matched, unmatched
    
    # ================== 主匹配函数 ==================
    
    def match(
        self,
        references: List[ReferenceEntry],
        citations: List[Union[NumericCitation, AuthorYearCitation]],
        match_type: Optional[MatchType] = None,
        min_score: float = 0.3,
        verbose: bool = True,
    ) -> Tuple[List[MatchedCitation], List[UnmatchedCitation]]:
        """
        匹配参考文献与引用文本
        
        Args:
            references: 参考文献列表
            citations: 引用列表（可以是数字型或作者年份型）
            match_type: 匹配类型（如果为 None，则自动检测）
            min_score: 作者年份型匹配的最低分数阈值
            verbose: 是否输出详细日志
            
        Returns:
            (匹配结果列表, 未匹配引用列表)
        """
        if not citations:
            if verbose:
                logger.warning("[匹配] 引用列表为空，无需匹配")
            return [], []
        
        if not references:
            if verbose:
                logger.warning("[匹配] 参考文献列表为空，无法匹配")
            return [], []
        
        # 自动检测匹配类型
        if match_type is None:
            if isinstance(citations[0], NumericCitation):
                match_type = MatchType.NUMERIC
            else:
                match_type = MatchType.AUTHOR_YEAR
        
        if verbose:
            logger.info(f"[匹配] 检测到匹配类型: {match_type.value}")
        
        # 根据类型调用相应的匹配函数
        if match_type == MatchType.NUMERIC:
            numeric_citations = [c for c in citations if isinstance(c, NumericCitation)]
            if verbose and len(numeric_citations) != len(citations):
                logger.warning(f"[匹配] 过滤掉 {len(citations) - len(numeric_citations)} 条非数字型引用")
            return self.match_numeric(references, numeric_citations, verbose=verbose)
        else:
            author_year_citations = [c for c in citations if isinstance(c, AuthorYearCitation)]
            if verbose and len(author_year_citations) != len(citations):
                logger.warning(f"[匹配] 过滤掉 {len(citations) - len(author_year_citations)} 条非作者年份型引用")
            return self.match_author_year(references, author_year_citations, min_score, verbose=verbose)
    
    def match_to_list(
        self,
        references: List[ReferenceEntry],
        citations: List[Union[NumericCitation, AuthorYearCitation]],
        match_type: Optional[MatchType] = None,
        min_score: float = 0.3,
    ) -> List[List]:
        """
        匹配并转换为列表格式
        
        Args:
            references: 参考文献列表
            citations: 引用列表
            match_type: 匹配类型
            min_score: 最低匹配分数阈值
            
        Returns:
            [[title, authors, year, abstract, pdf_url, citation_anchor, context, match_score], ...]
        """
        matched, _ = self.match(references, citations, match_type, min_score)
        return [m.to_list() for m in matched]


# ================== 便捷函数 ==================

def match_citations(
    references: List[ReferenceEntry],
    citations: List[Union[NumericCitation, AuthorYearCitation]],
    match_type: Optional[MatchType] = None,
    min_score: float = 0.3,
) -> Tuple[List[MatchedCitation], List[UnmatchedCitation]]:
    """
    便捷函数：匹配参考文献与引用文本
    
    Args:
        references: 参考文献列表
        citations: 引用列表
        match_type: 匹配类型（None 则自动检测）
        min_score: 作者年份型匹配的最低分数阈值
        
    Returns:
        (匹配结果列表, 未匹配引用列表)
    """
    matcher = CitationMatcher()
    return matcher.match(references, citations, match_type, min_score)


def match_citations_to_list(
    references: List[ReferenceEntry],
    citations: List[Union[NumericCitation, AuthorYearCitation]],
    match_type: Optional[MatchType] = None,
    min_score: float = 0.3,
) -> List[List]:
    """
    便捷函数：匹配并转换为列表格式
    
    Args:
        references: 参考文献列表
        citations: 引用列表
        match_type: 匹配类型
        min_score: 最低匹配分数阈值
        
    Returns:
        [[title, authors, year, abstract, pdf_url, citation_anchor, context, match_score], ...]
    """
    matcher = CitationMatcher()
    return matcher.match_to_list(references, citations, match_type, min_score)
