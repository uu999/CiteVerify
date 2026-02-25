# -*- coding: utf-8 -*-
"""
参考文献数据模型
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class ListingStyle(Enum):
    """参考文献条目列举方式"""
    NUMBERED = "numbered"  # 数字标号型: 1. xxx, [1] xxx, (1) xxx
    AUTHOR_YEAR = "author_year"  # 作者年份型（无数字标号）


class CitationFormat(Enum):
    """参考文献引用格式"""
    MLA = "mla"  # Modern Language Association
    APA = "apa"  # American Psychological Association
    GB_T_7714 = "gb_t_7714"  # 中国国家标准
    IEEE = "ieee"  # Institute of Electrical and Electronics Engineers
    CHICAGO = "chicago"  # Chicago Manual of Style
    HARVARD = "harvard"  # Harvard Referencing
    VANCOUVER = "vancouver"  # Vancouver Style
    OTHER = "other"  # 其他格式，使用 LLM 从全文提取


@dataclass
class ReferenceEntry:
    """
    参考文献条目
    
    Attributes:
        full_content: 参考文献条目全内容
        title: 提取出的参考文献题目
        title_normalized: 标准化后的题目（去除标点符号）
        authors: 提取的作者列表
        year: 提取出的年份
        number: 文献编号（仅数字标号型有）
    """
    full_content: str
    title: Optional[str] = None
    title_normalized: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    number: Optional[int] = None
    
    def to_list(self) -> list:
        """
        转换为列表格式
        
        Returns:
            有数字标号型: [编号, 全内容, 题目, 作者, 年份]
            无数字标号型: [全内容, 题目, 作者, 年份]
        """
        authors_str = "; ".join(self.authors) if self.authors else ""
        if self.number is not None:
            return [
                self.number,
                self.full_content,
                self.title_normalized or self.title or "",
                authors_str,
                self.year or ""
            ]
        else:
            return [
                self.full_content,
                self.title_normalized or self.title or "",
                authors_str,
                self.year or ""
            ]

    def __repr__(self) -> str:
        fields = []
        if self.full_content is not None:
            fields.append(f"full_content={self.full_content}")
        if self.number is not None:
            fields.append(f"number={self.number}")

        fields.append(f"title={self.title!r}")

        if self.authors:
            fields.append(f"authors={self.authors!r}")

        if self.year:
            fields.append(f"year={self.year!r}")

        if self.title_normalized:
            fields.append(f"title_normalized={self.title_normalized!r}")

        return f"ReferenceEntry({', '.join(fields)})"

