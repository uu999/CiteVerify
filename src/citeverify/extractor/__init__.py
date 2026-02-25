# -*- coding: utf-8 -*-
"""
抽取器模块

包含参考文献抽取和引用定位等功能
"""
from .reference_extractor import (
    ReferenceExtractor,
    extract_references,
)
from .citation_extractor import (
    CitationExtractor,
    CitationFormat,
    CitationPosition,
    NumericCitation,
    AuthorYearCitation,
    extract_numeric_citations,
    extract_author_year_citations,
)

__all__ = [
    # 参考文献提取
    'ReferenceExtractor',
    'extract_references',
    # 引用定位
    'CitationExtractor',
    'CitationFormat',
    'CitationPosition',
    'NumericCitation',
    'AuthorYearCitation',
    'extract_numeric_citations',
    'extract_author_year_citations',
]
