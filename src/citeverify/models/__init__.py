# -*- coding: utf-8 -*-
"""
数据模型模块
"""
from .reference import (
    ListingStyle,
    CitationFormat,
    ReferenceEntry,
)

from .document import (
    SeparationMethod,
    Section,
    ConversionResult,
    MinerUConfig,
    LLMConfig,
    YaYiDocParserConfig,
)

__all__ = [
    # reference models
    'ListingStyle',
    'CitationFormat',
    'ReferenceEntry',
    # document models
    'SeparationMethod',
    'Section',
    'ConversionResult',
    'MinerUConfig',
    'LLMConfig',
    'YaYiDocParserConfig',
]
