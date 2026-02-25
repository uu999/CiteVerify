# -*- coding: utf-8 -*-
"""
数据源验证器模块

包括文献校验、引文-参考文献匹配、相关性分析等
"""
from .reference_checker import (
    ReferenceChecker,
    VerificationResult,
    SearchSource,
    verify_references,
    verify_single_reference,
)
from .enhanced_reference_checker import (
    EnhancedReferenceChecker,
    canonicalize_title,
    generate_title_variants,
)
from .citation_matcher import (
    CitationMatcher,
    MatchedCitation,
    UnmatchedCitation,
    MatchType,
    match_citations,
    match_citations_to_list,
)
from .relevance_analyzer import (
    RelevanceAnalyzer,
    RelevanceResult,
    RelevanceJudgment,
    analyze_relevance,
    analyze_relevance_batch,
    generate_relevance_report,
)

__all__ = [
    # 参考文献验证（原始版本 - 备用）
    'ReferenceChecker',
    # 参考文献验证（增强版本 - 默认使用）
    'EnhancedReferenceChecker',
    'canonicalize_title',
    'generate_title_variants',
    # 通用
    'VerificationResult',
    'SearchSource',
    'verify_references',
    'verify_single_reference',
    # 引用匹配
    'CitationMatcher',
    'MatchedCitation',
    'UnmatchedCitation',
    'MatchType',
    'match_citations',
    'match_citations_to_list',
    # 相关性分析
    'RelevanceAnalyzer',
    'RelevanceResult',
    'RelevanceJudgment',
    'analyze_relevance',
    'analyze_relevance_batch',
    'generate_relevance_report',
]
