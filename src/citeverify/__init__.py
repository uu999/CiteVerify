# CiteVerify - 引文验证工具
"""
CiteVerify 核心包

主要功能：
- extract_references_from_url: 从 PDF URL 提取参考文献
- extract_references_from_markdown: 从 Markdown 提取参考文献
- verify_references: 批量校验参考文献真伪
- verify_single_reference: 校验单条参考文献
- run_full_pipeline: 运行完整验证流水线
"""

__version__ = "0.1.0"

from .pipeline import (
    extract_references_from_url,
    extract_references_from_markdown,
    PipelineResult,
)

from .checkers import (
    ReferenceChecker,
    VerificationResult,
    SearchSource,
    verify_references,
    verify_single_reference,
    CitationMatcher,
    MatchedCitation,
    UnmatchedCitation,
    RelevanceAnalyzer,
    RelevanceResult,
    RelevanceJudgment,
)

from .full_pipeline import (
    FullPipeline,
    FullPipelineResult,
    VerificationReport,
    RelevanceReport,
    run_full_pipeline,
)

__all__ = [
    # 基础流水线
    'extract_references_from_url',
    'extract_references_from_markdown',
    'PipelineResult',
    # 参考文献校验
    'ReferenceChecker',
    'VerificationResult',
    'SearchSource',
    'verify_references',
    'verify_single_reference',
    # 引用匹配
    'CitationMatcher',
    'MatchedCitation',
    'UnmatchedCitation',
    # 相关性分析
    'RelevanceAnalyzer',
    'RelevanceResult',
    'RelevanceJudgment',
    # 完整流水线
    'FullPipeline',
    'FullPipelineResult',
    'VerificationReport',
    'RelevanceReport',
    'run_full_pipeline',
]
