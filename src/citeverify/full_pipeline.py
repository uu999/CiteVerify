# -*- coding: utf-8 -*-
"""
CiteVerify 完整流水线

完整流程：
1. 输入论文 URL、引用方式、参考文献格式、LLM 配置
2. PDF 转换为 Markdown
3. 并行提取参考文献列表和引用列表
4. 参考文献真伪性校验 -> 生成校验报告
5. 参考文献与引用匹配
6. 并行进行相关性分析 -> 生成相关性报告
"""
import time
import logging
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum

# 配置日志
logger = logging.getLogger(__name__)

from .converter import PDFConverter, separate_markdown
from .extractor import extract_references, CitationExtractor, CitationFormat as CitationExtractFormat
from .models import (
    MinerUConfig, 
    LLMConfig, 
    ListingStyle, 
    CitationFormat,
    YaYiDocParserConfig,
)
from .checkers import (
    ReferenceChecker,
    EnhancedReferenceChecker,
    VerificationResult,
    SearchSource,
    CitationMatcher,
    MatchedCitation,
    UnmatchedCitation,
    RelevanceAnalyzer,
    RelevanceResult,
    RelevanceJudgment,
    generate_relevance_report,
)


@dataclass
class VerificationReport:
    """参考文献真伪性校验报告"""
    total_count: int = 0
    verified_count: int = 0
    not_found_count: int = 0
    arxiv_count: int = 0
    arxiv_via_id_count: int = 0
    semantic_scholar_count: int = 0
    openalex_count: int = 0
    crossref_count: int = 0
    
    # 详细列表
    verified_refs: List[Dict] = field(default_factory=list)
    not_found_refs: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "verified_count": self.verified_count,
            "not_found_count": self.not_found_count,
            "arxiv_count": self.arxiv_count,
            "arxiv_via_id_count": self.arxiv_via_id_count,
            "semantic_scholar_count": self.semantic_scholar_count,
            "openalex_count": self.openalex_count,
            "crossref_count": self.crossref_count,
            "verified_refs": self.verified_refs,
            "not_found_refs": self.not_found_refs,
        }


@dataclass
class RelevanceReport:
    """相关性分析报告"""
    total_citations: int = 0
    matched_count: int = 0
    unmatched_count: int = 0
    
    # 相关性统计
    strongly_supports_count: int = 0
    weakly_supports_count: int = 0
    not_supports_count: int = 0
    unclear_count: int = 0
    error_count: int = 0
    
    # 详细列表
    matched_results: List[Dict] = field(default_factory=list)
    unmatched_citations: List[Dict] = field(default_factory=list)
    unmatched_references: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "matched_count": self.matched_count,
            "unmatched_count": self.unmatched_count,
            "strongly_supports_count": self.strongly_supports_count,
            "weakly_supports_count": self.weakly_supports_count,
            "not_supports_count": self.not_supports_count,
            "unclear_count": self.unclear_count,
            "error_count": self.error_count,
            "matched_results": self.matched_results,
            "unmatched_citations": self.unmatched_citations,
            "unmatched_references": self.unmatched_references,
        }


@dataclass
class FullPipelineResult:
    """完整流水线结果"""
    success: bool = False
    error: Optional[str] = None
    
    # 中间结果
    markdown_content: str = ""
    main_text: str = ""
    references_text: str = ""
    
    # 提取结果
    references: List[list] = field(default_factory=list)
    citations: List = field(default_factory=list)
    
    # 报告
    verification_report: Optional[VerificationReport] = None
    relevance_report: Optional[RelevanceReport] = None
    
    # 处理时间
    conversion_time: float = 0.0
    extraction_time: float = 0.0
    verification_time: float = 0.0
    matching_time: float = 0.0
    analysis_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "references_count": len(self.references),
            "citations_count": len(self.citations),
            "verification_report": self.verification_report.to_dict() if self.verification_report else None,
            "relevance_report": self.relevance_report.to_dict() if self.relevance_report else None,
            "timing": {
                "conversion_time": round(self.conversion_time, 2),
                "extraction_time": round(self.extraction_time, 2),
                "verification_time": round(self.verification_time, 2),
                "matching_time": round(self.matching_time, 2),
                "analysis_time": round(self.analysis_time, 2),
                "total_time": round(self.total_time, 2),
            }
        }


class FullPipeline:
    """
    CiteVerify 完整流水线
    
    处理流程：
    1. PDF 转 Markdown
    2. 并行提取参考文献和引用
    3. 真伪性校验
    4. 匹配
    5. 相关性分析
    """
    
    def __init__(
        self,
        # PDF 转换配置
        mineru_config: Optional[MinerUConfig] = None,
        yayi_config: Optional[YaYiDocParserConfig] = None,
        # 参考文献校验配置
        semantic_scholar_api_key: Optional[str] = None,
        request_delay: float = 1.0,
        # LLM 配置
        llm_model_name: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        # 并行配置
        max_workers: int = 5,
    ):
        """
        初始化流水线
        
        Args:
            mineru_config: MinerU 配置
            yayi_config: 雅意文档解析配置
            semantic_scholar_api_key: Semantic Scholar API Key
            request_delay: API 请求间隔
            llm_model_name: LLM 模型名称
            llm_api_key: LLM API Key
            llm_base_url: LLM API Base URL
            max_workers: 并行相关性分析的最大线程数
        """
        self.mineru_config = mineru_config
        # 如果没有提供 yayi_config，默认使用 YaYiDocParserConfig
        self.yayi_config = yayi_config or YaYiDocParserConfig()
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.request_delay = request_delay
        self.llm_model_name = llm_model_name
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.max_workers = max_workers
        
        # 初始化组件
        self.pdf_converter = PDFConverter(
            mineru_config=mineru_config,
            yayi_config=self.yayi_config,
        )
        # 使用增强校验器（强规范化 + 多候选 + arXiv ID 兜底 + Crossref 补充）
        # 原始 ReferenceChecker 保留作为备用方案
        self.reference_checker = EnhancedReferenceChecker(
            request_delay=request_delay,
            semantic_scholar_api_key=semantic_scholar_api_key,
        )
        self.citation_matcher = CitationMatcher()
        
        # LLM 分析器（延迟初始化）
        self._relevance_analyzer = None
    
    @property
    def relevance_analyzer(self) -> Optional[RelevanceAnalyzer]:
        """获取相关性分析器（延迟初始化）"""
        if self._relevance_analyzer is None and self.llm_api_key:
            self._relevance_analyzer = RelevanceAnalyzer(
                model_name=self.llm_model_name,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
        return self._relevance_analyzer
    
    def run(
        self,
        pdf_url: str,
        citation_format: Union[str, CitationFormat],
        listing_style: Union[str, ListingStyle],
        progress_callback: Optional[callable] = None,
    ) -> FullPipelineResult:
        """
        运行完整流水线
        
        Args:
            pdf_url: 论文 PDF URL
            citation_format: 参考文献格式 (apa, ieee, mla, etc.)
            listing_style: 列举方式 (numbered, author_year)
            progress_callback: 进度回调函数 callback(step, message, progress)
            
        Returns:
            FullPipelineResult 对象
        """
        result = FullPipelineResult()
        start_time = time.time()
        
        # 转换参数
        if isinstance(citation_format, str):
            citation_format = CitationFormat(citation_format)
        if isinstance(listing_style, str):
            listing_style = ListingStyle(listing_style)
        
        def report_progress(step: str, message: str, progress: float):
            if progress_callback:
                progress_callback(step, message, progress)
            print(f"[{step}] {message} ({progress:.0%})")
        
        logger.info("[Pipeline] 流水线开始")
        try:
            # ==================== Step 1: PDF 转 Markdown ====================
            logger.info("[Pipeline] PDF 转 Markdown 开始")
            report_progress("conversion", "正在转换 PDF 为 Markdown...", 0.05)
            step_start = time.time()
            
            conversion_result = self.pdf_converter.convert(
                pdf_url,
                yayi_config=self.yayi_config,
            )
            
            result.markdown_content = conversion_result.full_markdown
            result.main_text = conversion_result.main_text
            result.references_text = conversion_result.references_text
            result.conversion_time = time.time() - step_start
            
            if not conversion_result.separation_success:
                result.error = "无法从文档中分离出参考文献部分"
                logger.info("[Pipeline] PDF 转 Markdown 结束（失败：未分离出参考文献）")
                return result
            
            logger.info(f"[Pipeline] PDF 转 Markdown 结束，耗时 {result.conversion_time:.1f}s")
            report_progress("conversion", f"PDF 转换完成，耗时 {result.conversion_time:.1f}s", 0.15)
            
            # ==================== Step 2: 并行提取参考文献和引用 ====================
            logger.info("[Pipeline] 提取参考文献、提取引用 开始")
            report_progress("extraction", "正在提取参考文献和引用...", 0.20)
            step_start = time.time()
            
            # 确定引用提取格式
            is_numeric = listing_style == ListingStyle.NUMBERED
            citation_extract_format = CitationExtractFormat.NUMERIC if is_numeric else CitationExtractFormat.AUTHOR_YEAR
            
            # 初始化引用提取器
            citation_extractor = CitationExtractor()
            
            # 并行提取
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # 提取参考文献（传递 LLM 参数用于备选提取）
                future_refs = executor.submit(
                    extract_references,
                    conversion_result.references_text,
                    listing_style,
                    citation_format,
                    llm_model=self.llm_model_name,
                    llm_api_key=self.llm_api_key,
                    llm_base_url=self.llm_base_url,
                    use_llm_fallback=True,
                )
                
                # 提取引用 - 使用 extract_citations 返回对象，而不是 extract_to_list 返回列表
                future_citations = executor.submit(
                    citation_extractor.extract_citations,
                    conversion_result.main_text,
                    citation_extract_format,
                )
                
                result.references = future_refs.result()
                result.citations = future_citations.result()
            
            result.extraction_time = time.time() - step_start
            logger.info(f"[Pipeline] 提取参考文献、提取引用 结束，耗时 {result.extraction_time:.1f}s，参考文献 {len(result.references)} 条，引用 {len(result.citations)} 条")
            
            # 打印提取结果调试信息
            logger.info("=" * 60)
            logger.info("[Pipeline Step 2] 提取结果")
            logger.info(f"  listing_style: {listing_style}")
            logger.info(f"  is_numeric: {is_numeric}")
            logger.info(f"  参考文献数量: {len(result.references)}")
            logger.info(f"  引用数量: {len(result.citations)}")
            
            # 打印参考文献样例
            if result.references:
                logger.info("  参考文献样例 (前3条):")
                for i, ref in enumerate(result.references[:3]):
                    logger.info(f"    [{i}] {ref}")
            
            # 打印引用样例
            if result.citations:
                logger.info(f"  引用类型: {type(result.citations[0]).__name__}")
                logger.info("  引用样例 (前3条):")
                for i, c in enumerate(result.citations[:3]):
                    logger.info(f"    [{i}] {c}")
            logger.info("=" * 60)
            
            report_progress(
                "extraction", 
                f"提取完成: {len(result.references)} 条参考文献, {len(result.citations)} 条引用, 耗时 {result.extraction_time:.1f}s",
                0.35
            )
            
            if not result.references:
                result.error = "未能提取到参考文献"
                return result
            
            # ==================== Step 3: 参考文献真伪性校验（增强管线） ====================
            logger.info("[Pipeline] 真伪性校验 开始")
            report_progress("verification", "正在校验参考文献真伪性（增强管线：强规范化+多候选+arXiv ID兜底+Crossref补充）...", 0.40)
            step_start = time.time()
            
            has_number = listing_style == ListingStyle.NUMBERED
            verified_refs = self.reference_checker.verify_references(
                result.references,
                has_number=has_number,
                verbose=True,
                max_workers=10,
                batch_size=10,
                batch_delay=0.2,
            )
            
            # 生成校验报告
            result.verification_report = self._build_verification_report(
                verified_refs, has_number
            )
            result.verification_time = time.time() - step_start
            
            logger.info(f"[Pipeline] 真伪性校验 结束，耗时 {result.verification_time:.1f}s，可验证 {result.verification_report.verified_count}/{result.verification_report.total_count}")
            report_progress(
                "verification",
                f"校验完成: {result.verification_report.verified_count}/{result.verification_report.total_count} 可验证, 耗时 {result.verification_time:.1f}s",
                0.60
            )
            
            # ==================== Step 4: 参考文献与引用匹配 ====================
            logger.info("[Pipeline] 引用-参考文献配对 开始")
            report_progress("matching", "正在匹配参考文献与引用...", 0.65)
            step_start = time.time()
            
            # 构建带摘要和 PDF URL 的参考文献列表
            refs_with_abstract = self._build_refs_with_abstract(verified_refs, has_number)
            
            # 打印调试信息
            logger.info("=" * 60)
            logger.info("[Pipeline Step 4] 参考文献与引用匹配")
            logger.info(f"  has_number (is_numeric): {has_number}")
            logger.info(f"  refs_with_abstract 数量: {len(refs_with_abstract)}")
            logger.info(f"  citations 数量: {len(result.citations)}")
            
            if refs_with_abstract:
                logger.info("  refs_with_abstract 样例 (前3条):")
                for i, ref in enumerate(refs_with_abstract[:3]):
                    logger.info(f"    [{i}] number={ref.number}, title={ref.title[:40] if ref.title else 'None'}..., authors={ref.authors[:2] if ref.authors else []}")
            
            if result.citations:
                logger.info(f"  citations 类型: {type(result.citations[0]).__name__}")
                logger.info("  citations 样例 (前3条):")
                for i, c in enumerate(result.citations[:3]):
                    if hasattr(c, 'number'):
                        logger.info(f"    [{i}] number={c.number}, anchor={c.citation_anchor[:40]}...")
                    elif hasattr(c, 'authors'):
                        logger.info(f"    [{i}] authors={c.authors}, year={c.year}, anchor={c.citation_anchor[:40]}...")
                    else:
                        logger.info(f"    [{i}] {c}")
            logger.info("=" * 60)
            
            # match 会自动检测引用类型
            matched, unmatched = self.citation_matcher.match(
                refs_with_abstract,
                result.citations,
            )
            
            # 补充 abstract 和 pdf_url
            ref_map = {}
            for ref in refs_with_abstract:
                key = (ref.title_normalized or ref.title or "").lower()
                ref_map[key] = ref
                if ref.number is not None:
                    ref_map[f"num_{ref.number}"] = ref
            
            for m in matched:
                # 通过标题或编号查找对应的参考文献
                ref = None
                if m.reference_number is not None:
                    ref = ref_map.get(f"num_{m.reference_number}")
                if not ref and m.title:
                    ref = ref_map.get(m.title.lower())
                
                if ref:
                    m.abstract = getattr(ref, 'abstract', '') or ''
                    m.pdf_url = getattr(ref, 'pdf_url', '') or ''
            
            result.matching_time = time.time() - step_start
            logger.info(f"[Pipeline] 引用-参考文献配对 结束，耗时 {result.matching_time:.1f}s，匹配 {len(matched)}，未匹配 {len(unmatched)}")
            report_progress(
                "matching",
                f"匹配完成: {len(matched)} 匹配, {len(unmatched)} 未匹配, 耗时 {result.matching_time:.1f}s",
                0.75
            )
            
            # ==================== Step 5: 相关性分析 ====================
            if self.relevance_analyzer and matched:
                logger.info("[Pipeline] 相关性分析 开始")
                report_progress("analysis", "正在进行相关性分析...", 0.80)
                step_start = time.time()
                
                # 过滤出有摘要的匹配项
                analyzable = [m for m in matched if m.abstract]
                
                if analyzable:
                    # 进行分析
                    def analysis_progress(current, total):
                        progress = 0.80 + 0.15 * (current / total)
                        report_progress("analysis", f"分析进度: {current}/{total}", progress)
                    
                    relevance_results = self.relevance_analyzer.analyze_matched_citations(
                        analyzable,
                        progress_callback=analysis_progress,
                        max_workers=self.max_workers,
                    )
                    
                    # 生成相关性报告
                    result.relevance_report = self._build_relevance_report(
                        matched, unmatched, relevance_results, refs_with_abstract
                    )
                else:
                    # 没有可分析的匹配项（都没有摘要）
                    result.relevance_report = self._build_relevance_report(
                        matched, unmatched, [], refs_with_abstract
                    )
                
                result.analysis_time = time.time() - step_start
                logger.info(f"[Pipeline] 相关性分析 结束，耗时 {result.analysis_time:.1f}s")
                report_progress(
                    "analysis",
                    f"分析完成, 耗时 {result.analysis_time:.1f}s",
                    0.95
                )
            else:
                # 没有 LLM 配置，跳过相关性分析
                logger.info("[Pipeline] 相关性分析 跳过（无 LLM 或无匹配项）")
                result.relevance_report = self._build_relevance_report(
                    matched, unmatched, [], refs_with_abstract, skip_analysis=True
                )
            
            result.success = True
            result.total_time = time.time() - start_time
            logger.info(f"[Pipeline] 全部完成，总耗时 {result.total_time:.1f}s")
            report_progress("done", f"全部完成，总耗时 {result.total_time:.1f}s", 1.0)
            
        except Exception as e:
            result.error = str(e)
            result.total_time = time.time() - start_time
            logger.exception("[Pipeline] 执行异常: %s", e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def _build_verification_report(
        self, 
        verified_refs: List[list],
        has_number: bool,
    ) -> VerificationReport:
        """构建真伪性校验报告"""
        report = VerificationReport()
        report.total_count = len(verified_refs)
        
        for ref in verified_refs:
            # 解析字段位置
            if has_number:
                # [编号, 全文, 标题, 作者, 年份, can_get, abstract, pdf_url, source]
                number = ref[0]
                title = ref[2]
                can_get = ref[5] if len(ref) > 5 else False
                abstract = ref[6] if len(ref) > 6 else None
                pdf_url = ref[7] if len(ref) > 7 else None
                source = ref[8] if len(ref) > 8 else "not_found"
            else:
                # [全文, 标题, 作者, 年份, can_get, abstract, pdf_url, source]
                number = None
                title = ref[1]
                can_get = ref[4] if len(ref) > 4 else False
                abstract = ref[5] if len(ref) > 5 else None
                pdf_url = ref[6] if len(ref) > 6 else None
                source = ref[7] if len(ref) > 7 else "not_found"
            
            ref_info = {
                "number": number,
                "title": title,
                "can_get": can_get,
                "source": source,
                "pdf_url": pdf_url,
                "abstract": abstract or "",  # 返回完整摘要内容
                "has_abstract": bool(abstract),
            }
            
            if can_get:
                report.verified_count += 1
                report.verified_refs.append(ref_info)
                
                if source == "arxiv":
                    report.arxiv_count += 1
                elif source == "arxiv_via_id":
                    report.arxiv_via_id_count += 1
                elif source == "semantic_scholar":
                    report.semantic_scholar_count += 1
                elif source == "openalex":
                    report.openalex_count += 1
                elif source == "crossref":
                    report.crossref_count += 1
            else:
                report.not_found_count += 1
                report.not_found_refs.append(ref_info)
        
        return report
    
    def _build_refs_with_abstract(
        self,
        verified_refs: List[list],
        has_number: bool,
    ) -> List:
        """构建带摘要的参考文献列表（用于 CitationMatcher）"""
        from .models.reference import ReferenceEntry
        
        entries = []
        for ref in verified_refs:
            if has_number:
                # [编号, 全文, 标题, 作者, 年份, can_get, abstract, pdf_url, source]
                entry = ReferenceEntry(
                    full_content=ref[1],
                    title=ref[2],
                    title_normalized=ref[2],
                    authors=ref[3].split("; ") if ref[3] else [],
                    year=ref[4],
                    number=ref[0],
                )
                # 附加摘要和 PDF URL
                entry.abstract = ref[6] if len(ref) > 6 else None
                entry.pdf_url = ref[7] if len(ref) > 7 else None
            else:
                entry = ReferenceEntry(
                    full_content=ref[0],
                    title=ref[1],
                    title_normalized=ref[1],
                    authors=ref[2].split("; ") if ref[2] else [],
                    year=ref[3],
                )
                entry.abstract = ref[5] if len(ref) > 5 else None
                entry.pdf_url = ref[6] if len(ref) > 6 else None
            
            entries.append(entry)
        
        return entries
    
    def _build_relevance_report(
        self,
        matched: List[MatchedCitation],
        unmatched: List[UnmatchedCitation],
        relevance_results: List[RelevanceResult],
        all_refs: List,
        skip_analysis: bool = False,
    ) -> RelevanceReport:
        """构建相关性报告"""
        report = RelevanceReport()
        report.total_citations = len(matched) + len(unmatched)
        report.matched_count = len(matched)
        report.unmatched_count = len(unmatched)
        
        # 处理匹配结果
        result_map = {}
        for r in relevance_results:
            key = (r.title, r.citation_anchor)
            result_map[key] = r
        
        for m in matched:
            key = (m.title, m.citation_anchor)
            r = result_map.get(key)
            
            matched_info = {
                "title": m.title,
                "authors": m.authors,
                "year": m.year,
                "citation_anchor": m.citation_anchor,
                "context": m.context,  # 返回完整 context，前端负责截断显示
                "match_score": m.match_score,
                "has_abstract": bool(m.abstract),
            }
            
            if r:
                matched_info["claim"] = r.claim
                matched_info["judgment"] = r.judgment.value
                matched_info["reason"] = r.reason
                matched_info["analysis_success"] = r.success
                matched_info["error_message"] = r.error_message or ""
                
                # 统计
                if r.judgment == RelevanceJudgment.STRONGLY_SUPPORTS:
                    report.strongly_supports_count += 1
                elif r.judgment == RelevanceJudgment.WEAKLY_SUPPORTS:
                    report.weakly_supports_count += 1
                elif r.judgment == RelevanceJudgment.DOES_NOT_SUPPORT:
                    report.not_supports_count += 1
                elif r.judgment == RelevanceJudgment.ERROR:
                    report.error_count += 1
                else:
                    report.unclear_count += 1
            elif skip_analysis:
                matched_info["claim"] = ""
                matched_info["judgment"] = "skipped"
                matched_info["reason"] = "未配置 LLM，跳过相关性分析"
                matched_info["analysis_success"] = False
                matched_info["error_message"] = ""
            else:
                matched_info["claim"] = ""
                matched_info["judgment"] = "no_abstract"
                matched_info["reason"] = "参考文献无摘要，无法进行相关性分析"
                matched_info["analysis_success"] = False
                matched_info["error_message"] = ""
                report.unclear_count += 1
            
            report.matched_results.append(matched_info)
        
        # 处理未匹配的引用
        for u in unmatched:
            report.unmatched_citations.append({
                "citation_anchor": u.citation_anchor,
                "context": u.context,  # 返回完整 context
                "reason": u.reason,
            })
        
        # 找出未被引用的参考文献
        cited_titles = set()
        for m in matched:
            if m.title:
                cited_titles.add(m.title.lower())
        
        for ref in all_refs:
            title = ref.title if hasattr(ref, 'title') else ""
            if title and title.lower() not in cited_titles:
                report.unmatched_references.append({
                    "title": title,
                    "number": ref.number if hasattr(ref, 'number') else None,
                    "reason": "未在正文中找到对应引用",
                })
        
        return report


def run_full_pipeline(
    pdf_url: str,
    citation_format: str,
    listing_style: str,
    llm_model_name: str = "gpt-4o-mini",
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    semantic_scholar_api_key: Optional[str] = None,
    yayi_config: Optional[YaYiDocParserConfig] = None,
    progress_callback: Optional[callable] = None,
) -> FullPipelineResult:
    """
    便捷函数：运行完整流水线
    
    Args:
        pdf_url: 论文 PDF URL
        citation_format: 参考文献格式 (apa, ieee, mla, gb_t_7714, chicago, harvard, vancouver)
        listing_style: 列举方式 (numbered, author_year)
        llm_model_name: LLM 模型名称
        llm_api_key: LLM API Key
        llm_base_url: LLM API Base URL
        semantic_scholar_api_key: Semantic Scholar API Key
        yayi_config: 雅意文档解析配置
        progress_callback: 进度回调
        
    Returns:
        FullPipelineResult 对象
    """
    pipeline = FullPipeline(
        yayi_config=yayi_config,  # 默认值在 FullPipeline 中处理
        semantic_scholar_api_key=semantic_scholar_api_key,
        llm_model_name=llm_model_name,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )
    
    return pipeline.run(
        pdf_url=pdf_url,
        citation_format=citation_format,
        listing_style=listing_style,
        progress_callback=progress_callback,
    )
