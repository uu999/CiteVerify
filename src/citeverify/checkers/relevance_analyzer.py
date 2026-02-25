# -*- coding: utf-8 -*-
"""
引文相关性分析器

使用 LLM 分析引用文本与参考文献之间的相关性。
判断参考文献是否支持引用处的论点。
"""
import re
import json
import logging
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class RelevanceJudgment(Enum):
    """相关性判断结果"""
    STRONGLY_SUPPORTS = "strongly_supports"
    WEAKLY_SUPPORTS = "weakly_supports"
    DOES_NOT_SUPPORT = "does_not_support"
    UNCLEAR = "unclear"
    ERROR = "error"  # 分析失败
    
    @classmethod
    def from_string(cls, text: str) -> 'RelevanceJudgment':
        """从字符串解析判断结果"""
        text_lower = text.lower().strip()
        
        if 'strongly support' in text_lower:
            return cls.STRONGLY_SUPPORTS
        elif 'weakly support' in text_lower:
            return cls.WEAKLY_SUPPORTS
        elif 'does not support' in text_lower or 'not support' in text_lower:
            return cls.DOES_NOT_SUPPORT
        elif 'unclear' in text_lower:
            return cls.UNCLEAR
        else:
            return cls.UNCLEAR  # 默认返回 unclear


@dataclass
class RelevanceResult:
    """相关性分析结果"""
    # 输入信息
    title: str                          # 参考文献标题
    abstract: str                       # 参考文献摘要
    citation_anchor: str                # 引用所在句子
    context: str                        # 引用上下文
    
    # 分析结果
    claim: str                          # 推断的论点
    judgment: RelevanceJudgment         # 相关性判断
    reason: str                         # 判断理由
    
    # 元信息
    raw_response: str = ""              # LLM 原始响应
    success: bool = True                # 是否分析成功
    error_message: str = ""             # 错误信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "title": self.title,
            "abstract": self.abstract[:200] + "..." if len(self.abstract) > 200 else self.abstract,
            "citation_anchor": self.citation_anchor,
            "context": self.context,
            "claim": self.claim,
            "judgment": self.judgment.value,
            "reason": self.reason,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    def to_list(self) -> List:
        """转换为列表格式"""
        return [
            self.title,
            self.citation_anchor,
            self.claim,
            self.judgment.value,
            self.reason,
            self.success,
        ]


# 提示词模板
RELEVANCE_PROMPT_TEMPLATE = """# Role
You are an academic citation verification assistant.

# Task
Your task is to determine whether a given reference paper supports the claim made at a specific citation point in a manuscript.

# Input
Reference title: {title}
Reference abstract: {abstract}
Citation anchor: {citation_anchor}
Context: {context}

# Instructions
You must follow the steps below strictly and must NOT introduce any information that is not explicitly stated in the provided text.

Step 1:  
Based only on the citation anchor and its surrounding context, infer the claim that the authors intend to make at this citation point.  
If the claim involves reference, omission, comparison, or negation, use the context to clarify it.  
If no clear claim can be inferred, state that explicitly.

Step 2:  
Based only on the reference title and abstract, determine whether the reference supports the inferred claim.

Your judgment must be one of the following four categories:
- Strongly supports
- Weakly supports
- Does not support
- Unclear

Provide a brief reason for your judgment, strictly grounded in the given texts.

# Output Format
You MUST respond with a valid JSON object in the following format (no other text before or after):
```json
{{
    "claim": "The inferred claim from the citation context",
    "judge": "One of: Strongly supports / Weakly supports / Does not support / Unclear",
    "reason": "Brief reason for the judgment"
}}
```
"""


class RelevanceAnalyzer:
    """
    引文相关性分析器
    
    使用 LLM 分析引用文本与参考文献之间的相关性。
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称（如 "gpt-4o-mini", "gpt-4o", "deepseek-chat" 等）
            api_key: API 密钥
            base_url: API 基础 URL（用于兼容其他 OpenAI 风格的 API）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        if OpenAI is None:
            raise ImportError(
                "openai 包未安装。请运行: pip install openai"
            )
        
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 初始化 OpenAI 客户端
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout:
            client_kwargs["timeout"] = timeout
        
        self.client = OpenAI(**client_kwargs)
    
    def _build_prompt(
        self,
        title: str,
        abstract: str,
        citation_anchor: str,
        context: str,
    ) -> str:
        """
        构建提示词
        
        Args:
            title: 参考文献标题
            abstract: 参考文献摘要
            citation_anchor: 引用所在句子
            context: 引用上下文
            
        Returns:
            格式化后的提示词
        """
        return RELEVANCE_PROMPT_TEMPLATE.format(
            title=title,
            abstract=abstract,
            citation_anchor=citation_anchor,
            context=context,
        )
    
    def _parse_response(self, response_text: str) -> Tuple[str, RelevanceJudgment, str]:
        """
        解析 LLM 响应（JSON 格式）
        
        Args:
            response_text: LLM 的原始响应文本
            
        Returns:
            (claim, judgment, reason)
        """
        claim = ""
        judgment = RelevanceJudgment.UNCLEAR
        reason = ""
        
        # 尝试提取 JSON 块
        # 1. 先尝试提取 ```json ... ``` 代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 2. 尝试直接查找 JSON 对象
            json_match = re.search(r'\{[^{}]*"claim"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 3. 回退：尝试整个响应作为 JSON
                json_str = response_text.strip()
        
        # 解析 JSON
        try:
            data = json.loads(json_str)
            claim = data.get("claim", "")
            judge_text = data.get("judge", "")
            reason = data.get("reason", "")
            
            if judge_text:
                judgment = RelevanceJudgment.from_string(judge_text)
            
        except json.JSONDecodeError:
            # JSON 解析失败，回退到正则解析（兼容性）
            claim_match = re.search(
                r'["\']?claim["\']?\s*[:：]\s*["\']?(.+?)["\']?\s*[,}]',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            if claim_match:
                claim = claim_match.group(1).strip().strip('"\'')
            
            judge_match = re.search(
                r'["\']?judge["\']?\s*[:：]\s*["\']?(.+?)["\']?\s*[,}]',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            if judge_match:
                judge_text = judge_match.group(1).strip().strip('"\'')
                judgment = RelevanceJudgment.from_string(judge_text)
            
            reason_match = re.search(
                r'["\']?reason["\']?\s*[:：]\s*["\']?(.+?)["\']?\s*}',
                response_text,
                re.DOTALL | re.IGNORECASE
            )
            if reason_match:
                reason = reason_match.group(1).strip().strip('"\'')
        
        return claim, judgment, reason
    
    def analyze(
        self,
        title: str,
        abstract: str,
        citation_anchor: str,
        context: str,
        temperature: float = 0.1,
    ) -> RelevanceResult:
        """
        分析单个引用的相关性
        
        Args:
            title: 参考文献标题
            abstract: 参考文献摘要
            citation_anchor: 引用所在句子
            context: 引用上下文
            temperature: 生成温度（越低越确定性）
            
        Returns:
            相关性分析结果
        """
        # 检查输入
        if not title:
            return RelevanceResult(
                title=title,
                abstract=abstract,
                citation_anchor=citation_anchor,
                context=context,
                claim="",
                judgment=RelevanceJudgment.ERROR,
                reason="",
                success=False,
                error_message="参考文献标题为空",
            )
        
        if not abstract:
            return RelevanceResult(
                title=title,
                abstract=abstract,
                citation_anchor=citation_anchor,
                context=context,
                claim="",
                judgment=RelevanceJudgment.UNCLEAR,
                reason="无法分析：参考文献摘要为空",
                success=True,
            )
        
        # 构建提示词
        prompt = self._build_prompt(title, abstract, citation_anchor, context)
        
        # 调用 LLM
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an academic citation verification assistant. Respond in the exact format requested."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=1024,
                )
                
                # 提取响应文本
                raw_response = response.choices[0].message.content
                
                # 解析响应
                claim, judgment, reason = self._parse_response(raw_response)
                
                return RelevanceResult(
                    title=title,
                    abstract=abstract,
                    citation_anchor=citation_anchor,
                    context=context,
                    claim=claim,
                    judgment=judgment,
                    reason=reason,
                    raw_response=raw_response,
                    success=True,
                )
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
        
        # 所有重试都失败
        return RelevanceResult(
            title=title,
            abstract=abstract,
            citation_anchor=citation_anchor,
            context=context,
            claim="",
            judgment=RelevanceJudgment.ERROR,
            reason="",
            raw_response="",
            success=False,
            error_message=f"LLM 调用失败: {str(last_error)}",
        )
    
    def analyze_batch(
        self,
        items: List[Dict[str, str]],
        temperature: float = 0.1,
        progress_callback: Optional[callable] = None,
        max_workers: int = 5,
    ) -> List[RelevanceResult]:
        """
        批量并行分析引用相关性
        
        Args:
            items: 待分析项列表，每项包含:
                - title: 参考文献标题
                - abstract: 参考文献摘要
                - citation_anchor: 引用所在句子
                - context: 引用上下文
            temperature: 生成温度
            progress_callback: 进度回调函数，签名为 callback(current, total)
            max_workers: 最大并行数（默认 5）
            
        Returns:
            相关性分析结果列表（按原始顺序）
        """
        total = len(items)
        if total == 0:
            return []
        
        # 存储结果（按原始索引）
        results = [None] * total
        completed = 0
        
        def analyze_single(idx_item: Tuple[int, Dict[str, str]]) -> Tuple[int, RelevanceResult]:
            """分析单个项目"""
            idx, item = idx_item
            result = self.analyze(
                title=item.get("title", ""),
                abstract=item.get("abstract", ""),
                citation_anchor=item.get("citation_anchor", ""),
                context=item.get("context", ""),
                temperature=temperature,
            )
            return idx, result
        
        # 准备带索引的任务
        indexed_items = list(enumerate(items))
        
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_single, item): item[0] for item in indexed_items}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                    completed += 1
                    
                    logger.debug(f"相关性分析 [{completed}/{total}] 完成: {result.title[:30] if result.title else 'N/A'}...")
                    
                    if progress_callback:
                        progress_callback(completed, total)
                        
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"相关性分析 [{idx}] 失败: {e}")
                    results[idx] = RelevanceResult(
                        title=items[idx].get("title", ""),
                        citation_anchor=items[idx].get("citation_anchor", ""),
                        context=items[idx].get("context", ""),
                        judgment=RelevanceJudgment.ERROR,
                        success=False,
                        error_message=str(e),
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
        
        return results
    
    def analyze_matched_citations(
        self,
        matched_citations: List,
        temperature: float = 0.1,
        progress_callback: Optional[callable] = None,
        max_workers: int = 5,
    ) -> List[RelevanceResult]:
        """
        分析匹配后的引用列表（并行）
        
        接收 CitationMatcher 的输出，进行相关性分析。
        
        Args:
            matched_citations: 匹配的引用列表，每项为 MatchedCitation 或
                [title, authors, year, abstract, pdf_url, citation_anchor, context, match_score]
            temperature: 生成温度
            progress_callback: 进度回调函数
            max_workers: 最大并行数（默认 5）
            
        Returns:
            相关性分析结果列表
        """
        items = []
        
        for citation in matched_citations:
            # 处理列表格式
            if isinstance(citation, (list, tuple)):
                title = citation[0] if len(citation) > 0 else ""
                abstract = citation[3] if len(citation) > 3 else ""
                citation_anchor = citation[5] if len(citation) > 5 else ""
                context = citation[6] if len(citation) > 6 else ""
            # 处理 MatchedCitation 对象
            elif hasattr(citation, 'title'):
                title = citation.title
                abstract = getattr(citation, 'abstract', "")
                citation_anchor = citation.citation_anchor
                context = citation.context
            else:
                continue
            
            items.append({
                "title": title,
                "abstract": abstract,
                "citation_anchor": citation_anchor,
                "context": context,
            })
        
        return self.analyze_batch(items, temperature, progress_callback, max_workers)


# ================== 便捷函数 ==================

def analyze_relevance(
    title: str,
    abstract: str,
    citation_anchor: str,
    context: str,
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1,
) -> RelevanceResult:
    """
    便捷函数：分析单个引用的相关性
    
    Args:
        title: 参考文献标题
        abstract: 参考文献摘要
        citation_anchor: 引用所在句子
        context: 引用上下文
        model_name: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        temperature: 生成温度
        
    Returns:
        相关性分析结果
    """
    analyzer = RelevanceAnalyzer(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    return analyzer.analyze(title, abstract, citation_anchor, context, temperature)


def analyze_relevance_batch(
    items: List[Dict[str, str]],
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1,
    progress_callback: Optional[callable] = None,
    max_workers: int = 5,
) -> List[RelevanceResult]:
    """
    便捷函数：批量并行分析引用相关性
    
    Args:
        items: 待分析项列表
        model_name: 模型名称
        api_key: API 密钥
        base_url: API 基础 URL
        temperature: 生成温度
        progress_callback: 进度回调函数
        max_workers: 最大并行数（默认 5）
        
    Returns:
        相关性分析结果列表
    """
    analyzer = RelevanceAnalyzer(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    return analyzer.analyze_batch(items, temperature, progress_callback, max_workers)


def generate_relevance_report(results: List[RelevanceResult]) -> str:
    """
    生成相关性分析报告
    
    Args:
        results: 分析结果列表
        
    Returns:
        格式化的报告文本
    """
    lines = []
    lines.append("=" * 70)
    lines.append("引文相关性分析报告")
    lines.append("=" * 70)
    
    # 统计
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    
    judgment_counts = {
        RelevanceJudgment.STRONGLY_SUPPORTS: 0,
        RelevanceJudgment.WEAKLY_SUPPORTS: 0,
        RelevanceJudgment.DOES_NOT_SUPPORT: 0,
        RelevanceJudgment.UNCLEAR: 0,
        RelevanceJudgment.ERROR: 0,
    }
    for r in results:
        judgment_counts[r.judgment] += 1
    
    lines.append(f"\n总计: {total} 条引用")
    lines.append(f"成功分析: {success_count}")
    lines.append(f"分析失败: {total - success_count}")
    lines.append("")
    lines.append("判断分布:")
    lines.append(f"  - 强支持 (Strongly supports): {judgment_counts[RelevanceJudgment.STRONGLY_SUPPORTS]}")
    lines.append(f"  - 弱支持 (Weakly supports): {judgment_counts[RelevanceJudgment.WEAKLY_SUPPORTS]}")
    lines.append(f"  - 不支持 (Does not support): {judgment_counts[RelevanceJudgment.DOES_NOT_SUPPORT]}")
    lines.append(f"  - 不确定 (Unclear): {judgment_counts[RelevanceJudgment.UNCLEAR]}")
    lines.append(f"  - 错误 (Error): {judgment_counts[RelevanceJudgment.ERROR]}")
    
    lines.append("\n" + "-" * 70)
    lines.append("详细结果:")
    lines.append("-" * 70)
    
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r.title[:50]}...")
        lines.append(f"    引用句: {r.citation_anchor[:60]}...")
        lines.append(f"    推断论点: {r.claim[:80]}..." if r.claim else "    推断论点: (无)")
        
        # 判断结果使用不同标记
        judgment_marks = {
            RelevanceJudgment.STRONGLY_SUPPORTS: "✅ 强支持",
            RelevanceJudgment.WEAKLY_SUPPORTS: "🔶 弱支持",
            RelevanceJudgment.DOES_NOT_SUPPORT: "❌ 不支持",
            RelevanceJudgment.UNCLEAR: "❓ 不确定",
            RelevanceJudgment.ERROR: "⚠️ 错误",
        }
        lines.append(f"    判断: {judgment_marks.get(r.judgment, r.judgment.value)}")
        lines.append(f"    理由: {r.reason[:100]}..." if r.reason else "    理由: (无)")
        
        if not r.success:
            lines.append(f"    错误: {r.error_message}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)
