# -*- coding: utf-8 -*-
"""
增强参考文献真伪校验器

在原有 ReferenceChecker 基础上，增加以下改进：
  Step 1: 标题强规范化（CamelCase 拆分、数字/字母边界拆分、标点→空格、Unicode NFC、压缩空格）
  Step 2: 生成等价标题候选（canonical / 去停用词 / 前N关键词 / bigram 片段）
  Step 3a: arXiv 多候选 all: 查询
  Step 3b: 从 SS / OpenAlex 拿 arXiv ID 直取（杀手级兜底）
  Step 3c: Crossref 摘要 / PDF 补充

原有的 ReferenceChecker 保留不变，作为备用方案。
"""
import re
import time
import logging
import unicodedata
import urllib.parse
import concurrent.futures
from typing import Optional, Dict, Any, List, Tuple

from .reference_checker import (
    ReferenceChecker,
    VerificationResult,
    SearchSource,
    _is_weak_pdf_url,
)

logger = logging.getLogger(__name__)

# =====================================================================
#  常量
# =====================================================================
ARXIV_API_URL = "http://export.arxiv.org/api/query"
CROSSREF_API_URL = "https://api.crossref.org/works"

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'of', 'for', 'and', 'to', 'in', 'on', 'with',
    'or', 'is', 'are', 'at', 'by', 'from', 'as', 'its', 'it',
    'but', 'not', 'be', 'was', 'were', 'has', 'have', 'had',
    'this', 'that', 'which', 'than', 'how', 'what', 'when', 'where',
})


# =====================================================================
#  Step 1: 标题强规范化
# =====================================================================
def canonicalize_title(raw: str) -> str:
    """
    标题强规范化（5 步）:
    1. Unicode NFC 正规化
    2. CamelCase 拆分 (e.g. "ChatGPT" → "Chat GPT", "BiLSTM" → "Bi LSTM")
    3. 数字/字母边界拆分 (e.g. "GPT4" → "GPT 4", "Llama3" → "Llama 3")
    4. 标点统一为空格（保留中文字符、字母、数字、连字符）
    5. 多余空格压缩
    """
    if not raw:
        return ""
    s = raw.strip()

    # 1. Unicode NFC
    s = unicodedata.normalize("NFC", s)

    # 2. CamelCase 拆分: 在 小写→大写 / 大写→大写小写 边界插空格
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)

    # 3. 数字/字母边界拆分
    s = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', s)
    s = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', s)

    # 4. 标点统一为空格（保留中文字符、字母、数字、连字符）
    s = re.sub(r'[^\w\s\u4e00-\u9fff-]', ' ', s)

    # 5. 多余空格压缩
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# =====================================================================
#  Step 2: 生成等价标题候选
# =====================================================================
def generate_title_variants(raw_title: str) -> List[str]:
    """
    生成 4 类等价标题候选:
    1. canonical 完整规范化标题
    2. 去停用词版
    3. 前 N 关键词（防长标题溢出）
    4. 中间 bigram 片段（应对 OCR 连写偏移）
    """
    canon = canonicalize_title(raw_title)
    if not canon:
        return []

    tokens = canon.split()
    variants = []

    # 1. canonical
    variants.append(canon)

    # 2. 去停用词
    no_stop = " ".join(t for t in tokens if t.lower() not in _STOP_WORDS)
    if no_stop and no_stop != canon:
        variants.append(no_stop)

    # 3. 前 N 关键词（N=6），适用于长标题
    if len(tokens) > 6:
        variants.append(" ".join(tokens[:6]))

    # 4. 中间 bigram 片段
    if len(tokens) >= 4:
        mid = " ".join(tokens[1:5])
        if mid not in variants:
            variants.append(mid)

    # 去重，保持顺序
    seen = set()
    result = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


# =====================================================================
#  辅助函数
# =====================================================================
def _sanitize_for_arxiv(text: str) -> str:
    """arXiv 查询清理：去掉 Lucene 特殊字符"""
    s = re.sub(r'[()[\]{}:"\'\\/&|!^~*?]', ' ', text)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _parse_arxiv_entries(content: str) -> List[Dict]:
    """从 arXiv Atom XML 解析条目"""
    entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
    results = []
    for entry in entries:
        tm = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
        if not tm:
            continue
        title = re.sub(r'\s+', ' ', tm.group(1).strip())
        pm = re.search(r'<published>(\d{4})-', entry)
        year = int(pm.group(1)) if pm else None
        am = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
        abstract = re.sub(r'\s+', ' ', am.group(1).strip()) if am else None
        pdf_url = None
        im = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', entry)
        if im:
            pdf_url = f"https://arxiv.org/pdf/{im.group(1)}.pdf"
        results.append({
            "title": title,
            "year": year,
            "abstract": abstract,
            "pdf_url": pdf_url,
        })
    return results


# =====================================================================
#  EnhancedReferenceChecker
# =====================================================================
class EnhancedReferenceChecker(ReferenceChecker):
    """
    增强参考文献校验器

    继承 ReferenceChecker 的所有搜索方法（search_arxiv, search_semantic_scholar,
    search_openalex, titles_match 等），重写 verify_reference 使用增强检索管线：

    1. 增强 arXiv 搜索（标题强规范化 + 多候选 all: 查询）
    2. Semantic Scholar（继承原实现）
    3. OpenAlex（继承原实现）
    4. SS/OpenAlex arXiv ID 直取兜底
    5. Crossref 摘要/PDF 补充

    原有的 verify_reference（基于 ti: AND 查询）通过父类 ReferenceChecker 保留。
    """

    # =====================================================================
    #  Step 3a: 增强 arXiv 搜索 —— 多候选 all: 查询
    # =====================================================================
    def enhanced_search_arxiv(
        self,
        raw_title: str,
        year: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        增强 arXiv 搜索:
        - 对每个标题候选生成 all: 查询（最大召回）
        - 所有候选的返回条目合并去重
        - 用 titles_match 对原始标题匹配
        """
        import requests

        variants = generate_title_variants(raw_title)
        if not variants:
            return None

        target_year = None
        if year:
            try:
                target_year = int(year)
            except ValueError:
                pass

        all_entries = []
        seen_titles = set()

        for variant in variants[:3]:  # 最多查 3 个候选
            sanitized = _sanitize_for_arxiv(variant)
            if not sanitized:
                continue

            encoded = urllib.parse.quote(sanitized)
            url = f"{ARXIV_API_URL}?search_query=all:{encoded}&start=0&max_results=15"

            try:
                self._wait_for_rate_limit()
                resp = requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                entries = _parse_arxiv_entries(resp.text)
                for e in entries:
                    key = e["title"].lower().strip()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        all_entries.append(e)
            except Exception as exc:
                logger.debug(f"arXiv 增强搜索候选失败: {exc}")

        if not all_entries:
            return None

        # 匹配
        best_match = None
        best_sim = 0.0
        for e in all_entries:
            if target_year and e["year"]:
                if abs(e["year"] - target_year) > 1:
                    continue
            is_match, sim = self.titles_match(raw_title, e["title"])
            if is_match and sim > best_sim:
                best_match = e
                best_sim = sim

        if best_match:
            best_match["similarity"] = best_sim
        return best_match

    # =====================================================================
    #  Step 3b: 从 SS / OpenAlex 拿 arXiv ID 直取（杀手级兜底）
    # =====================================================================
    def get_arxiv_from_external_ids(
        self,
        raw_title: str,
        year: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        如果 Semantic Scholar 返回了包含 arXiv ID 的 PDF URL，
        直接用该 ID 查 arXiv API 获取摘要。
        """
        import requests

        ss_result = self.search_semantic_scholar(raw_title, year)
        if not ss_result:
            return None

        pdf_url = ss_result.get("pdf_url", "") or ""
        arxiv_id = None
        m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)', pdf_url)
        if m:
            arxiv_id = m.group(1)

        if not arxiv_id:
            return None

        # 用 arXiv ID 直接查 arXiv API
        url = f"{ARXIV_API_URL}?id_list={arxiv_id}"
        try:
            self._wait_for_rate_limit()
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            entries = _parse_arxiv_entries(resp.text)
            if entries:
                entry = entries[0]
                entry["similarity"] = ss_result.get("similarity", 0.0)
                return entry
        except Exception as exc:
            logger.debug(f"arXiv ID 直查失败 ({arxiv_id}): {exc}")

        return None

    # =====================================================================
    #  Step 3c: Crossref 补充
    # =====================================================================
    def search_crossref(
        self,
        title: str,
        year: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """通过 Crossref 搜索论文元数据，返回 abstract + pdf_url"""
        import requests

        if not title or not title.strip():
            return None

        params = {
            "query.title": title.strip(),
            "rows": 5,
            "select": "title,abstract,link,DOI",
        }
        headers = {"User-Agent": "CiteVerify/1.0 (mailto:citeverify@example.com)"}

        try:
            self._wait_for_rate_limit()
            resp = requests.get(
                CROSSREF_API_URL,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        items = data.get("message", {}).get("items", [])
        if not items:
            return None

        target_year = None
        if year:
            try:
                target_year = int(year)
            except ValueError:
                pass

        for item in items:
            cr_titles = item.get("title", [])
            cr_title = cr_titles[0] if cr_titles else ""
            if not cr_title:
                continue

            if target_year:
                dp = item.get("published-print", item.get("published-online", {}))
                if dp:
                    parts = dp.get("date-parts", [[]])
                    if parts and parts[0]:
                        if parts[0][0] and abs(parts[0][0] - target_year) > 1:
                            continue

            is_match, sim = self.titles_match(title, cr_title)
            if not is_match:
                continue

            abstract_raw = item.get("abstract", "")
            abstract = re.sub(r'<[^>]+>', '', abstract_raw).strip() if abstract_raw else None
            if abstract and not abstract.strip():
                abstract = None

            pdf_url = None
            for link in item.get("link", []):
                if "pdf" in link.get("content-type", "").lower() and link.get("URL"):
                    pdf_url = link["URL"]
                    break
            if not pdf_url:
                doi = item.get("DOI", "")
                if doi:
                    pdf_url = f"https://doi.org/{doi}"

            return {
                "title": cr_title,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "similarity": sim,
                "doi": item.get("DOI"),
            }

        return None

    # =====================================================================
    #  重写 verify_reference —— 增强检索管线
    # =====================================================================
    def verify_reference(self, title: str, year: Optional[str] = None) -> VerificationResult:
        """
        增强版参考文献校验（重写父类方法）

        管线流程:
        1. 并行执行：增强 arXiv 搜索 + Semantic Scholar + OpenAlex
        2. arXiv ID 直取兜底（如果 arXiv 未直接找到但 SS 返回了 arXiv PDF URL）
        3. Crossref 补充缺失的 abstract / pdf_url

        优先级: arXiv > arXiv_via_ID > Semantic Scholar > OpenAlex > Crossref

        性能优化: 步骤 1 中三个搜索源并行执行，大幅缩短单篇校验耗时。
        """
        result = VerificationResult()

        if not title or not title.strip():
            result.error = "标题为空"
            return result

        title = title.strip()

        # =========== 阶段 1: 并行搜索 arXiv / SS / OpenAlex ===========
        all_results: Dict[str, Dict[str, Any]] = {}

        def _search_arxiv():
            return "arxiv", self.enhanced_search_arxiv(title, year)

        def _search_ss():
            if self.use_semantic_scholar:
                return "semantic_scholar", self.search_semantic_scholar(title, year)
            return "semantic_scholar", None

        def _search_oa():
            if self.use_openalex:
                return "openalex", self.search_openalex(title, year)
            return "openalex", None

        # 三个搜索源并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(_search_arxiv),
                pool.submit(_search_ss),
                pool.submit(_search_oa),
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    src_name, src_result = f.result()
                    if src_result:
                        all_results[src_name] = src_result
                except Exception as exc:
                    logger.debug(f"并行搜索异常: {exc}")

        # =========== 阶段 2: arXiv ID 直取兜底 ===========
        if "arxiv" not in all_results and self.use_semantic_scholar:
            arxiv_via_id = self.get_arxiv_from_external_ids(title, year)
            if arxiv_via_id:
                all_results["arxiv_via_id"] = arxiv_via_id

        # 无任何命中
        if not all_results:
            result.can_get = False
            result.source = SearchSource.NOT_FOUND
            logger.info(f"所有源均未找到: {title[:50]}...")
            return result

        # 按优先级选择主源
        priority_order = ["arxiv", "arxiv_via_id", "semantic_scholar", "openalex"]
        source_enum_map = {
            "arxiv": SearchSource.ARXIV,
            "arxiv_via_id": SearchSource.ARXIV_VIA_ID,
            "semantic_scholar": SearchSource.SEMANTIC_SCHOLAR,
            "openalex": SearchSource.OPENALEX,
        }

        primary_source = None
        primary_result_dict = None
        for src in priority_order:
            if src in all_results:
                primary_source = src
                primary_result_dict = all_results[src]
                break

        # 填充主源结果
        result.can_get = True
        result.source = source_enum_map[primary_source]
        result.matched_title = primary_result_dict.get("title")
        result.similarity = primary_result_dict.get("similarity", 0.0)
        result.abstract = primary_result_dict.get("abstract")
        result.pdf_url = primary_result_dict.get("pdf_url")

        # 从其他源补充缺失的 abstract 和 pdf_url
        for src in priority_order:
            if src == primary_source or src not in all_results:
                continue

            fallback = all_results[src]

            if not result.abstract and fallback.get("abstract"):
                result.abstract = fallback["abstract"]
                logger.info(f"从 {src} 补充 abstract: {title[:40]}...")

            fallback_pdf = fallback.get("pdf_url")
            if fallback_pdf and fallback_pdf.strip():
                if not result.pdf_url:
                    result.pdf_url = fallback_pdf
                    logger.info(f"从 {src} 补充 pdf_url: {title[:40]}...")
                elif _is_weak_pdf_url(result.pdf_url) and not _is_weak_pdf_url(fallback_pdf):
                    result.pdf_url = fallback_pdf
                    logger.info(f"从 {src} 替换更优 pdf_url: {title[:40]}...")

            if result.abstract and result.pdf_url:
                break

        # =========== 阶段 3: Crossref 兜底补充 ===========
        if not result.abstract or not result.pdf_url:
            cr = self.search_crossref(title, year)
            if cr:
                if not result.can_get:
                    # Crossref 作为唯一命中源
                    result.can_get = True
                    result.source = SearchSource.CROSSREF
                    result.matched_title = cr.get("title")
                    result.similarity = cr.get("similarity", 0.0)
                if not result.abstract and cr.get("abstract"):
                    result.abstract = cr["abstract"]
                    logger.info(f"从 crossref 补充 abstract: {title[:40]}...")
                if cr.get("pdf_url"):
                    if not result.pdf_url:
                        result.pdf_url = cr["pdf_url"]
                    elif _is_weak_pdf_url(result.pdf_url) and not _is_weak_pdf_url(cr["pdf_url"]):
                        result.pdf_url = cr["pdf_url"]

        logger.info(
            f"校验完成: {title[:40]}... | source={result.source.value} | "
            f"abstract={'有' if result.abstract else '无'} "
            f"({len(result.abstract) if result.abstract else 0}字) | "
            f"pdf_url={'有' if result.pdf_url else '无'}"
        )

        return result
