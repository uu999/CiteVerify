# -*- coding: utf-8 -*-
"""
测试搜索策略优化：
1. 两阶段 arXiv 查询（严格 → 宽松）命中率对比
2. Semantic Scholar + Crossref 摘要/PDF 提取率对比
"""
import sys
import os
import re
import time
import urllib.parse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citeverify.checkers.reference_checker import ReferenceChecker

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 测试文献列表 ==========
TEST_REFERENCES = [
    ["Business Process Model and Notation-BPMN", 2015],
    ["Generating Structured Plan Representation of Procedures with LLMs", 2025],
    ["Beyond rule-based named entity recognition and relation extraction for process model generation from natural language text", 2023],
    ["Challenges and opportunities of applying natural language processing in business process management", 2018],
    ["Process extraction from text: benchmarking the state of the art and paving the way for future challenges", 2021],
    ["Gpt-4 technical report", 2023],
    ["Gemini: a family of highly capable multimodal models", 2023],
    ["Welcome to the era of chatgpt et al. the prospects of large language models", 2023],
    ["A universal prompting strategy for extracting process model information from natural language text using large language models", 2024],
    ["Translating Workflow Nets into the Partially Ordered Workflow Language", 2025],
    ["MermaidFlow: Redefining Agentic Workflow Generation via Safety-Constrained Evolutionary Programming", 2025],
    ["Opus: A Prompt Intention Framework for Complex Workflow Generation", 2025],
    ["Just tell me: Prompt engineering in business process management", 2023],
    ["Chit-chat or deep talk: prompt engineering for process mining", 2023],
    ["基于大语言模型的业务流程自动建模方法", 2025],
    ["Instantaneous, Comprehensible, and Fixable Soundness Checking of Realistic BPMN Models", 2024],
    ["Size matters less: how fine-tuned small LLMs excel in BPMN generation", 2025],
    ["Process modeling with large language models", 2024],
    ["FLOW-BENCH: Towards Conversational Generation of Enterprise Workflows", 2025],
    ["A decomposed hybrid approach to business process modeling with llms", 2024],
    ["Conceptual modeling and large language models: impressions from first experiments with ChatGPT", 2023],
    ["Process model generation from natural language text", 2011],
    ["Combining NLP approaches for rule extraction from legal documents", 2016],
    ["Extracting declarative process models from natural language", 2019],
    ["Automatic business process structure discovery using ordered neurons LSTM: a preliminary study", 2020],
    ["Causality extraction based on self-attentive BiLSTM-CRF with transferred embeddings", 2021],
    ["A survey on evaluation of large language models", 2024],
    ["Large language models (LLMs): survey, technical frameworks, and future challenges", 2024],
    ["BPMN Assistant: An LLM-Based Approach to Business Process Modeling", 2025],
    ["Clarify when necessary: Resolving ambiguity through interaction with lms", 2025],
    ["Clarigen: Bridging instruction gaps via interactive clarification in code generation", 2025],
    ["PET: an annotated dataset for process extraction from natural language text tasks", 2022],
]

# ==========================================
#  arXiv 相关工具函数
# ==========================================

# 停用词（与项目中 search_arxiv 相同）
_STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'not', 'no',
    'by', 'with', 'from', 'as', 'all', 'its', 'it', 'do', 'you', 'we',
    'he', 'she', 'they', 'this', 'that', 'these', 'those', 'can', 'will',
    'has', 'have', 'had', 'may', 'might', 'should', 'would', 'could',
    'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'than', 'too', 'very', 'just', 'also', 'only', 'own', 'same', 'so',
}

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _sanitize_arxiv_query(title: str) -> str:
    """清理 arXiv 查询中的特殊字符（与项目中一致）"""
    if not title:
        return ""
    cleaned = re.sub(r'[()[\]{}:"\'\\/&|!^~*?]', ' ', title)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _parse_arxiv_entries(content: str):
    """从 arXiv Atom XML 中解析条目"""
    entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
    results = []
    for entry in entries:
        title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
        if not title_match:
            continue
        found_title = re.sub(r'\s+', ' ', title_match.group(1).strip())

        published_match = re.search(r'<published>(\d{4})-', entry)
        entry_year = int(published_match.group(1)) if published_match else None

        abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
        abstract = re.sub(r'\s+', ' ', abstract_match.group(1).strip()) if abstract_match else None

        pdf_url = None
        id_match = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', entry)
        if id_match:
            pdf_url = f"https://arxiv.org/pdf/{id_match.group(1)}.pdf"

        results.append({
            "title": found_title,
            "year": entry_year,
            "abstract": abstract,
            "pdf_url": pdf_url,
        })
    return results


def _match_entries(checker, query_title, year, entries):
    """从 arXiv 返回条目中匹配最佳结果"""
    target_year = None
    if year:
        try:
            target_year = int(year)
        except ValueError:
            pass

    best_match = None
    best_sim = 0.0
    for e in entries:
        if target_year and e["year"]:
            if abs(e["year"] - target_year) > 1:
                continue
        is_match, sim = checker.titles_match(query_title, e["title"])
        if is_match and sim > best_sim:
            best_match = e
            best_sim = sim
    return best_match, best_sim


# ==========================================
#  阶段一（严格）：与当前项目实现一致
# ==========================================
def arxiv_search_strict(checker, title, year=None, timeout=15):
    """
    严格搜索（当前项目实现）：
    - _sanitize_arxiv_query 清理特殊字符
    - 过滤停用词，取内容词
    - ti:<word1> AND ti:<word2> AND ... 构建查询
    - 本地 year±1 过滤 + titles_match 匹配
    """
    import requests

    sanitized = _sanitize_arxiv_query(title)
    if not sanitized:
        return None

    all_words = sanitized.split()
    content_words = [w for w in all_words if w.lower() not in _STOP_WORDS and len(w) > 1]
    if len(content_words) < 2:
        content_words = [w for w in all_words if len(w) > 1][:8]
    else:
        content_words = content_words[:10]
    if not content_words:
        return None

    query_parts = [f"ti:{urllib.parse.quote(w)}" for w in content_words]
    query_string = "+AND+".join(query_parts)
    url = f"{ARXIV_API_URL}?search_query={query_string}&start=0&max_results=20"

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"arXiv 严格搜索失败: {e}")
        return None

    entries = _parse_arxiv_entries(resp.text)
    if not entries:
        return None

    best, sim = _match_entries(checker, title, year, entries)
    return best


# ==========================================
#  阶段二（宽松）：OR 组合 + 少量关键 token
# ==========================================
def _extract_key_tokens(title: str, max_tokens: int = 6) -> list:
    """
    从标题中提取少量最具辨识度的关键 token：
    - 去除停用词
    - 按长度降序（长词通常更有辨识度）
    - 保留缩写词（全大写，如 LLM, BPMN）不论长度
    - 最多 max_tokens 个
    """
    sanitized = _sanitize_arxiv_query(title)
    if not sanitized:
        return []

    words = sanitized.split()
    # 保留缩写词（全大写 >= 2字符）和非停用词
    acronyms = [w for w in words if w.isupper() and len(w) >= 2]
    content = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2 and not w.isupper()]
    # 按长度降序排（长词更具辨识度）
    content.sort(key=lambda w: len(w), reverse=True)

    # 合并：缩写优先 + 长内容词
    tokens = []
    seen = set()
    for w in acronyms + content:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            tokens.append(w)
        if len(tokens) >= max_tokens:
            break

    return tokens


def _percent_encode_conservative(term: str) -> str:
    """
    保守的 URL percent-encoding：
    - 保留字母、数字、连字符
    - 其他字符均 percent-encode
    """
    return urllib.parse.quote(term, safe='-')


def arxiv_search_loose(checker, title, year=None, timeout=15):
    """
    宽松搜索（阶段二）：
    - 提取少量（≤6个）关键 token
    - 用 all: 前缀 + OR 组合（提高召回率）
    - 同时尝试 ti: 前缀 + OR 组合
    - 对每个 token 做保守 percent-encoding
    - 本地 year±1 过滤 + titles_match 匹配
    """
    import requests

    tokens = _extract_key_tokens(title, max_tokens=6)
    if not tokens:
        return None

    # 策略A: all: 前缀 + OR 组合（搜索所有字段）
    query_parts_all = [f"all:{_percent_encode_conservative(t)}" for t in tokens]
    query_a = "+OR+".join(query_parts_all)
    url_a = f"{ARXIV_API_URL}?search_query={query_a}&start=0&max_results=20"

    best_overall = None
    best_overall_sim = 0.0

    try:
        resp = requests.get(url_a, timeout=timeout)
        resp.raise_for_status()
        entries = _parse_arxiv_entries(resp.text)
        if entries:
            match, sim = _match_entries(checker, title, year, entries)
            if match and sim > best_overall_sim:
                best_overall = match
                best_overall_sim = sim
    except Exception as e:
        logger.debug(f"arXiv 宽松搜索(all:OR) 失败: {e}")

    # 如果策略A已经找到高相似度结果，直接返回
    if best_overall and best_overall_sim >= 0.9:
        return best_overall

    time.sleep(0.5)

    # 策略B: ti: 前缀 + OR 组合（只搜索标题字段，更精准但召回率低）
    query_parts_ti = [f"ti:{_percent_encode_conservative(t)}" for t in tokens]
    query_b = "+OR+".join(query_parts_ti)
    url_b = f"{ARXIV_API_URL}?search_query={query_b}&start=0&max_results=20"

    try:
        resp = requests.get(url_b, timeout=timeout)
        resp.raise_for_status()
        entries = _parse_arxiv_entries(resp.text)
        if entries:
            match, sim = _match_entries(checker, title, year, entries)
            if match and sim > best_overall_sim:
                best_overall = match
                best_overall_sim = sim
    except Exception as e:
        logger.debug(f"arXiv 宽松搜索(ti:OR) 失败: {e}")

    return best_overall


def arxiv_search_two_stage(checker, title, year=None, timeout=15):
    """
    两阶段搜索：严格 → 宽松
    先用严格模式（ti: AND），命中则返回；
    否则降级到宽松模式（all:/ti: OR）。
    """
    result = arxiv_search_strict(checker, title, year, timeout)
    if result:
        return result

    time.sleep(1)  # arXiv 速率限制：间隔 ≥ 3s（保守等待）

    result = arxiv_search_loose(checker, title, year, timeout)
    return result


# ==========================================
#  Crossref 查询
# ==========================================
def search_crossref(title: str, year=None, timeout=15):
    """
    通过 Crossref API 搜索论文元数据。
    
    返回 dict: {title, abstract, pdf_url, doi} 或 None
    """
    import requests

    if not title or not title.strip():
        return None

    params = {
        "query.title": title.strip(),
        "rows": 5,
        "select": "title,abstract,link,DOI",
    }
    headers = {
        "User-Agent": "CiteVerify/1.0 (mailto:citeverify@example.com)"
    }

    try:
        resp = requests.get(
            "https://api.crossref.org/works",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"Crossref 查询失败: {e}")
        return None

    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    checker = ReferenceChecker()

    target_year = None
    if year:
        try:
            target_year = int(year)
        except ValueError:
            pass

    for item in items:
        # Crossref 的 title 是列表
        cr_titles = item.get("title", [])
        cr_title = cr_titles[0] if cr_titles else ""
        if not cr_title:
            continue

        # 年份过滤
        if target_year:
            date_parts = item.get("published-print", item.get("published-online", {}))
            if date_parts:
                parts = date_parts.get("date-parts", [[]])
                if parts and parts[0]:
                    cr_year = parts[0][0]
                    if cr_year and abs(cr_year - target_year) > 1:
                        continue

        is_match, sim = checker.titles_match(title, cr_title)
        if not is_match:
            continue

        # 提取摘要（Crossref 的 abstract 可能是带 XML/HTML 标签的 JATS 格式）
        abstract_raw = item.get("abstract", "")
        abstract = None
        if abstract_raw:
            # 去除 JATS XML 标签
            abstract = re.sub(r'<[^>]+>', '', abstract_raw).strip()
            if not abstract:
                abstract = None

        # 提取 PDF URL
        pdf_url = None
        links = item.get("link", [])
        for link in links:
            content_type = link.get("content-type", "")
            url = link.get("URL", "")
            if "pdf" in content_type.lower() and url:
                pdf_url = url
                break
        # 回退: 使用 DOI 链接
        if not pdf_url:
            doi = item.get("DOI", "")
            if doi:
                pdf_url = f"https://doi.org/{doi}"

        return {
            "title": cr_title,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "doi": item.get("DOI"),
            "similarity": sim,
        }

    return None


# ==========================================
#  测试函数 1：arXiv 两阶段检索命中率对比
# ==========================================
def test_arxiv_two_stage():
    """
    对比当前严格检索方式 vs 两阶段检索方式的命中率。
    命中 = search 函数返回非 None（经过标题匹配确认）
    """
    print("\n" + "=" * 80)
    print("  测试 1：arXiv 检索策略对比 —— 严格模式 vs 两阶段（严格→宽松）")
    print("=" * 80)

    checker = ReferenceChecker(timeout=15)
    total = len(TEST_REFERENCES)

    strict_hits = []
    two_stage_hits = []
    details = []

    for i, (title, year) in enumerate(TEST_REFERENCES):
        year_str = str(year)
        print(f"\n[{i+1}/{total}] {title[:60]}{'...' if len(title)>60 else ''} ({year})")

        # --- 严格模式 ---
        t0 = time.time()
        strict_result = arxiv_search_strict(checker, title, year_str)
        t_strict = time.time() - t0
        strict_ok = strict_result is not None
        strict_hits.append(strict_ok)
        strict_label = f"命中 ({strict_result['title'][:40]}...)" if strict_ok else "未命中"
        print(f"  [严格] {strict_label}  ({t_strict:.1f}s)")

        time.sleep(1.5)  # arXiv 速率限制

        # --- 两阶段模式 ---
        t0 = time.time()
        two_stage_result = arxiv_search_two_stage(checker, title, year_str)
        t_two = time.time() - t0
        two_ok = two_stage_result is not None
        two_stage_hits.append(two_ok)
        two_label = f"命中 ({two_stage_result['title'][:40]}...)" if two_ok else "未命中"
        print(f"  [两阶段] {two_label}  ({t_two:.1f}s)")

        improvement = ""
        if not strict_ok and two_ok:
            improvement = " <<< 宽松阶段新增命中"
        print(f"  {improvement}")

        details.append({
            "title": title,
            "year": year,
            "strict": strict_ok,
            "two_stage": two_ok,
            "gained_by_loose": not strict_ok and two_ok,
        })

        time.sleep(2)  # arXiv 速率限制

    # --- 汇总 ---
    strict_count = sum(strict_hits)
    two_stage_count = sum(two_stage_hits)
    gained = sum(1 for d in details if d["gained_by_loose"])

    print("\n" + "-" * 80)
    print("  汇总统计")
    print("-" * 80)
    print(f"  文献总数:         {total}")
    print(f"  严格模式命中:     {strict_count}/{total}  ({strict_count/total*100:.1f}%)")
    print(f"  两阶段模式命中:   {two_stage_count}/{total}  ({two_stage_count/total*100:.1f}%)")
    print(f"  宽松阶段新增命中: {gained}")
    print()

    # 详细对比
    print("  详细结果:")
    print(f"  {'序号':<4} {'严格':^6} {'两阶段':^6} {'新增':^4}  标题")
    print(f"  {'----':<4} {'------':^6} {'------':^6} {'----':^4}  {'----'}")
    for i, d in enumerate(details):
        s = "Y" if d["strict"] else "-"
        t = "Y" if d["two_stage"] else "-"
        g = "+" if d["gained_by_loose"] else ""
        print(f"  {i+1:<4} {s:^6} {t:^6} {g:^4}  {d['title'][:55]}")

    print()
    return details


# ==========================================
#  测试函数 2：Semantic Scholar + Crossref 摘要/PDF 提取率对比
# ==========================================
def test_semantic_scholar_with_crossref():
    """
    对比 Semantic Scholar 单独使用 vs Semantic Scholar + Crossref 补充的
    摘要提取率和 PDF URL 提取率。
    
    流程：
    - 对每篇文献先用 Semantic Scholar 搜索
    - 如果 abstract 或 pdf_url 缺失，用 Semantic Scholar 返回的 DOI（或标题）查 Crossref 补充
    """
    print("\n" + "=" * 80)
    print("  测试 2：Semantic Scholar + Crossref 补充 —— 摘要/PDF 提取率对比")
    print("=" * 80)

    checker = ReferenceChecker(
        semantic_scholar_api_key="k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV",
        timeout=15,
    )

    total = len(TEST_REFERENCES)

    # 统计
    ss_found = 0
    ss_abstract_ok = 0
    ss_pdf_ok = 0

    combined_found = 0
    combined_abstract_ok = 0
    combined_pdf_ok = 0

    details = []

    for i, (title, year) in enumerate(TEST_REFERENCES):
        year_str = str(year)
        print(f"\n[{i+1}/{total}] {title[:60]}{'...' if len(title)>60 else ''} ({year})")

        # --- Semantic Scholar ---
        ss_result = checker.search_semantic_scholar(title, year_str)
        ss_hit = ss_result is not None
        ss_has_abstract = bool(ss_result and ss_result.get("abstract"))
        ss_has_pdf = bool(ss_result and ss_result.get("pdf_url"))

        if ss_hit:
            ss_found += 1
        if ss_has_abstract:
            ss_abstract_ok += 1
        if ss_has_pdf:
            ss_pdf_ok += 1

        ss_label = "命中" if ss_hit else "未命中"
        ss_abs_label = f"摘要:{'有' if ss_has_abstract else '无'}"
        ss_pdf_label = f"PDF:{'有' if ss_has_pdf else '无'}"
        print(f"  [SS] {ss_label}  {ss_abs_label}  {ss_pdf_label}")

        # --- 加入 Crossref 补充 ---
        final_abstract = ss_result.get("abstract") if ss_result else None
        final_pdf = ss_result.get("pdf_url") if ss_result else None
        cr_supplemented_abstract = False
        cr_supplemented_pdf = False
        cr_result = None

        # 当 SS 未命中或缺少 abstract/pdf 时，查 Crossref
        if not ss_hit or not final_abstract or not final_pdf:
            time.sleep(0.3)
            cr_result = search_crossref(title, year_str)

            if cr_result:
                if not ss_hit:
                    # SS 完全未找到，使用 Crossref 结果
                    final_abstract = cr_result.get("abstract")
                    final_pdf = cr_result.get("pdf_url")
                else:
                    # SS 找到但缺信息，从 Crossref 补充
                    if not final_abstract and cr_result.get("abstract"):
                        final_abstract = cr_result["abstract"]
                        cr_supplemented_abstract = True
                    if not final_pdf and cr_result.get("pdf_url"):
                        final_pdf = cr_result["pdf_url"]
                        cr_supplemented_pdf = True

        combined_hit = ss_hit or (cr_result is not None)
        combined_has_abstract = bool(final_abstract)
        combined_has_pdf = bool(final_pdf)

        if combined_hit:
            combined_found += 1
        if combined_has_abstract:
            combined_abstract_ok += 1
        if combined_has_pdf:
            combined_pdf_ok += 1

        cr_label = ""
        if cr_result:
            parts = []
            if cr_supplemented_abstract:
                parts.append("补充摘要")
            elif not ss_hit and cr_result.get("abstract"):
                parts.append("Crossref提供摘要")
            if cr_supplemented_pdf:
                parts.append("补充PDF")
            elif not ss_hit and cr_result.get("pdf_url"):
                parts.append("Crossref提供PDF")
            if parts:
                cr_label = f"  <<< {', '.join(parts)}"
            else:
                cr_label = "  (Crossref命中但无新增信息)"
        elif not ss_hit:
            cr_label = "  (Crossref也未找到)"

        combined_abs_label = f"摘要:{'有' if combined_has_abstract else '无'}"
        combined_pdf_label = f"PDF:{'有' if combined_has_pdf else '无'}"
        print(f"  [SS+CR] {combined_abs_label}  {combined_pdf_label}{cr_label}")

        details.append({
            "title": title,
            "year": year,
            "ss_hit": ss_hit,
            "ss_abstract": ss_has_abstract,
            "ss_pdf": ss_has_pdf,
            "combined_hit": combined_hit,
            "combined_abstract": combined_has_abstract,
            "combined_pdf": combined_has_pdf,
            "cr_supplement_abstract": cr_supplemented_abstract,
            "cr_supplement_pdf": cr_supplemented_pdf,
        })

        time.sleep(1)  # 速率限制

    # --- 汇总 ---
    print("\n" + "-" * 80)
    print("  汇总统计")
    print("-" * 80)
    print(f"  文献总数:           {total}")
    print()
    print(f"  [仅 Semantic Scholar]")
    print(f"    检索命中:         {ss_found}/{total}  ({ss_found/total*100:.1f}%)")
    print(f"    有摘要:           {ss_abstract_ok}/{total}  ({ss_abstract_ok/total*100:.1f}%)")
    print(f"    有 PDF URL:       {ss_pdf_ok}/{total}  ({ss_pdf_ok/total*100:.1f}%)")
    print()
    print(f"  [Semantic Scholar + Crossref]")
    print(f"    检索命中:         {combined_found}/{total}  ({combined_found/total*100:.1f}%)")
    print(f"    有摘要:           {combined_abstract_ok}/{total}  ({combined_abstract_ok/total*100:.1f}%)")
    print(f"    有 PDF URL:       {combined_pdf_ok}/{total}  ({combined_pdf_ok/total*100:.1f}%)")
    print()

    gained_abs = combined_abstract_ok - ss_abstract_ok
    gained_pdf = combined_pdf_ok - ss_pdf_ok
    print(f"  Crossref 新增摘要:  +{gained_abs}")
    print(f"  Crossref 新增 PDF:  +{gained_pdf}")
    print()

    # 详细对比
    print("  详细结果:")
    print(f"  {'序号':<4} {'SS命中':^6} {'SS摘要':^6} {'SS_PDF':^6} | {'合并摘要':^6} {'合并PDF':^6} {'CR补充':^10}  标题")
    print(f"  {'----':<4} {'------':^6} {'------':^6} {'------':^6} | {'------':^6} {'------':^6} {'------':^10}  {'----'}")
    for i, d in enumerate(details):
        sh = "Y" if d["ss_hit"] else "-"
        sa = "Y" if d["ss_abstract"] else "-"
        sp = "Y" if d["ss_pdf"] else "-"
        ca = "Y" if d["combined_abstract"] else "-"
        cp = "Y" if d["combined_pdf"] else "-"
        cr = ""
        if d["cr_supplement_abstract"]:
            cr += "+abs "
        if d["cr_supplement_pdf"]:
            cr += "+pdf"
        if not d["ss_hit"] and d["combined_hit"]:
            cr += "+hit"
        print(f"  {i+1:<4} {sh:^6} {sa:^6} {sp:^6} | {ca:^6} {cp:^6} {cr:^10}  {d['title'][:50]}")

    print()
    return details


# ==========================================
#  主函数
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="搜索策略对比测试")
    parser.add_argument("--test", choices=["arxiv", "crossref", "all"], default="all",
                        help="运行哪个测试: arxiv / crossref / all")
    args = parser.parse_args()

    if args.test in ("arxiv", "all"):
        test_arxiv_two_stage()

    if args.test in ("crossref", "all"):
        test_semantic_scholar_with_crossref()
