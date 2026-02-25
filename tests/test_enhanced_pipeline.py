# -*- coding: utf-8 -*-
"""
增强检索管线 vs 当前管线 —— 端到端命中率 / 摘要率 / PDF率 对比测试

增强管线核心改进（Step 1-3）:
  Step 1: 标题强规范化（CamelCase拆分、数字/字母边界拆分、标点→空格、Unicode NFC、压缩空格）
  Step 2: 生成等价标题候选（canonical / 去停用词 / 前N关键词 / bigram片段）
  Step 3: arXiv 多候选 all: 查询 + SS/OpenAlex arXiv ID 直取兜底 + Crossref 补充

运行:
  python tests/test_enhanced_pipeline.py
"""
import sys
import os
import re
import time
import unicodedata
import urllib.parse
import logging
from typing import Optional, Dict, Any, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from citeverify.checkers.reference_checker import ReferenceChecker, _is_weak_pdf_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
CROSSREF_API_URL = "https://api.crossref.org/works"

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


# =====================================================================
#  Step 1: 标题强规范化
# =====================================================================
def canonicalize_title(raw: str) -> str:
    """
    标题强规范化（5 步）:
    1. Unicode NFC 正规化
    2. CamelCase 拆分 (e.g. "ChatGPT" → "Chat GPT", "BiLSTM" → "Bi LSTM")
    3. 数字/字母边界拆分 (e.g. "GPT4" → "GPT 4", "Llama3" → "Llama 3")
    4. 标点统一为空格
    5. 多余空格压缩
    """
    if not raw:
        return ""
    s = raw.strip()

    # 1. Unicode NFC
    s = unicodedata.normalize("NFC", s)

    # 2. CamelCase 拆分: 在 小写→大写 / 大写→大写小写 边界插空格
    #    "ChatGPT" → "Chat GPT",  "BiLSTM" → "Bi LSTM"
    #    "ProMoAI" → "Pro Mo AI"
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)          # aA → a A
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)    # AAa → AA a

    # 3. 数字/字母边界拆分
    s = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', s)           # a1 → a 1
    s = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', s)           # 1a → 1 a

    # 4. 标点统一为空格（保留中文字符、字母、数字、连字符）
    s = re.sub(r'[^\w\s\u4e00-\u9fff-]', ' ', s)

    # 5. 多余空格压缩
    s = re.sub(r'\s+', ' ', s).strip()

    return s


# =====================================================================
#  Step 2: 生成等价标题候选
# =====================================================================
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'of', 'for', 'and', 'to', 'in', 'on', 'with',
    'or', 'is', 'are', 'at', 'by', 'from', 'as', 'its', 'it',
    'but', 'not', 'be', 'was', 'were', 'has', 'have', 'had',
    'this', 'that', 'which', 'than', 'how', 'what', 'when', 'where',
})


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

    # 1️⃣ canonical
    variants.append(canon)

    # 2️⃣ 去停用词
    no_stop = " ".join(t for t in tokens if t.lower() not in _STOP_WORDS)
    if no_stop and no_stop != canon:
        variants.append(no_stop)

    # 3️⃣ 前 N 关键词（N=6），适用于长标题
    if len(tokens) > 6:
        variants.append(" ".join(tokens[:6]))

    # 4️⃣ 中间 bigram 片段
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
#  Step 3a: 增强 arXiv 搜索 —— 多候选 all: 查询
# =====================================================================
def _sanitize_for_arxiv(text: str) -> str:
    """arXiv 查询清理：去掉 Lucene 特殊字符，保守 percent-encode"""
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
        results.append({"title": title, "year": year, "abstract": abstract, "pdf_url": pdf_url})
    return results


def enhanced_search_arxiv(
    checker: ReferenceChecker,
    raw_title: str,
    year: Optional[str] = None,
    timeout: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    增强 arXiv 搜索:
    - 对每个标题候选生成 all: 查询（OR 模式，最大召回）
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

    for variant in variants[:3]:  # 最多查 3 个候选，控制 API 调用量
        sanitized = _sanitize_for_arxiv(variant)
        if not sanitized:
            continue

        encoded = urllib.parse.quote(sanitized)
        url = f"{ARXIV_API_URL}?search_query=all:{encoded}&start=0&max_results=15"

        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            entries = _parse_arxiv_entries(resp.text)
            for e in entries:
                key = e["title"].lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    all_entries.append(e)
        except Exception as e:
            logger.debug(f"arXiv 增强搜索候选失败: {e}")

        time.sleep(1)  # arXiv 速率限制

    if not all_entries:
        return None

    # 匹配
    best_match = None
    best_sim = 0.0
    for e in all_entries:
        if target_year and e["year"]:
            if abs(e["year"] - target_year) > 1:
                continue
        is_match, sim = checker.titles_match(raw_title, e["title"])
        if is_match and sim > best_sim:
            best_match = e
            best_sim = sim

    return best_match


# =====================================================================
#  Step 3b: 从 SS / OpenAlex 拿 arXiv ID 直取（杀手级兜底）
# =====================================================================
def get_arxiv_from_external_ids(
    checker: ReferenceChecker,
    raw_title: str,
    year: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    如果 SS 或 OpenAlex 返回了 externalIds.ArXiv，
    直接信任，构建 arXiv abstract URL 获取摘要。
    """
    import requests

    # 先查 Semantic Scholar 获取 arXiv ID
    ss_result = checker.search_semantic_scholar(raw_title, year)
    if not ss_result:
        return None

    # SS 的 best_match 不直接暴露 externalIds，需要重新查一次
    # 但我们可以利用它已经返回的 pdf_url 判断是否含 arxiv
    pdf_url = ss_result.get("pdf_url", "") or ""
    arxiv_id = None

    # 从 pdf_url 提取 arXiv ID
    m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+(?:v\d+)?)', pdf_url)
    if m:
        arxiv_id = m.group(1)
    
    if not arxiv_id:
        return None

    # 用 arXiv ID 直接查 arXiv API 拿 abstract
    url = f"{ARXIV_API_URL}?id_list={arxiv_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        entries = _parse_arxiv_entries(resp.text)
        if entries:
            return entries[0]
    except Exception as e:
        logger.debug(f"arXiv ID 直查失败 ({arxiv_id}): {e}")

    return None


# =====================================================================
#  Step 3c: Crossref 补充
# =====================================================================
def search_crossref(
    title: str,
    year: Optional[str] = None,
    timeout: int = 15,
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
        resp = requests.get(CROSSREF_API_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
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

        is_match, sim = checker.titles_match(title, cr_title)
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

        return {"title": cr_title, "abstract": abstract, "pdf_url": pdf_url, "doi": item.get("DOI")}

    return None


# =====================================================================
#  增强管线：完整 verify_reference
# =====================================================================
def enhanced_verify_reference(
    checker: ReferenceChecker,
    title: str,
    year: Optional[str] = None,
) -> Dict[str, Any]:
    """
    增强检索管线：
    1. 增强 arXiv 搜索（多候选 all: 查询）
    2. Semantic Scholar（当前实现）
    3. OpenAlex（当前实现）
    4. SS/OpenAlex arXiv ID 直取兜底
    5. Crossref 摘要/PDF 补充
    """
    result = {
        "hit": False,
        "source": None,
        "title": None,
        "abstract": None,
        "pdf_url": None,
        "similarity": 0.0,
    }

    all_results = {}

    # 1. 增强 arXiv 搜索
    arxiv_res = enhanced_search_arxiv(checker, title, year)
    if arxiv_res:
        all_results["arxiv"] = arxiv_res

    # 2. Semantic Scholar
    ss_res = checker.search_semantic_scholar(title, year)
    if ss_res:
        all_results["semantic_scholar"] = ss_res

    # 3. OpenAlex
    oa_res = checker.search_openalex(title, year)
    if oa_res:
        all_results["openalex"] = oa_res

    # 4. 如果 arXiv 未直接找到，但 SS 返回了 arXiv PDF URL → arXiv ID 直取
    if "arxiv" not in all_results:
        arxiv_via_id = get_arxiv_from_external_ids(checker, title, year)
        if arxiv_via_id:
            all_results["arxiv_via_id"] = arxiv_via_id

    # 无任何命中
    if not all_results:
        return result

    # 选择主源（优先级）
    priority = ["arxiv", "arxiv_via_id", "semantic_scholar", "openalex"]
    primary_key = None
    for k in priority:
        if k in all_results:
            primary_key = k
            break

    primary = all_results[primary_key]
    result["hit"] = True
    result["source"] = primary_key
    result["title"] = primary.get("title")
    result["abstract"] = primary.get("abstract")
    result["pdf_url"] = primary.get("pdf_url")
    result["similarity"] = primary.get("similarity", 0.0)

    # 从其他源补充
    for k in priority:
        if k == primary_key or k not in all_results:
            continue
        fb = all_results[k]
        if not result["abstract"] and fb.get("abstract"):
            result["abstract"] = fb["abstract"]
        if fb.get("pdf_url"):
            if not result["pdf_url"]:
                result["pdf_url"] = fb["pdf_url"]
            elif _is_weak_pdf_url(result["pdf_url"]) and not _is_weak_pdf_url(fb["pdf_url"]):
                result["pdf_url"] = fb["pdf_url"]
        if result["abstract"] and result["pdf_url"]:
            break

    # 5. Crossref 兜底补充
    if not result["abstract"] or not result["pdf_url"]:
        time.sleep(0.3)
        cr = search_crossref(title, year)
        if cr:
            if not result["hit"]:
                result["hit"] = True
                result["source"] = "crossref"
                result["title"] = cr.get("title")
            if not result["abstract"] and cr.get("abstract"):
                result["abstract"] = cr["abstract"]
            if cr.get("pdf_url"):
                if not result["pdf_url"]:
                    result["pdf_url"] = cr["pdf_url"]
                elif _is_weak_pdf_url(result["pdf_url"]) and not _is_weak_pdf_url(cr["pdf_url"]):
                    result["pdf_url"] = cr["pdf_url"]

    return result


# =====================================================================
#  当前管线（直接调用项目 verify_reference）
# =====================================================================
def current_verify_reference(
    checker: ReferenceChecker,
    title: str,
    year: Optional[str] = None,
) -> Dict[str, Any]:
    """封装当前项目的 verify_reference 为统一输出格式"""
    vr = checker.verify_reference(title, year)
    return {
        "hit": vr.can_get,
        "source": vr.source.value if vr.source else None,
        "title": vr.matched_title,
        "abstract": vr.abstract,
        "pdf_url": vr.pdf_url,
        "similarity": vr.similarity,
    }


# =====================================================================
#  测试函数
# =====================================================================
def test_pipeline_comparison():
    """
    端到端对比：当前检索管线 vs 增强检索管线
    统计每篇文献的 命中、摘要、PDF URL 三项指标
    """
    print("\n" + "=" * 90)
    print("  检索管线对比测试：当前管线 vs 增强管线（强规范化+多候选+arXiv ID兜底+Crossref）")
    print("=" * 90)

    checker = ReferenceChecker(
        semantic_scholar_api_key="k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV",
        timeout=20,
    )

    total = len(TEST_REFERENCES)
    details = []

    for i, (title, year) in enumerate(TEST_REFERENCES):
        year_str = str(year)
        short_title = title[:55] + ('...' if len(title) > 55 else '')
        print(f"\n[{i+1}/{total}] {short_title} ({year})")

        # --- 当前管线 ---
        t0 = time.time()
        cur = current_verify_reference(checker, title, year_str)
        t_cur = time.time() - t0
        c_hit = "Y" if cur["hit"] else "-"
        c_abs = "Y" if cur["abstract"] else "-"
        c_pdf = "Y" if cur["pdf_url"] else "-"
        print(f"  [当前]  命中:{c_hit}  摘要:{c_abs}  PDF:{c_pdf}  来源:{cur['source'] or '-'}  ({t_cur:.1f}s)")

        time.sleep(2)

        # --- 增强管线 ---
        t0 = time.time()
        enh = enhanced_verify_reference(checker, title, year_str)
        t_enh = time.time() - t0
        e_hit = "Y" if enh["hit"] else "-"
        e_abs = "Y" if enh["abstract"] else "-"
        e_pdf = "Y" if enh["pdf_url"] else "-"
        print(f"  [增强]  命中:{e_hit}  摘要:{e_abs}  PDF:{e_pdf}  来源:{enh['source'] or '-'}  ({t_enh:.1f}s)")

        # 增量标记
        gains = []
        if not cur["hit"] and enh["hit"]:
            gains.append("+命中")
        if not cur["abstract"] and enh["abstract"]:
            gains.append("+摘要")
        if not cur["pdf_url"] and enh["pdf_url"]:
            gains.append("+PDF")
        if gains:
            print(f"  <<< 增强新增: {', '.join(gains)}")

        details.append({
            "title": title, "year": year,
            "cur_hit": cur["hit"], "cur_abs": bool(cur["abstract"]), "cur_pdf": bool(cur["pdf_url"]),
            "enh_hit": enh["hit"], "enh_abs": bool(enh["abstract"]), "enh_pdf": bool(enh["pdf_url"]),
            "cur_source": cur["source"], "enh_source": enh["source"],
        })

        time.sleep(2)

    # ========== 汇总 ==========
    cur_hit = sum(1 for d in details if d["cur_hit"])
    cur_abs = sum(1 for d in details if d["cur_abs"])
    cur_pdf = sum(1 for d in details if d["cur_pdf"])
    enh_hit = sum(1 for d in details if d["enh_hit"])
    enh_abs = sum(1 for d in details if d["enh_abs"])
    enh_pdf = sum(1 for d in details if d["enh_pdf"])

    print("\n" + "=" * 90)
    print("  汇总统计")
    print("=" * 90)
    print(f"  文献总数: {total}")
    print()
    print(f"  {'指标':<12} {'当前管线':>12} {'增强管线':>12} {'提升':>8}")
    print(f"  {'-'*12} {'-'*12:>12} {'-'*12:>12} {'-'*8:>8}")
    print(f"  {'检索命中':<12} {cur_hit:>5}/{total} ({cur_hit/total*100:4.1f}%) {enh_hit:>5}/{total} ({enh_hit/total*100:4.1f}%) {'+' + str(enh_hit-cur_hit):>6}")
    print(f"  {'有摘要':<12} {cur_abs:>5}/{total} ({cur_abs/total*100:4.1f}%) {enh_abs:>5}/{total} ({enh_abs/total*100:4.1f}%) {'+' + str(enh_abs-cur_abs):>6}")
    print(f"  {'有PDF URL':<12} {cur_pdf:>5}/{total} ({cur_pdf/total*100:4.1f}%) {enh_pdf:>5}/{total} ({enh_pdf/total*100:4.1f}%) {'+' + str(enh_pdf-cur_pdf):>6}")
    print()

    # 详细表
    print("  详细对比表:")
    header = f"  {'#':<3} {'当前':^15} {'增强':^15} {'增量':^8} 标题"
    print(header)
    print(f"  {'---':<3} {'-'*15:^15} {'-'*15:^15} {'-'*8:^8} {'----'}")
    for i, d in enumerate(details):
        c = f"{'Y' if d['cur_hit'] else '-'}/{'Y' if d['cur_abs'] else '-'}/{'Y' if d['cur_pdf'] else '-'}"
        e = f"{'Y' if d['enh_hit'] else '-'}/{'Y' if d['enh_abs'] else '-'}/{'Y' if d['enh_pdf'] else '-'}"
        g = ""
        if not d['cur_hit'] and d['enh_hit']:
            g += "H"
        if not d['cur_abs'] and d['enh_abs']:
            g += "A"
        if not d['cur_pdf'] and d['enh_pdf']:
            g += "P"
        print(f"  {i+1:<3} {c:^15} {e:^15} {g:^8} {d['title'][:48]}")

    print()
    print("  图例: H=命中, A=摘要, P=PDF  |  列格式: 命中/摘要/PDF")
    print()

    return details


# =====================================================================
#  Step 1 单元测试
# =====================================================================
def test_canonicalize():
    """验证标题强规范化效果"""
    print("\n" + "=" * 70)
    print("  Step 1 单元测试: canonicalize_title")
    print("=" * 70)

    cases = [
        ("ChatGPT", "Chat GPT"),
        ("BiLSTM-CRF", "Bi LSTM-CRF"),
        ("GPT-4oSystem Card", "GPT- 4 o System Card"),
        ("ProMoAI", "Pro Mo AI"),
        ("FLOW-BENCH", "FLOW-BENCH"),
        ("Gpt4TechnicalReport", "Gpt 4 Technical Report"),
        ("基于大语言模型的业务流程", "基于大语言模型的业务流程"),
        ("Process Mining:A Research Agenda", "Process Mining A Research Agenda"),
        ("  Multiple   spaces   here  ", "Multiple spaces here"),
    ]
    all_pass = True
    for raw, expected in cases:
        got = canonicalize_title(raw)
        ok = got == expected
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{mark}] '{raw}' → '{got}'" + (f" (expected '{expected}')" if not ok else ""))

    print()
    return all_pass


def test_generate_variants():
    """验证等价标题候选生成"""
    print("\n" + "=" * 70)
    print("  Step 2 单元测试: generate_title_variants")
    print("=" * 70)

    titles = [
        "PET: an annotated dataset for process extraction from natural language text tasks",
        "BiLSTM-CRF",
        "A survey on evaluation of large language models",
        "基于大语言模型的业务流程自动建模方法",
    ]
    for t in titles:
        variants = generate_title_variants(t)
        print(f"\n  原始: {t}")
        for j, v in enumerate(variants):
            print(f"    [{j+1}] {v}")

    print()


# =====================================================================
#  完整流程对比测试：PDF URL → MD → 提取参考文献 → 真伪性校验
# =====================================================================
def test_full_pipeline_comparison(
    pdf_url: str,
    listing_style: str = "numbered",
    citation_format: str = "ieee",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    semantic_scholar_api_key: str = "k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV",
):
    """
    完整流程对比测试：从 PDF URL 开始，经过 MD 转换、参考文献提取、真伪性校验，
    对比当前管线 vs 增强管线的检索效果。

    Args:
        pdf_url: 论文 PDF 的 URL 或本地路径
        listing_style: 条目列举方式 "numbered" 或 "author_year"
        citation_format: 引用格式 "apa"/"mla"/"ieee"/"gb_t_7714"/"chicago"/"harvard"/"vancouver"
        llm_model: LLM 模型名称（用于参考文献提取 fallback）
        llm_api_key: LLM API Key
        llm_base_url: LLM API Base URL
        semantic_scholar_api_key: Semantic Scholar API Key
    """
    from citeverify.converter import PDFConverter
    from citeverify.extractor import extract_references
    from citeverify.models import YaYiDocParserConfig

    print("\n" + "=" * 90)
    print("  完整流程对比测试: PDF URL → MD → 提取参考文献 → 真伪性校验")
    print("  当前管线 vs 增强管线（强规范化+多候选+arXiv ID兜底+Crossref）")
    print("=" * 90)
    print(f"  PDF URL:      {pdf_url}")
    print(f"  listing_style: {listing_style}")
    print(f"  citation_format: {citation_format}")
    print(f"  LLM model:    {llm_model or '(未配置)'}")
    print(f"  LLM fallback: {'启用' if llm_api_key else '禁用'}")
    print("-" * 90)

    # ================ Step 1: PDF → Markdown ================
    print("\n[Step 1] PDF 转换为 Markdown ...")
    step_start = time.time()

    converter = PDFConverter(yayi_config=YaYiDocParserConfig())
    try:
        conversion_result = converter.convert(pdf_url, yayi_config=YaYiDocParserConfig())
    except Exception as e:
        print(f"  ❌ PDF 转换失败: {e}")
        return None

    conversion_time = time.time() - step_start
    print(f"  ✅ 转换完成 ({conversion_time:.1f}s)")
    print(f"     全文长度: {len(conversion_result.full_markdown)} 字符")
    print(f"     正文长度: {len(conversion_result.main_text)} 字符")
    print(f"     参考文献部分长度: {len(conversion_result.references_text)} 字符")
    print(f"     分离成功: {conversion_result.separation_success}")

    if not conversion_result.separation_success or not conversion_result.references_text:
        print("  ❌ 无法分离出参考文献部分，终止测试")
        return None

    # ================ Step 2: 提取参考文献 ================
    print(f"\n[Step 2] 从 Markdown 中提取参考文献 (listing_style={listing_style}, format={citation_format}) ...")
    step_start = time.time()

    references = extract_references(
        conversion_result.references_text,
        listing_style=listing_style,
        citation_format=citation_format,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        use_llm_fallback=bool(llm_api_key),
    )

    extraction_time = time.time() - step_start
    is_numbered = listing_style == "numbered"

    print(f"  ✅ 提取完成 ({extraction_time:.1f}s)")
    print(f"     参考文献条数: {len(references)}")

    if not references:
        print("  ❌ 未提取到参考文献，终止测试")
        return None

    # 打印提取结果概览
    print("\n  提取结果概览 (前10条):")
    for i, ref in enumerate(references[:10]):
        if is_numbered:
            num, full, title, authors, year = ref[0], ref[1], ref[2], ref[3], ref[4]
            title_display = title[:60] if title else "(无标题)"
            print(f"    [{num}] {title_display}  | 作者: {(authors or '-')[:30]}  | 年份: {year or '-'}")
        else:
            full, title, authors, year = ref[0], ref[1], ref[2], ref[3]
            title_display = title[:60] if title else "(无标题)"
            print(f"    [{i+1}] {title_display}  | 作者: {(authors or '-')[:30]}  | 年份: {year or '-'}")

    if len(references) > 10:
        print(f"    ... 还有 {len(references) - 10} 条")

    # ================ Step 3: 真伪性校验对比 ================
    print(f"\n[Step 3] 真伪性校验对比: 当前管线 vs 增强管线 ...")
    print(f"         共 {len(references)} 篇参考文献待校验\n")

    checker = ReferenceChecker(
        semantic_scholar_api_key=semantic_scholar_api_key,
        timeout=20,
    )

    total = len(references)
    details = []

    for i, ref in enumerate(references):
        # 提取标题和年份
        if is_numbered:
            number, full_content, title, authors, year = ref[0], ref[1], ref[2], ref[3], ref[4]
        else:
            full_content, title, authors, year = ref[0], ref[1], ref[2], ref[3]
            number = i + 1

        # 跳过无标题的文献
        if not title or not title.strip():
            short = full_content[:55] if full_content else "(空)"
            print(f"  [{i+1}/{total}] ⚠️ 无标题，跳过: {short}...")
            details.append({
                "number": number, "title": full_content[:60] if full_content else "",
                "year": year, "authors": authors,
                "cur_hit": False, "cur_abs": False, "cur_pdf": False,
                "enh_hit": False, "enh_abs": False, "enh_pdf": False,
                "cur_source": None, "enh_source": None,
                "skipped": True, "skip_reason": "无标题",
            })
            continue

        year_str = str(year) if year else None
        short_title = title[:55] + ('...' if len(title) > 55 else '')
        print(f"  [{i+1}/{total}] {short_title} ({year or '?'})")

        # --- 当前管线 ---
        t0 = time.time()
        cur = current_verify_reference(checker, title, year_str)
        t_cur = time.time() - t0
        c_hit = "Y" if cur["hit"] else "-"
        c_abs = "Y" if cur["abstract"] else "-"
        c_pdf = "Y" if cur["pdf_url"] else "-"
        print(f"    [当前]  命中:{c_hit}  摘要:{c_abs}  PDF:{c_pdf}  来源:{cur['source'] or '-'}  ({t_cur:.1f}s)")

        time.sleep(2)

        # --- 增强管线 ---
        t0 = time.time()
        enh = enhanced_verify_reference(checker, title, year_str)
        t_enh = time.time() - t0
        e_hit = "Y" if enh["hit"] else "-"
        e_abs = "Y" if enh["abstract"] else "-"
        e_pdf = "Y" if enh["pdf_url"] else "-"
        print(f"    [增强]  命中:{e_hit}  摘要:{e_abs}  PDF:{e_pdf}  来源:{enh['source'] or '-'}  ({t_enh:.1f}s)")

        # 增量标记
        gains = []
        if not cur["hit"] and enh["hit"]:
            gains.append("+命中")
        if not cur["abstract"] and enh["abstract"]:
            gains.append("+摘要")
        if not cur["pdf_url"] and enh["pdf_url"]:
            gains.append("+PDF")
        if gains:
            print(f"    <<< 增强新增: {', '.join(gains)}")

        details.append({
            "number": number, "title": title, "year": year, "authors": authors,
            "cur_hit": cur["hit"], "cur_abs": bool(cur["abstract"]), "cur_pdf": bool(cur["pdf_url"]),
            "enh_hit": enh["hit"], "enh_abs": bool(enh["abstract"]), "enh_pdf": bool(enh["pdf_url"]),
            "cur_source": cur["source"], "enh_source": enh["source"],
            "skipped": False,
        })

        time.sleep(2)

    # ================ 汇总统计 ================
    # 排除跳过的
    valid_details = [d for d in details if not d.get("skipped")]
    skipped_count = len(details) - len(valid_details)
    valid_total = len(valid_details)

    cur_hit = sum(1 for d in valid_details if d["cur_hit"])
    cur_abs = sum(1 for d in valid_details if d["cur_abs"])
    cur_pdf = sum(1 for d in valid_details if d["cur_pdf"])
    enh_hit = sum(1 for d in valid_details if d["enh_hit"])
    enh_abs = sum(1 for d in valid_details if d["enh_abs"])
    enh_pdf = sum(1 for d in valid_details if d["enh_pdf"])

    print("\n" + "=" * 90)
    print("  汇总统计")
    print("=" * 90)
    print(f"  参考文献总数: {total}")
    print(f"  有效校验数:  {valid_total}  (跳过: {skipped_count} 篇无标题)")
    print()

    if valid_total > 0:
        print(f"  {'指标':<12} {'当前管线':>16} {'增强管线':>16} {'提升':>8}")
        print(f"  {'-'*12} {'-'*16:>16} {'-'*16:>16} {'-'*8:>8}")
        print(f"  {'检索命中':<12} {cur_hit:>5}/{valid_total} ({cur_hit/valid_total*100:5.1f}%) {enh_hit:>5}/{valid_total} ({enh_hit/valid_total*100:5.1f}%) {'+' + str(enh_hit-cur_hit):>6}")
        print(f"  {'有摘要':<12} {cur_abs:>5}/{valid_total} ({cur_abs/valid_total*100:5.1f}%) {enh_abs:>5}/{valid_total} ({enh_abs/valid_total*100:5.1f}%) {'+' + str(enh_abs-cur_abs):>6}")
        print(f"  {'有PDF URL':<12} {cur_pdf:>5}/{valid_total} ({cur_pdf/valid_total*100:5.1f}%) {enh_pdf:>5}/{valid_total} ({enh_pdf/valid_total*100:5.1f}%) {'+' + str(enh_pdf-cur_pdf):>6}")
    print()

    # 详细表
    print("  详细对比表:")
    header = f"  {'#':<4} {'当前':^15} {'增强':^15} {'增量':^8} 标题"
    print(header)
    print(f"  {'----':<4} {'-'*15:^15} {'-'*15:^15} {'-'*8:^8} {'----'}")
    for d in details:
        num_str = str(d["number"]) if d.get("number") else "?"
        if d.get("skipped"):
            print(f"  {num_str:<4} {'SKIP':^15} {'SKIP':^15} {'':^8} {d['title'][:48]} [{d.get('skip_reason', '')}]")
            continue
        c = f"{'Y' if d['cur_hit'] else '-'}/{'Y' if d['cur_abs'] else '-'}/{'Y' if d['cur_pdf'] else '-'}"
        e = f"{'Y' if d['enh_hit'] else '-'}/{'Y' if d['enh_abs'] else '-'}/{'Y' if d['enh_pdf'] else '-'}"
        g = ""
        if not d['cur_hit'] and d['enh_hit']:
            g += "H"
        if not d['cur_abs'] and d['enh_abs']:
            g += "A"
        if not d['cur_pdf'] and d['enh_pdf']:
            g += "P"
        print(f"  {num_str:<4} {c:^15} {e:^15} {g:^8} {d['title'][:48]}")

    print()
    print("  图例: H=命中, A=摘要, P=PDF  |  列格式: 命中/摘要/PDF  |  SKIP=跳过(无标题)")
    print()

    return details


# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="增强检索管线 vs 当前管线 —— 对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅运行单元测试
  python tests/test_enhanced_pipeline.py --test unit

  # 使用硬编码参考文献列表进行管线对比测试
  python tests/test_enhanced_pipeline.py --test pipeline

  # 完整流程测试: 从 PDF URL 开始
  python tests/test_enhanced_pipeline.py --test full --pdf-url "https://arxiv.org/pdf/2501.12345.pdf" -l numbered -c ieee

  # 完整流程测试 + LLM 参考文献提取 fallback
  python tests/test_enhanced_pipeline.py --test full --pdf-url "https://arxiv.org/pdf/2501.12345.pdf" -l numbered -c ieee --llm-model gpt-4o-mini --llm-api-key YOUR_KEY --llm-base-url https://api.openai.com/v1
        """,
    )
    parser.add_argument("--test", choices=["unit", "pipeline", "full", "all"], default="all",
                        help="选择测试类型: unit=单元测试, pipeline=硬编码列表对比, full=完整流程对比, all=unit+pipeline")
    # 完整流程参数
    parser.add_argument("--pdf-url", type=str, default=None,
                        help="PDF 文件 URL 或本地路径 (--test full 时必须)")
    parser.add_argument("-l", "--listing-style", type=str, default="numbered",
                        choices=["numbered", "author_year"],
                        help="参考文献条目列举方式 (默认: numbered)")
    parser.add_argument("-c", "--citation-format", type=str, default="ieee",
                        choices=["apa", "mla", "ieee", "gb_t_7714", "chicago", "harvard", "vancouver"],
                        help="参考文献引用格式 (默认: ieee)")
    # LLM 配置
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM 模型名称 (用于参考文献提取 fallback)")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="LLM API Key")
    parser.add_argument("--llm-base-url", type=str, default=None,
                        help="LLM API Base URL")
    # Semantic Scholar
    parser.add_argument("--ss-key", type=str, default="k2PL5V0UK25YRzHgin5F18R06qyyuH662LOxywaV",
                        help="Semantic Scholar API Key")

    args = parser.parse_args()

    if args.test in ("unit", "all"):
        test_canonicalize()
        test_generate_variants()

    if args.test in ("pipeline", "all"):
        test_pipeline_comparison()

    if args.test == "full":
        if not args.pdf_url:
            parser.error("--test full 需要提供 --pdf-url 参数")
        test_full_pipeline_comparison(
            pdf_url=args.pdf_url,
            listing_style=args.listing_style,
            citation_format=args.citation_format,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_base_url=args.llm_base_url,
            semantic_scholar_api_key=args.ss_key,
        )
