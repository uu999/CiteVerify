# -*- coding: utf-8 -*-
"""
参考文献真伪校验器

根据参考文献标题搜索论文，验证其真实性并获取摘要和 PDF 链接。
搜索优先级：arXiv -> Semantic Scholar -> OpenAlex
"""
import re
import time
import logging
import urllib.parse
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SearchSource(Enum):
    """搜索来源"""
    ARXIV = "arxiv"
    ARXIV_VIA_ID = "arxiv_via_id"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    NOT_FOUND = "not_found"


@dataclass
class VerificationResult:
    """
    单条参考文献校验结果
    
    Attributes:
        can_get: 是否能搜索到
        abstract: 论文摘要
        pdf_url: 论文 PDF 链接
        source: 搜索来源（arxiv/semantic_scholar/not_found）
        matched_title: 搜索到的匹配标题
        similarity: 标题相似度（0-1）
        error: 错误信息（如有）
    """
    can_get: bool = False
    abstract: Optional[str] = None
    pdf_url: Optional[str] = None
    source: SearchSource = SearchSource.NOT_FOUND
    matched_title: Optional[str] = None
    similarity: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "can_get": self.can_get,
            "abstract": self.abstract,
            "pdf_url": self.pdf_url,
            "source": self.source.value,
            "matched_title": self.matched_title,
            "similarity": self.similarity,
            "error": self.error,
        }


def _is_weak_pdf_url(url: str) -> bool:
    """
    判断一个 pdf_url 是否是"弱"链接（非直接 PDF 下载地址）
    
    弱链接包括：doi.org 页面、semanticscholar 页面等
    强链接包括：arxiv.org/pdf/、直接 .pdf 结尾的 URL
    """
    if not url:
        return True
    url_lower = url.lower()
    # doi.org 通常是落地页
    if "doi.org/" in url_lower:
        return True
    # Semantic Scholar 查询页面
    if "semanticscholar.org/paper/" in url_lower and "/pdf" not in url_lower:
        return True
    # 含 pdf 的 URL 或 arxiv pdf 链接被认为是强链接
    return False


class ReferenceChecker:
    """
    参考文献校验器
    
    使用 arXiv、Semantic Scholar 和 OpenAlex API 搜索论文，验证参考文献真实性。
    搜索优先级：arXiv -> Semantic Scholar -> OpenAlex
    """
    
    # API 配置
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    OPENALEX_API_URL = "https://api.openalex.org/works"
    
    def __init__(
        self,
        request_delay: float = 1,
        timeout: int = 30,
        semantic_scholar_api_key: Optional[str] = None,
        use_semantic_scholar: bool = True,
        use_openalex: bool = True,
    ):
        """
        初始化校验器
        
        Args:
            request_delay: API 请求间隔（秒）
            timeout: 请求超时时间（秒）
            semantic_scholar_api_key: Semantic Scholar API Key（有 Key 时限流为 1 req/s）
            use_semantic_scholar: 是否使用 Semantic Scholar
            use_openalex: 是否使用 OpenAlex（作为第三搜索源）
        """
        self.request_delay = request_delay
        self.timeout = timeout
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.use_semantic_scholar = use_semantic_scholar
        self.use_openalex = use_openalex
        self._last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """等待以遵守 API 限流"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    @staticmethod
    def clean_title(title: str) -> str:
        """
        清理标题（最小化处理）
        
        只做：
        1. 去除首尾空白和多余符号
        2. 合并多个空格为单个空格
        
        不做：转小写、去除所有标点
        
        Args:
            title: 原始标题
            
        Returns:
            清理后的标题
        """
        if not title:
            return ""
        # 去除首尾空白和常见的引用符号
        title = title.strip().strip('"""\'\'.,;:')
        # 合并多个空格
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    @staticmethod
    def _char_similarity(s1: str, s2: str) -> float:
        """
        计算字符级相似度（去除空格后的字符重叠率）
        
        用于处理 OCR 导致的空格丢失问题，如 "ASvstematic" vs "A Systematic"
        """
        if not s1 or not s2:
            return 0.0
        
        # 去除所有空格和标点，只保留字母数字
        import re
        c1 = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s1.lower())
        c2 = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s2.lower())
        
        if not c1 or not c2:
            return 0.0
        
        # 精确匹配
        if c1 == c2:
            return 1.0
        
        # 包含关系
        if c1 in c2 or c2 in c1:
            return min(len(c1), len(c2)) / max(len(c1), len(c2))
        
        # 字符级 Jaccard（使用 n-gram）
        def get_ngrams(s, n=3):
            return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}
        
        ngrams1 = get_ngrams(c1)
        ngrams2 = get_ngrams(c2)
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def titles_match(query_title: str, found_title: str) -> Tuple[bool, float]:
        """
        判断两个标题是否匹配（分层匹配策略）
        
        匹配策略：
        1. 忽略大小写的精确匹配 -> similarity=1.0
        2. 包含关系（处理提取不完整的情况）-> similarity=0.9~0.95
        3. 词级 Jaccard 相似度（阈值 0.65）-> similarity=计算值
        4. 字符级相似度（处理空格丢失）-> similarity=计算值
        
        Args:
            query_title: 查询标题（从参考文献提取的，已清理）
            found_title: 搜索到的标题
            
        Returns:
            (是否匹配, 相似度)
        """
        if not query_title or not found_title:
            return False, 0.0
        
        # 查询标题已在提取时清理，只需处理搜索结果标题
        q = query_title.strip()
        f = ReferenceChecker.clean_title(found_title)
        
        if not q or not f:
            return False, 0.0
        
        # 小写化用于比较（不修改原始标题）
        q_lower = q.lower()
        f_lower = f.lower()
        
        # 第一层：忽略大小写的精确匹配
        if q_lower == f_lower:
            return True, 1.0
        
        # 第二层：包含关系（处理标题提取不完整的情况）
        if q_lower in f_lower or f_lower in q_lower:
            ratio = min(len(q), len(f)) / max(len(q), len(f))
            if ratio >= 0.5:
                return True, 0.9 + ratio * 0.05
        
        # 第三层：词级 Jaccard 相似度
        words_q = set(q_lower.split())
        words_f = set(f_lower.split())
        
        if words_q and words_f:
            intersection = len(words_q & words_f)
            union = len(words_q | words_f)
            jaccard = intersection / union if union > 0 else 0.0
            
            if jaccard >= 0.65:
                return True, jaccard
        
        # 第四层：字符级相似度（处理空格丢失问题）
        char_sim = ReferenceChecker._char_similarity(q, f)
        if char_sim >= 0.85:
            return True, char_sim
        
        # 返回最高的相似度供参考
        return False, max(jaccard if words_q and words_f else 0.0, char_sim)
    
    @staticmethod
    def _sanitize_arxiv_query(title: str) -> str:
        """
        清理 arXiv 查询字符串
        
        arXiv 查询语法中的特殊字符需要去除或转义：
        - 括号 () [] {} 是分组和字段查询语法
        - 冒号 : 是字段前缀
        - 引号 " ' 是精确匹配
        - 其他特殊符号
        
        Args:
            title: 原始标题
            
        Returns:
            清理后的查询字符串
        """
        if not title:
            return ""
        # 去除 arXiv 查询语法特殊字符
        # 保留字母、数字、空格和中文
        cleaned = re.sub(r'[()[\]{}:"\'\\/&|!^~*?]', ' ', title)
        # 合并多个空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def search_arxiv(self, title: str, year: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        在 arXiv 搜索论文
        
        Args:
            title: 论文标题
            year: 发表年份（可选，用于本地过滤）
            
        Returns:
            搜索结果字典，包含 title, abstract, pdf_url；未找到返回 None
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests 未安装，请运行: pip install requests")
        
        self._wait_for_rate_limit()
        
        # 清理并构建查询
        sanitized_title = self._sanitize_arxiv_query(title)
        if not sanitized_title:
            return None
        
        # 使用 ti: 前缀逐词搜索标题字段
        # 必须过滤停用词，因为 arXiv（Lucene）会忽略 ti:is, ti:of 等查询
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
        
        all_words = sanitized_title.split()
        content_words = [w for w in all_words if w.lower() not in _STOP_WORDS and len(w) > 1]
        
        # 如果过滤后太少词，回退使用原始词
        if len(content_words) < 2:
            content_words = [w for w in all_words if len(w) > 1][:8]
        else:
            content_words = content_words[:10]  # 限制最多10个内容词
        
        if not content_words:
            return None
        
        # 构建查询：每个内容词都用 ti: 前缀，用 AND 连接
        query_parts = [f"ti:{urllib.parse.quote(w)}" for w in content_words]
        query_string = "+AND+".join(query_parts)
        url = f"{self.ARXIV_API_URL}?search_query={query_string}&start=0&max_results=20"
        
        logger.debug(f"arXiv 查询 URL: {url}")
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"arXiv API 请求失败: {e}")
            return None
        
        # 解析 Atom XML 响应
        content = response.text
        
        # 提取所有条目
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        
        if not entries:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        # 解析目标年份（用于本地过滤）
        target_year = None
        if year:
            try:
                target_year = int(year)
            except ValueError:
                pass
        
        for entry in entries:
            # 提取标题
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            if not title_match:
                continue
            found_title = title_match.group(1).strip()
            # 清理标题中的换行
            found_title = re.sub(r'\s+', ' ', found_title)
            
            # 本地年份过滤（如果提供了年份）
            if target_year:
                # 从 arXiv 条目中提取发表日期
                published_match = re.search(r'<published>(\d{4})-', entry)
                if published_match:
                    entry_year = int(published_match.group(1))
                    # 允许 ±1 年的误差
                    if abs(entry_year - target_year) > 1:
                        continue
            
            # 判断是否匹配
            is_match, similarity = self.titles_match(title, found_title)
            
            if is_match and similarity > best_similarity:
                # 提取摘要
                abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                abstract = abstract_match.group(1).strip() if abstract_match else None
                if abstract:
                    abstract = re.sub(r'\s+', ' ', abstract)
                
                # 提取 PDF 链接
                pdf_url = None
                # arXiv 的 PDF 链接格式
                id_match = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', entry)
                if id_match:
                    arxiv_id = id_match.group(1)
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                best_match = {
                    "title": found_title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "similarity": similarity,
                }
                best_similarity = similarity
        
        return best_match
    
    def search_semantic_scholar(self, title: str, year: Optional[str] = None, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """
        在 Semantic Scholar 搜索论文
        
        Args:
            title: 论文标题（已清理，直接用于查询）
            year: 发表年份（可选，用于精确过滤）
            max_retries: 遇到 429 限流时的最大重试次数
            
        Returns:
            搜索结果字典，包含 title, abstract, pdf_url；未找到返回 None
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests 未安装，请运行: pip install requests")
        
        if not title or not title.strip():
            return None
        
        # 构建查询（使用引号进行精确标题匹配）
        params = {
            "query": f'"{title.strip()}"',
            "limit": 20,
            "fields": "title,abstract,openAccessPdf,year,externalIds,url",
        }
        
        # 添加年份过滤（如果提供，使用 year±1 范围）
        if year:
            try:
                y = int(year)
                # Semantic Scholar 支持年份范围：year-year
                params["year"] = f"{y-1}-{y+1}"
            except ValueError:
                pass  # 年份格式不正确，跳过过滤
        
        # 构建请求头（如果有 API Key）
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key
        
        retry_count = 0
        while retry_count <= max_retries:
            self._wait_for_rate_limit()
            
            try:
                response = requests.get(
                    self.SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    headers=headers if headers else None,
                    timeout=self.timeout,
                )
                
                # 处理 429 限流
                if response.status_code == 429:
                    retry_count += 1
                    if retry_count <= max_retries:
                        # 指数退避：3s, 9s, 27s...
                        wait_time = 3 ** retry_count
                        print(f"Semantic Scholar 限流，等待 {wait_time}s 后重试 ({retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Semantic Scholar 限流，已达最大重试次数，跳过")
                        return None
                
                response.raise_for_status()
                break  # 成功，跳出循环
                
            except requests.exceptions.RequestException as e:
                print(f"Semantic Scholar API 请求失败: {e}")
                return None
        else:
            # 循环正常结束（未 break），说明达到最大重试
            return None
        
        try:
            result = response.json()
        except ValueError:
            return None
        
        papers = result.get("data", [])
        if not papers:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for paper in papers:
            found_title = paper.get("title", "")
            if not found_title:
                continue
            
            # 判断是否匹配
            is_match, similarity = self.titles_match(title, found_title)
            
            if is_match and similarity > best_similarity:
                abstract = paper.get("abstract") or None  # 确保空字符串变为 None
                
                # 获取 PDF 链接 - 多种来源尝试（不使用 SS 页面链接作为 pdf_url）
                pdf_url = None
                
                # 1. 优先使用 openAccessPdf（注意 url 可能为空字符串）
                open_access_pdf = paper.get("openAccessPdf")
                if open_access_pdf and isinstance(open_access_pdf, dict):
                    oa_url = open_access_pdf.get("url", "")
                    if oa_url and oa_url.strip():  # 排除空字符串
                        pdf_url = oa_url.strip()
                
                # 2. 如果没有 openAccessPdf，尝试从 externalIds 构建 arXiv PDF URL
                if not pdf_url:
                    external_ids = paper.get("externalIds")
                    if external_ids and isinstance(external_ids, dict):
                        arxiv_id = external_ids.get("ArXiv")
                        if arxiv_id:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                # 3. 尝试通过 DOI 构建 Sci-Hub / doi.org 链接
                if not pdf_url:
                    external_ids = paper.get("externalIds")
                    if external_ids and isinstance(external_ids, dict):
                        doi = external_ids.get("DOI")
                        if doi:
                            pdf_url = f"https://doi.org/{doi}"
                
                # 注意：不使用 SS 页面 URL 作为 pdf_url，它不是 PDF 下载链接
                
                best_match = {
                    "title": found_title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "similarity": similarity,
                }
                best_similarity = similarity
        
        return best_match
    
    @staticmethod
    def _convert_openalex_abstract(inverted_index: Dict[str, List[int]]) -> str:
        """
        将 OpenAlex 的 inverted index 格式摘要转换为可读文本
        
        OpenAlex 返回的 abstract 是 inverted index 格式：
        {"word1": [0, 5], "word2": [1], ...}
        表示 word1 出现在位置 0 和 5，word2 出现在位置 1
        
        Args:
            inverted_index: OpenAlex 的 abstract_inverted_index
            
        Returns:
            可读的摘要文本
        """
        if not inverted_index:
            return ""
        
        # 构建位置到单词的映射
        position_word = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_word[pos] = word
        
        # 按位置排序并拼接
        if not position_word:
            return ""
        
        max_pos = max(position_word.keys())
        words = [position_word.get(i, "") for i in range(max_pos + 1)]
        
        return " ".join(words)
    
    def search_openalex(self, title: str, year: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        在 OpenAlex 搜索论文
        
        Args:
            title: 论文标题（已清理，直接用于查询）
            year: 发表年份（可选，用于精确过滤）
            
        Returns:
            搜索结果字典，包含 title, abstract, pdf_url；未找到返回 None
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests 未安装，请运行: pip install requests")
        
        self._wait_for_rate_limit()
        
        if not title or not title.strip():
            return None
        
        # 构建查询 - 尝试两种策略：
        # 策略1: display_name.search 过滤器（只搜索标题字段，更精准）
        # 策略2: search 参数（全文搜索，作为回退）
        # 注意：必须清除标题中的冒号、逗号等特殊字符，否则会破坏 OpenAlex 过滤器语法
        # OpenAlex 过滤器中 , 是条件分隔符，: 是键值分隔符
        clean_title = re.sub(r'[:,;"\'\(\)\[\]\{\}\|]', ' ', title.strip())
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        if not clean_title:
            return None
        
        # 添加 mailto 参数进入 polite pool（更高速率限制，更完整数据）
        select_fields = "title,abstract_inverted_index,open_access,best_oa_location,publication_year,ids,primary_location"
        
        # 先尝试 display_name.search（标题字段搜索）
        params = {
            "filter": f"display_name.search:{clean_title}",
            "per_page": 20,
            "sort": "relevance_score:desc",
            "select": select_fields,
            "mailto": "citeverify@example.com",
        }
        
        # 注意：OpenAlex 的年份过滤会显著改变搜索排名，导致精确匹配的结果被排到后面
        # 因此我们不使用年份过滤，而是依靠标题匹配来确保结果质量
        # 年份验证在后续的 titles_match 中隐式处理（如果需要的话）
        
        logger.debug(f"OpenAlex 查询参数: {params}")
        
        # OpenAlex 建议在请求中添加邮箱以获得更好的服务
        headers = {
            "User-Agent": "CiteVerify/1.0 (https://github.com/citeverify; mailto:citeverify@example.com)"
        }
        
        works = None
        
        # 策略1: 使用 display_name.search 过滤器
        try:
            response = requests.get(
                self.OPENALEX_API_URL,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            works = result.get("results", [])
            if works:
                logger.debug(f"OpenAlex display_name.search 返回 {len(works)} 条结果")
        except Exception as e:
            logger.debug(f"OpenAlex display_name.search 失败: {e}")
            works = None
        
        # 策略2: 如果策略1无结果，回退到全文 search 参数
        if not works:
            self._wait_for_rate_limit()
            fallback_params = {
                "search": title.strip(),
                "per_page": 20,
                "select": select_fields,
                "mailto": "citeverify@example.com",
            }
            logger.debug(f"OpenAlex 回退到 search 参数: {title[:40]}...")
            try:
                response = requests.get(
                    self.OPENALEX_API_URL,
                    params=fallback_params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                works = result.get("results", [])
                if works:
                    logger.debug(f"OpenAlex search 回退返回 {len(works)} 条结果")
            except Exception as e:
                logger.debug(f"OpenAlex search 回退也失败: {e}")
                return None
        
        if not works:
            return None
        
        # 解析目标年份（用于本地过滤）
        target_year = None
        if year:
            try:
                target_year = int(year)
            except ValueError:
                pass
        
        best_match = None
        best_similarity = 0.0
        
        for work in works:
            found_title = work.get("title", "")
            if not found_title:
                continue
            
            # 本地年份过滤（如果提供了年份）
            if target_year:
                work_year = work.get("publication_year")
                if work_year:
                    try:
                        work_year_int = int(work_year)
                        # 允许 ±1 年的误差
                        if abs(work_year_int - target_year) > 1:
                            continue
                    except (ValueError, TypeError):
                        pass
            
            # 判断是否匹配
            is_match, similarity = self.titles_match(title, found_title)
            
            if is_match and similarity > best_similarity:
                # 转换 inverted index 格式的摘要
                abstract_inverted = work.get("abstract_inverted_index")
                abstract = self._convert_openalex_abstract(abstract_inverted) if abstract_inverted else None
                # 确保空字符串变为 None
                if abstract and not abstract.strip():
                    abstract = None
                
                # 获取 PDF 链接 - 多种来源尝试
                pdf_url = None
                
                # 1. best_oa_location.pdf_url（直接 PDF 下载链接）
                best_oa = work.get("best_oa_location")
                if best_oa and isinstance(best_oa, dict):
                    oa_pdf = best_oa.get("pdf_url")
                    if oa_pdf and oa_pdf.strip():
                        pdf_url = oa_pdf.strip()
                
                # 2. primary_location.pdf_url
                if not pdf_url:
                    primary_loc = work.get("primary_location")
                    if primary_loc and isinstance(primary_loc, dict):
                        pl_pdf = primary_loc.get("pdf_url")
                        if pl_pdf and pl_pdf.strip():
                            pdf_url = pl_pdf.strip()
                
                # 3. open_access.oa_url
                if not pdf_url:
                    open_access = work.get("open_access")
                    if open_access and isinstance(open_access, dict):
                        oa_url = open_access.get("oa_url")
                        if oa_url and oa_url.strip():
                            pdf_url = oa_url.strip()
                
                # 4. best_oa_location.landing_page_url（落地页，非直接 PDF）
                if not pdf_url:
                    if best_oa and isinstance(best_oa, dict):
                        landing = best_oa.get("landing_page_url")
                        if landing and landing.strip():
                            pdf_url = landing.strip()
                
                # 5. 通过 DOI 构建链接
                if not pdf_url:
                    ids = work.get("ids")
                    if ids and isinstance(ids, dict):
                        doi = ids.get("doi")
                        if doi and doi.strip():
                            pdf_url = doi.strip()  # doi 通常格式为 https://doi.org/...
                
                best_match = {
                    "title": found_title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "similarity": similarity,
                }
                best_similarity = similarity
        
        return best_match
    
    def verify_reference(self, title: str, year: Optional[str] = None) -> VerificationResult:
        """
        校验单条参考文献
        
        同时搜索所有启用的数据源（arXiv, Semantic Scholar, OpenAlex），
        按优先级选择主源，并从其他源补充缺失的 abstract 和 pdf_url。
        
        优先级：arXiv > Semantic Scholar > OpenAlex
        
        Args:
            title: 参考文献标题
            year: 发表年份（可选，用于提高搜索精度）
            
        Returns:
            VerificationResult 对象
        """
        result = VerificationResult()
        
        if not title or not title.strip():
            result.error = "标题为空"
            return result
        
        title = title.strip()
        
        # 搜索所有启用的数据源，收集结果
        all_results = {}  # source_name -> search_result_dict
        
        # 1. 搜索 arXiv
        arxiv_result = self.search_arxiv(title, year=year)
        if arxiv_result:
            all_results["arxiv"] = arxiv_result
            logger.debug(f"arXiv 找到: title={arxiv_result.get('title', '')[:50]}, "
                        f"has_abstract={bool(arxiv_result.get('abstract'))}, "
                        f"has_pdf={bool(arxiv_result.get('pdf_url'))}")
        
        # 2. 搜索 Semantic Scholar
        if self.use_semantic_scholar:
            ss_result = self.search_semantic_scholar(title, year=year)
            if ss_result:
                all_results["semantic_scholar"] = ss_result
                logger.debug(f"Semantic Scholar 找到: title={ss_result.get('title', '')[:50]}, "
                            f"has_abstract={bool(ss_result.get('abstract'))}, "
                            f"has_pdf={bool(ss_result.get('pdf_url'))}")
        
        # 3. 搜索 OpenAlex
        if self.use_openalex:
            openalex_result = self.search_openalex(title, year=year)
            if openalex_result:
                all_results["openalex"] = openalex_result
                logger.debug(f"OpenAlex 找到: title={openalex_result.get('title', '')[:50]}, "
                            f"has_abstract={bool(openalex_result.get('abstract'))}, "
                            f"has_pdf={bool(openalex_result.get('pdf_url'))}")
        
        # 4. 没有任何源找到
        if not all_results:
            result.can_get = False
            result.source = SearchSource.NOT_FOUND
            logger.info(f"所有源均未找到: {title[:50]}...")
            return result
        
        # 5. 按优先级选择主源
        priority_order = ["arxiv", "semantic_scholar", "openalex"]
        source_enum_map = {
            "arxiv": SearchSource.ARXIV,
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
        
        # 6. 从其他源补充缺失的 abstract 和 pdf_url
        for src in priority_order:
            if src == primary_source or src not in all_results:
                continue
            
            fallback = all_results[src]
            
            # 补充 abstract
            if not result.abstract and fallback.get("abstract"):
                result.abstract = fallback["abstract"]
                logger.info(f"从 {src} 补充 abstract: {title[:40]}...")
            
            # 补充 pdf_url
            fallback_pdf = fallback.get("pdf_url")
            if fallback_pdf and fallback_pdf.strip():
                if not result.pdf_url:
                    result.pdf_url = fallback_pdf
                    logger.info(f"从 {src} 补充 pdf_url: {title[:40]}...")
                elif _is_weak_pdf_url(result.pdf_url) and not _is_weak_pdf_url(fallback_pdf):
                    # 当前 pdf_url 不是直接 PDF 链接（如 doi.org 页面），用更好的替换
                    result.pdf_url = fallback_pdf
                    logger.info(f"从 {src} 替换更优 pdf_url: {title[:40]}...")
            
            # 如果都补全了就不用继续
            if result.abstract and result.pdf_url:
                break
        
        logger.info(f"校验完成: {title[:40]}... | source={primary_source} | "
                    f"abstract={'有' if result.abstract else '无'} ({len(result.abstract) if result.abstract else 0}字) | "
                    f"pdf_url={'有' if result.pdf_url else '无'}")
        
        return result
    
    def verify_references(
        self,
        references: List[List],
        has_number: bool = True,
        verbose: bool = True,
        max_workers: int = 5,
        batch_size: int = 5,
        batch_delay: float = 0.3,
    ) -> List[List]:
        """
        批量校验参考文献列表（受控并发）
        
        Args:
            references: 参考文献列表
                - 有编号: [[编号, 全文, 标题, 作者, 年份], ...]
                - 无编号: [[全文, 标题, 作者, 年份], ...]
            has_number: 是否有数字编号
            verbose: 是否打印进度
            max_workers: 最大并发数（默认 5）
            batch_size: 每批处理数量（默认 5）
            batch_delay: 批间延迟秒数（默认 0.3s）
            
        Returns:
            扩展后的参考文献列表，每个元素追加四个字段：
            [原有字段..., can_get, abstract, pdf_url, source]
            source: "arxiv" / "semantic_scholar" / "openalex" / "not_found"
        """
        total = len(references)
        if total == 0:
            return []
        
        # 准备任务数据
        tasks = []
        for i, ref in enumerate(references):
            if has_number:
                title = ref[2] if len(ref) > 2 else ""
                year = ref[4] if len(ref) > 4 else None
            else:
                title = ref[1] if len(ref) > 1 else ""
                year = ref[3] if len(ref) > 3 else None
            tasks.append((i, ref, title, year))
        
        # 存储结果（按原始顺序）
        results = [None] * total
        completed = 0
        
        def verify_single(task):
            """单个校验任务"""
            idx, ref, title, year = task
            verification = self.verify_reference(title, year=year)
            extended_ref = list(ref) + [
                verification.can_get,
                verification.abstract,
                verification.pdf_url,
                verification.source.value,
            ]
            return idx, extended_ref
        
        # 分批处理
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_tasks = tasks[batch_start:batch_end]
            
            if verbose:
                logger.info(f"校验批次 {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size}: "
                           f"第 {batch_start+1}-{batch_end} 篇 (共 {total} 篇)")
            
            # 使用线程池并发处理当前批次
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(verify_single, task): task for task in batch_tasks}
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, extended_ref = future.result()
                        results[idx] = extended_ref
                        completed += 1
                        
                        if verbose:
                            # 打印进度
                            title_preview = extended_ref[2][:30] if has_number and len(extended_ref) > 2 else \
                                          extended_ref[1][:30] if len(extended_ref) > 1 else "未知"
                            can_get = extended_ref[-4]
                            source = extended_ref[-1]
                            status = f"[{source}]" if can_get else "[未找到]"
                            print(f"  [{completed}/{total}] {status} {title_preview}...")
                    except Exception as e:
                        task = futures[future]
                        idx = task[0]
                        ref = task[1]
                        logger.error(f"校验失败: {e}")
                        # 记录失败结果
                        results[idx] = list(ref) + [False, None, None, "error"]
                        completed += 1
            
            # 批间延迟
            if batch_end < total:
                if verbose:
                    logger.debug(f"批间延迟 {batch_delay}s...")
                time.sleep(batch_delay)
        
        if verbose:
            # 统计
            found_count = sum(1 for r in results if r and r[-4])
            arxiv_count = sum(1 for r in results if r and r[-1] == "arxiv")
            ss_count = sum(1 for r in results if r and r[-1] == "semantic_scholar")
            oa_count = sum(1 for r in results if r and r[-1] == "openalex")
            logger.info(f"校验完成: {found_count}/{total} 篇文献可验证 "
                       f"(arXiv: {arxiv_count}, Semantic Scholar: {ss_count}, OpenAlex: {oa_count})")
        
        return results


def verify_references(
    references: List[List],
    has_number: bool = True,
    request_delay: float = 1.0,
    semantic_scholar_api_key: Optional[str] = None,
    use_semantic_scholar: bool = True,
    use_openalex: bool = True,
    verbose: bool = True,
    max_workers: int = 5,
    batch_size: int = 5,
    batch_delay: float = 0.3,
) -> List[List]:
    """
    便捷函数：批量校验参考文献（受控并发）
    
    搜索优先级：arXiv -> Semantic Scholar -> OpenAlex
    
    Args:
        references: 参考文献列表
            - 有编号: [[编号, 全文, 标题, 作者, 年份], ...]
            - 无编号: [[全文, 标题, 作者, 年份], ...]
        has_number: 是否有数字编号
        request_delay: 单个请求内的 API 请求间隔（秒）
        semantic_scholar_api_key: Semantic Scholar API Key
        use_semantic_scholar: 是否使用 Semantic Scholar
        use_openalex: 是否使用 OpenAlex
        verbose: 是否打印进度
        max_workers: 最大并发数（默认 5）
        batch_size: 每批处理数量（默认 5）
        batch_delay: 批间延迟秒数（默认 0.3s）
        
    Returns:
        扩展后的参考文献列表，每个元素追加四个字段：
        [...原有字段, can_get, abstract, pdf_url, source]
        source: "arxiv" / "semantic_scholar" / "openalex" / "not_found"
        
    Example:
        >>> refs = [[1, "...", "Attention is all you need", "Vaswani A", "2017"]]
        >>> results = verify_references(refs, has_number=True)
        >>> for r in results:
        ...     print(f"[{r[0]}] {r[2]}: can_get={r[-4]}, source={r[-1]}")
    """
    checker = ReferenceChecker(
        request_delay=request_delay,
        semantic_scholar_api_key=semantic_scholar_api_key,
        use_semantic_scholar=use_semantic_scholar,
        use_openalex=use_openalex,
    )
    return checker.verify_references(
        references, 
        has_number=has_number, 
        verbose=verbose,
        max_workers=max_workers,
        batch_size=batch_size,
        batch_delay=batch_delay,
    )


def verify_single_reference(
    title: str,
    semantic_scholar_api_key: Optional[str] = None,
) -> VerificationResult:
    """
    便捷函数：校验单条参考文献
    
    Args:
        title: 参考文献标题
        semantic_scholar_api_key: Semantic Scholar API Key
        
    Returns:
        VerificationResult 对象
    """
    checker = ReferenceChecker(semantic_scholar_api_key=semantic_scholar_api_key)
    return checker.verify_reference(title)
