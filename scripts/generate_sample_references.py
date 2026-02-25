# -*- coding: utf-8 -*-
"""
生成各引用格式的示例参考文献 MD 文件

输出结构：
  output/apa/     -> numbered_bracket.md, numbered_dot.md, numbered_paren.md, author_year.md
  output/mla/     -> 同上
  output/ieee/    -> 同上
  ...
每个 MD 文件包含 10 条参考文献，严格按对应引用格式书写。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

CITATION_FORMATS = (
    "apa",
    "mla",
    "ieee",
    "gb_t_7714",
    "chicago",
    "harvard",
    "vancouver",
)

# 10 条示例文献的原始数据（作者、标题、期刊、年份、卷期、页码、DOI 等）
SAMPLE_REFERENCES = [
    {
        "authors": "Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I",
        "title": "Attention is all you need",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2017",
        "volume": "30",
        "pages": "5998-6008",
        "doi": "10.48550/arXiv.1706.03762",
    },
    {
        "authors": "Devlin J, Chang M-W, Lee K, Toutanova K",
        "title": "BERT: Pre-training of deep bidirectional transformers for language understanding",
        "journal": "Proceedings of NAACL-HLT",
        "year": "2019",
        "volume": "1",
        "pages": "4171-4186",
        "doi": "10.18653/v1/N19-1423",
    },
    {
        "authors": "Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A",
        "title": "Language models are few-shot learners",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2020",
        "volume": "33",
        "pages": "1877-1901",
        "doi": "10.48550/arXiv.2005.14165",
    },
    {
        "authors": "Liu Y, Ott M, Goyal N, Du J, Joshi M, Chen D, Levy O, Lewis M, Zettlemoyer L, Stoyanov V",
        "title": "RoBERTa: A robustly optimized BERT pretraining approach",
        "journal": "arXiv preprint",
        "year": "2019",
        "pages": "arXiv:1907.11692",
        "doi": "10.48550/arXiv.1907.11692",
    },
    {
        "authors": "Raffel C, Shazeer N, Roberts A, Lee K, Narang S, Matena M, Zhou Y, Li W, Liu PJ",
        "title": "Exploring the limits of transfer learning with a unified text-to-text transformer",
        "journal": "Journal of Machine Learning Research",
        "year": "2020",
        "volume": "21",
        "pages": "1-67",
        "doi": "10.5555/3455716.3455856",
    },
    {
        "authors": "Lewis M, Liu Y, Goyal N, Ghazvininejad M, Mohamed A, Levy O, Stoyanov V, Zettlemoyer L",
        "title": "BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension",
        "journal": "Proceedings of the 58th Annual Meeting of the ACL",
        "year": "2020",
        "pages": "7871-7880",
        "doi": "10.18653/v1/2020.acl-main.703",
    },
    {
        "authors": "Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I",
        "title": "Language models are unsupervised multitask learners",
        "journal": "OpenAI blog",
        "year": "2019",
        "volume": "1",
        "pages": "9",
    },
    {
        "authors": "Touvron H, Lavril T, Izacard G, Martinet X, Lachaux M-A, Lacroix T, Rozière B, Goyal N, Hambro F, Azhar F",
        "title": "LLaMA: Open and efficient foundation language models",
        "journal": "arXiv preprint",
        "year": "2023",
        "pages": "arXiv:2302.13971",
        "doi": "10.48550/arXiv.2302.13971",
    },
    {
        "authors": "Achiam J, Adler S, Agarwal S, Ahmad L, Akkaya I, Aleman FL, Almeida D, Altenschmidt J, Altman S, Anadkat S",
        "title": "GPT-4 technical report",
        "journal": "arXiv preprint",
        "year": "2023",
        "pages": "arXiv:2303.08774",
        "doi": "10.48550/arXiv.2303.08774",
    },
    {
        "authors": "Ouyang L, Wu J, Jiang X, Almeida D, Wainwright C, Mishkin P, Zhang C, Agarwal S, Slama K, Ray A",
        "title": "Training language models to follow instructions with human feedback",
        "journal": "Advances in Neural Information Processing Systems",
        "year": "2022",
        "volume": "35",
        "pages": "27730-27744",
        "doi": "10.48550/arXiv.2203.02155",
    },
]

# 中文示例（用于 GB/T 7714 部分条目）
SAMPLE_REFERENCES_CN = [
    {
        "authors": "李明, 王华, 张伟",
        "title": "基于深度学习的自然语言处理综述",
        "journal": "计算机学报",
        "year": "2022",
        "volume": "45",
        "issue": "3",
        "pages": "542-560",
        "doi": "10.11897/SP.J.1016.2022.00542",
    },
    {
        "authors": "刘强, 陈静",
        "title": "大语言模型在信息检索中的应用",
        "journal": "软件学报",
        "year": "2023",
        "volume": "34",
        "issue": "2",
        "pages": "801-820",
    },
]


def _ieee_author(name: str) -> str:
    """IEEE 作者格式: A. LastName"""
    parts = name.strip().split()
    if not parts:
        return name
    initials = ".".join(p[0] for p in parts[:-1]) + "."
    return f"{initials} {parts[-1]}"


def format_apa(ref: dict, index: int) -> str:
    """APA: Author, A. A., & Author, B. B. (Year). Title. Journal, Volume(Issue), pp–pp."""
    names = [n.strip() for n in ref["authors"].split(",")]
    if len(names) == 1:
        a = names[0].split()
        authors_str = f"{a[-1]}, {' '.join(a[:-1])}" if len(a) >= 2 else names[0]
    elif len(names) == 2:
        a1, a2 = names[0].split(), names[1].split()
        last1 = a1[-1] + ", " + " ".join(a1[:-1]) if len(a1) >= 2 else names[0]
        last2 = a2[-1] + " " + " ".join(a2[:-1]) if len(a2) >= 2 else names[1]
        authors_str = f"{last1}, & {last2}"
    else:
        a = names[0].split()
        authors_str = f"{a[-1]}, {' '.join(a[:-1])} et al." if len(a) >= 2 else f"{names[0]} et al."
    vol = ref.get("volume", "")
    issue = ref.get("issue", "")
    vol_issue = f"{vol}({issue}), " if issue and vol else f"{vol}, " if vol else ""
    pp = ref.get("pages", "1-10")
    doi = ref.get("doi", "")
    doi_str = f" https://doi.org/{doi}." if doi else "."
    return f"{authors_str} ({ref['year']}). {ref['title']}. *{ref['journal']}*, {vol_issue}pp. {pp}{doi_str}"


def format_mla(ref: dict, index: int) -> str:
    """MLA: Author. "Title." Journal, vol. X, no. X, Year, pp. X-X."""
    first = ref["authors"].split(",")[0].strip()
    parts = first.split()
    if len(parts) >= 2:
        first = f"{parts[-1]}, {' '.join(parts[:-1])}"
    vol = ref.get("volume", "1")
    issue = ref.get("issue", "")
    no = f", no. {issue}" if issue else ""
    pp = ref.get("pages", "1-10")
    return f'{first}. "{ref["title"]}." *{ref["journal"]}*, vol. {vol}{no}, {ref["year"]}, pp. {pp}.'


def format_ieee(ref: dict, index: int) -> str:
    """IEEE: A. Author, B. Author, "Title," Journal, vol. X, pp. X-X, Year."""
    names = [n.strip() for n in ref["authors"].split(",")[:3]]
    authors_ieee = ", ".join(_ieee_author(n) for n in names)
    if ref["authors"].count(",") >= 2:
        authors_ieee += " et al."
    vol = ref.get("volume", "1")
    pp = ref.get("pages", "1-10")
    return f'{authors_ieee}, "{ref["title"]}," *{ref["journal"]}*, vol. {vol}, pp. {pp}, {ref["year"]}.'


def format_gb_t_7714(ref: dict, index: int) -> str:
    """GB/T 7714: 作者. 题名[J]. 刊名, 年, 卷(期): 页码."""
    authors = ref["authors"]
    if "," in authors and "等" not in authors and not any(c >= "\u4e00" and c <= "\u9fff" for c in authors):
        authors = authors.split(",")[0].strip() + ", 等"
    vol = ref.get("volume", "")
    issue = ref.get("issue", "")
    vol_issue = f"{vol}({issue})" if issue and vol else vol or "—"
    pp = ref.get("pages", "1-10")
    return f"{authors}. {ref['title']}[J]. {ref['journal']}, {ref['year']}, {vol_issue}: {pp}."


def format_chicago(ref: dict, index: int) -> str:
    """Chicago: Author. Year. "Title." Journal Volume, no. Issue: pp–pp."""
    first = ref["authors"].split(",")[0].strip()
    vol = ref.get("volume", "1")
    issue = ref.get("issue", "")
    no_issue = f", no. {issue}" if issue else ""
    pp = ref.get("pages", "1-10")
    return f'{first}. {ref["year"]}. "{ref["title"]}." *{ref["journal"]}* {vol}{no_issue}: {pp}.'


def format_harvard(ref: dict, index: int) -> str:
    """Harvard: Author (Year) 'Title', Journal Name, Volume(Issue), pp. xx-xx."""
    first = ref["authors"].split(",")[0].strip()
    vol = ref.get("volume", "1")
    issue = ref.get("issue", "")
    vol_issue = f"{vol}({issue})" if issue else vol
    pp = ref.get("pages", "1-10")
    return f"{first} ({ref['year']}) '{ref['title']}', *{ref['journal']}*, {vol_issue}, pp. {pp}."


def format_vancouver(ref: dict, index: int) -> str:
    """Vancouver: Author AA, Author BB, et al. Title. Journal Name. Year;Volume(Issue):pp-pp."""
    authors = ref["authors"]
    if authors.count(",") > 2:
        # Vancouver 规范：et al 后不加句号，句号在作者部分结束后统一加
        authors = ", ".join(authors.split(",")[:2]) + ", et al"
    vol = ref.get("volume", "1")
    issue = ref.get("issue", "")
    vol_issue = f"{vol}({issue})" if issue else vol
    pp = ref.get("pages", "1-10")
    return f"{authors}. {ref['title']}. {ref['journal']}. {ref['year']};{vol_issue}:{pp}."


FORMATTERS = {
    "apa": format_apa,
    "mla": format_mla,
    "ieee": format_ieee,
    "gb_t_7714": format_gb_t_7714,
    "chicago": format_chicago,
    "harvard": format_harvard,
    "vancouver": format_vancouver,
}

# 四种列举方式：(名称, 前缀函数: index -> str)
LISTING_STYLES = [
    ("numbered_bracket", lambda i: f"[{i}] "),   # [1] xxx
    ("numbered_dot", lambda i: f"{i}. "),        # 1. xxx
    ("numbered_paren", lambda i: f"（{i}） "),   # （1）xxx
    ("author_year", lambda i: ""),                # 无编号
]


def build_refs_for_format(fmt: str, use_cn: bool = False) -> list:
    """为某格式生成 10 条参考文献（gb_t_7714 可混入中文示例）。"""
    refs = list(SAMPLE_REFERENCES)
    if use_cn and fmt == "gb_t_7714":
        refs = refs[:8] + SAMPLE_REFERENCES_CN[:2]
    return refs[:10]


def generate_all():
    """生成所有格式、所有列举方式的 MD 文件。"""
    output_base = project_root / "output"
    
    for fmt in CITATION_FORMATS:
        dir_path = output_base / fmt
        dir_path.mkdir(parents=True, exist_ok=True)
        
        formatter = FORMATTERS[fmt]
        refs_data = build_refs_for_format(fmt, use_cn=(fmt == "gb_t_7714"))
        
        for style_name, prefix_fn in LISTING_STYLES:
            lines = ["# 参考文献", ""]
            for i, ref in enumerate(refs_data, 1):
                prefix = prefix_fn(i)
                line = formatter(ref, i)
                lines.append(prefix + line)
                lines.append("")
            
            md_path = dir_path / f"{style_name}.md"
            md_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Generated: {md_path}")
    
    print("\nDone. Each format has 4 files: numbered_bracket, numbered_dot, numbered_paren, author_year.")


if __name__ == "__main__":
    generate_all()
