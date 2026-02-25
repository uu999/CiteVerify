# -*- coding: utf-8 -*-
"""
参考文献提取校验测试

使用 generate_sample_references.py 中的 10 篇文献元数据，校验项目在不同
参考文献排列方式（有/无数字序号）和引用格式下提取的元数据（标题、作者、年份）是否准确，
并输出最终统计数据。
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if str(project_root / "scripts") not in sys.path:
    sys.path.insert(0, str(project_root / "scripts"))

# 从生成脚本导入元数据与生成函数
try:
    from generate_sample_references import (
        build_refs_for_format,
        generate_all,
    )
except ImportError:
    # 直接运行测试时可能从项目根执行
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_sample_references",
        project_root / "scripts" / "generate_sample_references.py",
    )
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)
    build_refs_for_format = gen_mod.build_refs_for_format
    generate_all = gen_mod.generate_all

from src.citeverify.extractor.reference_extractor import extract_references
from src.citeverify.models.reference import ListingStyle, CitationFormat

# 默认引用格式列表（若 import 失败则使用）
CITATION_FORMATS_DEFAULT = (
    "apa", "mla", "ieee", "gb_t_7714", "chicago", "harvard", "vancouver",
)


# 期望的 10 条元数据（与 SAMPLE_REFERENCES 一致，gb_t_7714 时后 2 条用中文）
def get_expected_metadata(citation_format: str) -> List[Dict[str, Any]]:
    """获取某格式下的 10 条期望元数据。"""
    refs = build_refs_for_format(citation_format, use_cn=(citation_format == "gb_t_7714"))
    return [
        {"title": r.get("title", ""), "authors": r.get("authors", ""), "year": str(r.get("year", ""))}
        for r in refs
    ]


def normalize_title(s: str) -> str:
    """标题标准化：去首尾空白、转小写、合并空白。"""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_authors_for_compare(s: str) -> set:
    """作者标准化为可比较集合：按分隔符拆分，转小写，去空白。"""
    if not s:
        return set()
    s = s.replace(";", ",").replace(" and ", ",")
    parts = [p.strip().lower() for p in re.split(r"[,;]", s) if p.strip()]
    return set(parts)


def get_first_author_last_name(authors_str: str) -> str:
    """从作者字符串中取第一个作者的姓氏（最后一词或首词）。"""
    if not authors_str:
        return ""
    first = authors_str.split(",")[0].strip().split(";")[0].strip()
    tokens = first.split()
    if not tokens:
        return ""
    # 西文常见 "LastName F." 或 "F. LastName"
    if len(tokens) == 1:
        return tokens[0].lower()
    # 若首词为大写单字母或带点，姓氏在最后
    if re.match(r"^[A-Z]\.?$", tokens[0], re.IGNORECASE):
        return tokens[-1].lower()
    return tokens[0].lower()


def authors_overlap(extracted: str, expected: str) -> bool:
    """判断提取的作者与期望作者是否有重叠（至少第一个作者姓氏匹配）。"""
    if not expected:
        return True
    exp_first_last = get_first_author_last_name(expected)
    if not exp_first_last:
        return bool(extracted)
    ext_lower = (extracted or "").lower()
    # 第一个作者姓氏出现在提取结果中即视为匹配
    if exp_first_last in ext_lower:
        return True
    ex_set = normalize_authors_for_compare(extracted or "")
    for e in ex_set:
        if exp_first_last in e or e.endswith(exp_first_last):
            return True
    return False


def normalize_year(s: str) -> str:
    """年份标准化：提取 4 位数字。"""
    if not s:
        return ""
    m = re.search(r"19\d{2}|20\d{2}", str(s))
    return m.group(0) if m else str(s).strip()


def compare_one(
    extracted_title: str,
    extracted_authors: str,
    extracted_year: str,
    expected: Dict[str, Any],
) -> Tuple[bool, bool, bool]:
    """
    比较单条提取结果与期望。
    Returns:
        (title_ok, authors_ok, year_ok)
    """
    exp_title = normalize_title(expected.get("title", ""))
    exp_authors = expected.get("authors", "")
    exp_year = normalize_year(expected.get("year", ""))

    ext_title_norm = normalize_title(extracted_title or "")
    title_ok = (
        (exp_title and (ext_title_norm == exp_title or exp_title in ext_title_norm or ext_title_norm in exp_title))
        or (not exp_title)
    )

    authors_ok = authors_overlap(extracted_authors or "", exp_authors)

    year_ok = (normalize_year(extracted_year or "") == exp_year) if exp_year else True

    return title_ok, authors_ok, year_ok


def run_validation() -> Dict[str, Any]:
    """
    运行完整校验：对每种引用格式、每种列举方式，用生成的 MD 做提取并统计。
    """
    output_base = project_root / "output"
    if not output_base.exists():
        generate_all()

    # 列举方式与文件名对应
    # numbered 对应 3 种文件；author_year 对应 1 种
    listing_file_map = [
        (ListingStyle.NUMBERED, "numbered_bracket"),
        (ListingStyle.NUMBERED, "numbered_dot"),
        (ListingStyle.NUMBERED, "numbered_paren"),
        (ListingStyle.AUTHOR_YEAR, "author_year"),
    ]

    results = []
    totals = {"title_ok": 0, "authors_ok": 0, "year_ok": 0, "all_ok": 0, "total": 0}

    for fmt in CITATION_FORMATS_DEFAULT:
        expected_list = get_expected_metadata(fmt)
        fmt_dir = output_base / fmt
        if not fmt_dir.exists():
            generate_all()
            if not fmt_dir.exists():
                results.append({"format": fmt, "error": "MD files not found"})
                continue

        for listing_style, file_stem in listing_file_map:
            md_path = fmt_dir / f"{file_stem}.md"
            if not md_path.exists():
                continue

            text = md_path.read_text(encoding="utf-8")
            # 只取参考文献部分（# 参考文献 之后）
            if "# 参考文献" in text:
                text = text.split("# 参考文献", 1)[-1].strip()

            try:
                refs_raw = extract_references(
                    text,
                    listing_style=listing_style.value,
                    citation_format=fmt,
                )
            except Exception as e:
                results.append({
                    "format": fmt,
                    "listing": file_stem,
                    "error": str(e),
                    "title_ok": 0,
                    "authors_ok": 0,
                    "year_ok": 0,
                    "all_ok": 0,
                    "count": 0,
                })
                continue

            # refs_raw: numbered -> [num, full, title, authors_str, year]; author_year -> [full, title, authors_str, year]
            n = len(refs_raw)
            title_ok = authors_ok = year_ok = all_ok = 0

            for i in range(min(n, len(expected_list))):
                exp = expected_list[i]
                if listing_style == ListingStyle.NUMBERED:
                    _, _, ext_title, ext_authors, ext_year = refs_raw[i]
                else:
                    _, ext_title, ext_authors, ext_year = refs_raw[i]

                t_ok, a_ok, y_ok = compare_one(ext_title, ext_authors, ext_year, exp)
                if t_ok:
                    title_ok += 1
                if a_ok:
                    authors_ok += 1
                if y_ok:
                    year_ok += 1
                if t_ok and a_ok and y_ok:
                    all_ok += 1

            total = min(n, len(expected_list))
            totals["title_ok"] += title_ok
            totals["authors_ok"] += authors_ok
            totals["year_ok"] += year_ok
            totals["all_ok"] += all_ok
            totals["total"] += total

            results.append({
                "format": fmt,
                "listing": file_stem,
                "title_ok": title_ok,
                "authors_ok": authors_ok,
                "year_ok": year_ok,
                "all_ok": all_ok,
                "count": total,
            })

    return {"results": results, "totals": totals}


def print_report(stats: Dict[str, Any]) -> None:
    """打印校验报告与最终统计。"""
    results = stats["results"]
    totals = stats["totals"]

    print("=" * 70)
    print("参考文献提取校验报告")
    print("=" * 70)

    # 按格式分组
    by_format = {}
    for r in results:
        if "error" in r:
            print(f"  [错误] {r['format']} ({r.get('listing', '')}): {r['error']}")
            continue
        fmt = r["format"]
        if fmt not in by_format:
            by_format[fmt] = []
        by_format[fmt].append(r)

    print("\n按引用格式与列举方式：")
    print("-" * 70)
    formats_order = [f for f in CITATION_FORMATS_DEFAULT if f in by_format]
    formats_order += [f for f in by_format if f not in CITATION_FORMATS_DEFAULT]
    for fmt in formats_order:
        if fmt not in by_format:
            continue
        rows = by_format[fmt]
        for r in rows:
            c = r["count"]
            if c == 0:
                continue
            t_pct = 100 * r["title_ok"] / c
            a_pct = 100 * r["authors_ok"] / c
            y_pct = 100 * r["year_ok"] / c
            all_pct = 100 * r["all_ok"] / c
            print(f"  {fmt:12} | {r['listing']:18} | 标题:{t_pct:5.1f}% 作者:{a_pct:5.1f}% 年份:{y_pct:5.1f}% 三项全对:{all_pct:5.1f}% ({r['all_ok']}/{c})")

    print("\n" + "=" * 70)
    print("最终统计数据（汇总所有格式与列举方式）")
    print("=" * 70)

    total = totals["total"]
    if total == 0:
        print("  无有效校验条目。请先运行: python scripts/generate_sample_references.py")
        return

    t_pct = 100 * totals["title_ok"] / total
    a_pct = 100 * totals["authors_ok"] / total
    y_pct = 100 * totals["year_ok"] / total
    all_pct = 100 * totals["all_ok"] / total

    print(f"  总条数:        {total}")
    print(f"  标题正确:      {totals['title_ok']} / {total}  ({t_pct:.1f}%)")
    print(f"  作者正确:      {totals['authors_ok']} / {total}  ({a_pct:.1f}%)")
    print(f"  年份正确:      {totals['year_ok']} / {total}  ({y_pct:.1f}%)")
    print(f"  三项全对:      {totals['all_ok']} / {total}  ({all_pct:.1f}%)")
    print("=" * 70)


def test_reference_extraction_validation():
    """
    校验函数：生成示例 MD -> 提取参考文献 -> 与期望元数据对比 -> 输出统计。
    """
    stats = run_validation()
    print_report(stats)
    return stats
def debug_ieee_vancouver():
    """
    专门调试 IEEE 与 Vancouver 格式提取失败的问题。
    打印每条参考文献的：
      - 原始引用字符串
      - 期望元数据
      - 提取结果
      - 各字段是否匹配
    """
    problematic_formats = ["ieee", "vancouver"]
    listing_file_map = [
        (ListingStyle.NUMBERED, "numbered_bracket"),
        (ListingStyle.NUMBERED, "numbered_dot"),
        (ListingStyle.NUMBERED, "numbered_paren"),
        (ListingStyle.AUTHOR_YEAR, "author_year"),
    ]

    output_base = project_root / "output"
    if not output_base.exists():
        generate_all()

    print("=" * 100)
    print("🔍 详细调试报告：IEEE 与 Vancouver 格式提取问题分析")
    print("=" * 100)

    for fmt in problematic_formats:
        print(f"\n{'='*40} {fmt.upper()} {'='*40}")
        expected_list = get_expected_metadata(fmt)
        fmt_dir = output_base / fmt

        for listing_style, file_stem in listing_file_map:
            md_path = fmt_dir / f"{file_stem}.md"
            if not md_path.exists():
                continue

            text = md_path.read_text(encoding="utf-8")
            if "# 参考文献" in text:
                text = text.split("# 参考文献", 1)[-1].strip()

            try:
                refs_raw = extract_references(
                    text,
                    listing_style=listing_style.value,
                    citation_format=fmt,
                )
            except Exception as e:
                print(f"❌ 提取失败 ({file_stem}): {e}")
                continue

            print(f"\n📄 文件: {file_stem}.md")
            print("-" * 90)

            n = min(len(refs_raw), len(expected_list))
            for i in range(n):
                exp = expected_list[i]

                if listing_style == ListingStyle.NUMBERED:
                    raw_ref_str = refs_raw[i][1]  # full reference string
                    _, _, ext_title, ext_authors, ext_year = refs_raw[i]
                else:
                    raw_ref_str = refs_raw[i][0]
                    _, ext_title, ext_authors, ext_year = refs_raw[i]

                t_ok, a_ok, y_ok = compare_one(ext_title, ext_authors, ext_year, exp)

                print(f"\n[条目 {i+1}]")
                print(f"  📌 原始引用: {repr(raw_ref_str[:150] + '...' if len(raw_ref_str) > 150 else raw_ref_str)}")
                print(f"  ✅ 期望: 标题={repr(exp['title'])}, 作者={repr(exp['authors'])}, 年份={repr(exp['year'])}")
                print(f"  🔍 提取: 标题={repr(ext_title)}, 作者={repr(ext_authors)}, 年份={repr(ext_year)}")

                status = []
                status.append("✅标题" if t_ok else "❌标题")
                status.append("✅作者" if a_ok else "❌作者")
                status.append("✅年份" if y_ok else "❌年份")
                print(f"  🧪 匹配: {' | '.join(status)}")

                # 如果作者或标题失败，额外打印标准化中间结果（帮助诊断）
                if not a_ok:
                    exp_auth_set = normalize_authors_for_compare(exp["authors"])
                    ext_auth_set = normalize_authors_for_compare(ext_authors or "")
                    first_last = get_first_author_last_name(exp["authors"])
                    print(f"     💡 期望作者姓氏（首作者）: '{first_last}'")
                    print(f"     💡 期望作者集合: {exp_auth_set}")
                    print(f"     💡 提取作者集合: {ext_auth_set}")

                if not t_ok:
                    exp_norm = normalize_title(exp["title"])
                    ext_norm = normalize_title(ext_title or "")
                    print(f"     💡 期望标题（标准化）: {repr(exp_norm)}")
                    print(f"     💡 提取标题（标准化）: {repr(ext_norm)}")

            # 补充未覆盖的条目（如果生成了10条但只提取出<10条）
            if len(refs_raw) < len(expected_list):
                for j in range(len(refs_raw), len(expected_list)):
                    print(f"\n[条目 {j+1}] ❗ 未提取到任何结果！")
                    exp = expected_list[j]
                    print(f"  ✅ 期望: 标题={repr(exp['title'])}, 作者={repr(exp['authors'])}, 年份={repr(exp['year'])}")


if __name__ == "__main__":
    # 测试所有
    test_reference_extraction_validation()

    # 新增：运行专门调试
    #debug_ieee_vancouver()


