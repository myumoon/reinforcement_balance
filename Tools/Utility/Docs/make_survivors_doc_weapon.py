#!/usr/bin/env python3
"""
Vampire Survivors wiki の武器個別ページから Markdown ドキュメントを生成する。
Usage: python make_survivors_doc_weapon.py <wiki_url> <output_path>
Example: python make_survivors_doc_weapon.py https://vampire.survivors.wiki/w/Garlic weapons_garlic.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


# インフォボックスの第1グループ（メタ情報）のラベル - Statsセクションから除外
_SKIP_STAT_LABELS = {
    "id", "type", "evolution", "evolved with", "max level", "rarity", "effects",
}


def fetch_soup(url: str) -> BeautifulSoup:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; VSWikiScraper/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_weapon_name(soup: BeautifulSoup) -> str:
    title_div = soup.find("div", class_="infobox-title")
    if title_div:
        return title_div.get_text(strip=True)
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else "Unknown"


def cell_to_text(elem: Tag) -> str:
    """
    要素をテキスト化する。<br> → <br/>、<a> → リンクテキスト、
    <sup class="reference"> → [n] として変換する。
    """
    parts: list[str] = []
    for node in elem.children:
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag):
            if node.name == "br":
                parts.append("<br/>")
            elif node.name == "sup" and "reference" in " ".join(node.get("class", [])):
                # 脚注参照 [n] を抽出
                num = re.search(r"\d+", node.get_text())
                if num:
                    parts.append(f"[{num.group()}]")
            elif node.name == "a":
                parts.append(node.get_text(strip=True))
            elif node.name == "span" and "mw-editsection" in " ".join(node.get("class", [])):
                pass  # edit リンクは無視
            else:
                parts.append(cell_to_text(node))

    result = "".join(parts).strip()
    result = re.sub(r"[ \t]+", " ", result)
    return result


def _extract_tab_stats(tab: Tag) -> list[tuple[str, str]]:
    """tabbertab 要素からステータス行を抽出する。"""
    stats: list[tuple[str, str]] = []
    for row in tab.find_all("div", class_="infobox-row-container"):
        label_div = row.find("div", class_="infobox-row-label")
        value_div = row.find("div", class_="infobox-row-value__inner")
        if not label_div or not value_div:
            continue
        label = label_div.get_text(strip=True)
        if label.lower() in _SKIP_STAT_LABELS:
            continue
        value = cell_to_text(value_div)
        if value:
            stats.append((label, value))
    return stats


def get_stats(soup: BeautifulSoup) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    インフォボックスから (Normal ステータス, Max ステータス) を返す。
    構造: div.infobox-root > div.infobox-group[1] > div.tabber > div.tabbertab[0/1]
    """
    infobox = soup.find("div", class_="infobox-root")
    if not infobox:
        return [], []

    groups = infobox.find_all("div", class_="infobox-group", recursive=False)
    if len(groups) < 2:
        return [], []

    tabber = groups[1].find("div", class_="tabber")
    if not tabber:
        return [], []

    tabs = tabber.find_all("div", class_="tabbertab", recursive=False)
    normal = _extract_tab_stats(tabs[0]) if len(tabs) > 0 else []
    max_   = _extract_tab_stats(tabs[1]) if len(tabs) > 1 else []
    return normal, max_


def _heading_text(heading_div: Tag) -> str:
    """
    div.mw-heading 内の h2/h3 要素のテキストを返す（編集リンクを除く）。
    """
    inner = heading_div.find(re.compile(r"^h[1-6]$"))
    return inner.get_text(strip=True) if inner else heading_div.get_text(strip=True)


def find_heading_div(parser_out: Tag, keyword: str, level: int = 2) -> Tag | None:
    """キーワードにマッチする mw-heading{level} div を探す。"""
    cls_name = f"mw-heading{level}"
    for div in parser_out.find_all("div", class_=lambda c: c and cls_name in c):
        if keyword.lower() in _heading_text(div).lower():
            return div
    return None


def iter_section_siblings(start: Tag, stop_classes: tuple[str, ...]) -> list[Tag]:
    """
    start の次の兄弟要素を stop_classes のいずれかを持つ div が出るまで収集する。
    """
    elements: list[Tag] = []
    node = start.next_sibling
    while node:
        if isinstance(node, Tag):
            node_classes = " ".join(node.get("class", []))
            if any(cls in node_classes for cls in stop_classes):
                break
            elements.append(node)
        node = node.next_sibling
    return elements


def get_effects(parser_out: Tag) -> list[str]:
    """Effects セクションの段落テキストを返す。"""
    h = find_heading_div(parser_out, "effect", level=2)
    if not h:
        return []

    paragraphs: list[str] = []
    for elem in iter_section_siblings(h, ("mw-heading2",)):
        if elem.name == "p":
            text = cell_to_text(elem)
            if text:
                paragraphs.append(text)
    return paragraphs


def parse_wikitable(table: Tag) -> list[list[str]]:
    """wikitable を [[cell, ...], ...] に変換する。"""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if cells:
            rows.append([cell_to_text(c) for c in cells])
    return rows


def get_levels(parser_out: Tag):
    """
    Levels セクションから以下を返す:
      (levels_desc, levels_rows, lb_desc, lb_rows, footnotes)
    """
    levels_h = find_heading_div(parser_out, "level", level=2)
    if not levels_h:
        return "", [], "", [], []

    section_elems = iter_section_siblings(levels_h, ("mw-heading2",))

    # Limit Break h3 を探す
    lb_h = None
    for elem in section_elems:
        if isinstance(elem, Tag):
            cls = " ".join(elem.get("class", []))
            if "mw-heading3" in cls and "limit break" in _heading_text(elem).lower():
                lb_h = elem
                break

    if lb_h:
        lb_idx = section_elems.index(lb_h)
        levels_elems = section_elems[:lb_idx]
        lb_elems = section_elems[lb_idx + 1:]
    else:
        levels_elems = section_elems
        lb_elems = []

    levels_desc = ""
    levels_rows: list[list[str]] = []
    for elem in levels_elems:
        if elem.name == "p" and not levels_desc:
            text = cell_to_text(elem)
            if text:
                levels_desc = text
        elif elem.name == "table" and not levels_rows:
            levels_rows = parse_wikitable(elem)

    lb_desc = ""
    lb_rows: list[list[str]] = []
    for elem in lb_elems:
        if elem.name == "p" and not lb_desc:
            text = cell_to_text(elem)
            if text:
                lb_desc = text
        elif elem.name == "table" and not lb_rows:
            lb_rows = parse_wikitable(elem)

    # 脚注（Limit Breakセクション内の最初の mw-references-wrap）
    footnotes: list[str] = []
    for elem in lb_elems:
        if isinstance(elem, Tag):
            ref_wrap = elem if "mw-references-wrap" in " ".join(elem.get("class", [])) else elem.find("div", class_="mw-references-wrap")
            if ref_wrap:
                for span in ref_wrap.find_all("span", class_="reference-text"):
                    # get_text(strip=True) は内部スペースを除去するため cell_to_text を使う
                    text = cell_to_text(span)
                    if text:
                        footnotes.append(text)
                break

    return levels_desc, levels_rows, lb_desc, lb_rows, footnotes


def rows_to_md_table(rows: list[list[str]]) -> list[str]:
    """行リストを markdown テーブル形式に変換する（1行目をヘッダーとして扱う）。"""
    if not rows:
        return []
    header = rows[0]
    sep = "|" + "|".join("---" for _ in header) + "|"
    lines = ["|" + "|".join(header) + "|", sep]
    for row in rows[1:]:
        lines.append("|" + "|".join(row) + "|")
    return lines


def stats_to_md_table(stats: list[tuple[str, str]]) -> list[str]:
    """ステータスリストを Stats 形式の markdown テーブルに変換する。"""
    lines = ["|Stats||", "|---|---|"]
    for label, value in stats:
        lines.append(f"|{label} | {value} |")
    return lines


def build_markdown(
    name: str,
    stats: list[tuple[str, str]],
    max_stats: list[tuple[str, str]],
    effects: list[str],
    levels_desc: str,
    levels_rows: list[list[str]],
    lb_desc: str,
    lb_rows: list[list[str]],
    footnotes: list[str],
) -> str:
    lines: list[str] = []

    lines += [f"# {name}", "", "## Stats", ""]
    if stats:
        lines += stats_to_md_table(stats)
    lines.append("")

    if max_stats:
        lines += ["### Max", ""]
        lines += stats_to_md_table(max_stats)
        lines.append("")

    lines += ["## Effects", ""]
    for para in effects:
        lines.append(para)
        lines.append("")

    lines += ["## Levels", ""]
    if levels_desc:
        lines.append(levels_desc)
        lines.append("")
    if levels_rows:
        lines += rows_to_md_table(levels_rows)
    lines.append("")

    if lb_rows:
        lines += ["### Limit Break", ""]
        if lb_desc:
            lines.append(lb_desc)
            lines.append("")
        lines += rows_to_md_table(lb_rows)
        lines.append("")
        for i, note in enumerate(footnotes, 1):
            lines.append(f"{i}. {note}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python make_survivors_doc_weapon.py <wiki_url> <output_path>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    out = Path(sys.argv[2])

    print(f"Fetching: {url}")
    soup = fetch_soup(url)

    parser_out = soup.find("div", class_="mw-parser-output")
    if not parser_out:
        print("Error: Could not find mw-parser-output div", file=sys.stderr)
        sys.exit(1)

    name = get_weapon_name(soup)
    print(f"Weapon: {name}")

    stats, max_stats = get_stats(soup)
    effects = get_effects(parser_out)
    levels_desc, levels_rows, lb_desc, lb_rows, footnotes = get_levels(parser_out)

    md = build_markdown(name, stats, max_stats, effects, levels_desc, levels_rows, lb_desc, lb_rows, footnotes)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
