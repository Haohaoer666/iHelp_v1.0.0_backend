"""
使用 RecursiveCharacterTextSplitter，优先按段落、换行、中文句号、问号、叹号、逗号等顺序切分。
标题不强制作为切片边界，标题只用于生成 category 和 section_path 元数据。
Markdown 表格会单独识别，按行数或字符数拆分，并且每个表格切片都会重复表头。
"""

from dataclasses import dataclass
from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 30

_FAQ_QUESTION_RE = re.compile(r"(?m)^[^\n]*?问[：:][ \t]*(.+?)[ \t]*$")
_CRITICAL_KEYWORDS = ("必须", "不得", "禁止", "退款", "运费", "售后规则", "影响二次销售")


@dataclass(frozen=True)      #记录标题在原文中的起始位置和完整层级路径，例如 ("商品FAQ", "退款政策")。
class _Heading:
    start: int
    path: tuple[str, ...]


@dataclass(frozen=True)     #记录普通文本或表格片段、它在全文中的起始 offset、是否为表格。
class _Segment:
    text: str
    start: int
    is_table: bool


@dataclass(frozen=True)     #记录 FAQ 问句的位置和文本。
class _Question:
    start: int
    text: str


#拆分 Markdown 表格。它只保留以 | 开头的行，少于 3 行就认为不是完整表格，返回空列表。
#每一块都重复带上表头 + 分隔线
def split_table_lines(
    lines: list[str],
    max_rows: int = 8,
    max_chars: int | None = None,
) -> list[str]:
    """Split a Markdown table into row groups, repeating the header in every chunk."""
    table_lines = [line for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return []

    header = table_lines[0]
    separator = table_lines[1]
    rows = table_lines[2:]
    chunks: list[str] = []
    current = [header, separator]
    current_rows = 0

    for row in rows:
        candidate = [*current, row]
        exceeds_rows = current_rows >= max_rows
        exceeds_chars = (
            max_chars is not None
            and current_rows > 0
            and len("\n".join(candidate)) > max_chars
        )
        if exceeds_rows or exceeds_chars:
            chunks.append("\n".join(current))
            current = [header, separator, row]
            current_rows = 1
        else:
            current.append(row)
            current_rows += 1

    if current_rows:
        chunks.append("\n".join(current))
    return chunks


def _extract_headings(markdown: str) -> list[_Heading]:
    headings: list[_Heading] = []
    path: list[str] = []

    for match in re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$", markdown):
        level = len(match.group(1))
        title = match.group(2).strip()
        path = path[: level - 1]
        path.append(title)
        headings.append(_Heading(start=match.start(), path=tuple(path)))

    return headings         # 类似于_Heading(start=0, path=("第一章 项目介绍",)),


def _extract_faq_questions(markdown: str) -> list[_Question]:
    return [
        _Question(start=match.start(), text=match.group(1).strip())
        for match in _FAQ_QUESTION_RE.finditer(markdown)
        if match.group(1).strip()
    ]


def _resolve_heading_path(
    start: int,
    text: str,
    headings: list[_Heading],
    fallback: str,
) -> tuple[str, ...]:
    end = start + len(text)
    path: tuple[str, ...] = (fallback,)
    for heading in headings:
        if heading.start >= end:
            break
        path = heading.path
    return path


def _is_fence(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


#逐行扫描 Markdown，跳过代码块，识别连续`|`开头的表格区域，把文档切分为**普通文本段、表格段**，记录各段在全文的字符偏移，返回分段列表。
def _iter_segments(markdown: str) -> list[_Segment]:
    lines = markdown.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    segments: list[_Segment] = []
    text_start = 0
    in_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]
        if _is_fence(line):
            in_fence = not in_fence
            index += 1
            continue

        if not in_fence and line.strip().startswith("|"):
            table_end_index = index + 1
            while (
                table_end_index < len(lines)
                and lines[table_end_index].strip().startswith("|")
            ):
                table_end_index += 1

            if table_end_index - index >= 3:
                table_start = offsets[index]
                table_end = (
                    offsets[table_end_index]
                    if table_end_index < len(lines)
                    else len(markdown)
                )
                if table_start > text_start:
                    segments.append(
                        _Segment(
                            text=markdown[text_start:table_start],
                            start=text_start,
                            is_table=False,
                        )
                    )
                segments.append(
                    _Segment(
                        text=markdown[table_start:table_end],
                        start=table_start,
                        is_table=True,
                    )
                )
                text_start = table_end

            index = table_end_index
            continue

        index += 1

    if text_start < len(markdown):
        segments.append(
            _Segment(
                text=markdown[text_start:],
                start=text_start,
                is_table=False,
            )
        )

    return segments


def _split_text_with_offsets(
    text: str,
    start: int,
    splitter: RecursiveCharacterTextSplitter,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    parts = splitter.split_text(text)
    results: list[tuple[str, int]] = []
    search_from = 0

    for part in parts:
        if not part:
            continue

        found = text.find(part, search_from)
        if found == -1:
            found = text.find(part)
        if found == -1:
            found = search_from

        results.append((part, start + found))
        search_from = max(found + 1, found + len(part) - chunk_overlap)

    return results


def _extract_questions(
    text: str,
    heading_path: tuple[str, ...],
    content_type: str,
    faq_questions: list[_Question],
    start: int,
    end: int,
) -> list[str]:
    questions: list[str] = []

    if content_type == "faq":
        questions.extend(
            match.group(1).strip()
            for match in _FAQ_QUESTION_RE.finditer(text)
            if match.group(1).strip()
        )
        if not questions:
            previous_question = next(
                (
                    question.text
                    for question in reversed(faq_questions)
                    if question.start < end
                ),
                "",
            )
            if previous_question:
                questions.append(previous_question)
    elif heading_path:
        questions.append(heading_path[-1])

    return list(dict.fromkeys(question for question in questions if question))


def split_markdown_document(
    markdown: str,
    source_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    content_type = "faq" if "faq" in source_name.lower() else "policy"
    category = Path(source_name).stem
    headings = _extract_headings(markdown)
    faq_questions = _extract_faq_questions(markdown) if content_type == "faq" else []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        keep_separator=True,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            ". ",
            "! ",
            "? ",
            "; ",
            "，",
            ", ",
            " ",
            "",
        ],
    )
    chunks: list[dict] = []

    for segment in _iter_segments(markdown):    #把markdown文档切分成多个Segment：区分普通文本 / Markdown表格
        if segment.is_table:        # 判断当前segment是不是表格
            parts = [
                (part, segment.start)
                for part in split_table_lines(  #表格单独切片
                    segment.text.splitlines(),
                    max_chars=chunk_size,
                )
            ]
        else:
            parts = _split_text_with_offsets(   #普通文本切片
                segment.text,
                segment.start,
                splitter,
                chunk_overlap,
            )

        for part, start in parts:   #遍历每一个分片结果
            answer = part.strip()
            if not answer:
                continue

            heading_path = _resolve_heading_path(   #根据分片在全文的起始位置，查找当前分片归属的标题层级路径
                start,
                part,
                headings,
                category,
            )
            section_path = " > ".join(heading_path) or category     # 把标题路径元组拼接成字符串，例如：产品说明 > 售后 > 退款规则
            end = start + len(part)
            chunks.append(  #组装chunk字典，后续存入knowledge_chunks
                {
                    "category": heading_path[0] if heading_path else category,
                    "questions": _extract_questions(    #提取当前分片对应的问题（FAQ提取问句；普通文档取标题作为问题）
                        answer,
                        heading_path,
                        content_type,
                        faq_questions,
                        start,
                        end,
                    ),
                    "answer": answer,
                    "section_path": section_path,
                    "content_type": content_type,
                    "is_critical": any(
                        keyword in answer for keyword in _CRITICAL_KEYWORDS
                    ),
                }
            )

    return chunks
