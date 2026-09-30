from app.services.document_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_markdown_document,
    split_table_lines,
)


def test_default_chunk_strategy_is_300_with_30_overlap():
    assert DEFAULT_CHUNK_SIZE == 300
    assert DEFAULT_CHUNK_OVERLAP == 30


def test_table_header_is_repeated_in_each_chunk():
    markdown = """| 商品 | 保修期 |
| --- | --- |
| A | 12 |
| B | 24 |
| C | 6 |
"""
    chunks = split_table_lines(markdown.splitlines(), max_rows=1)
    assert len(chunks) == 3
    for chunk in chunks:
        assert "| 商品 | 保修期 |" in chunk
        assert "| --- | --- |" in chunk


def test_faq_question_is_extracted():
    markdown = """# 商品FAQ

## 问：智能猫砂盆怎么清理？

答：断电后拆下集便仓，用清水冲洗晾干。
"""
    chunks = split_markdown_document(markdown, "商品FAQ.md")
    assert any("智能猫砂盆怎么清理" in str(item["questions"]) for item in chunks)


def test_policy_chunk_uses_section_title_as_question():
    markdown = """# 售后政策

## 运费说明

普通商品满 99 元包邮。
"""
    chunks = split_markdown_document(markdown, "售后政策.md")
    assert any(item["questions"] == ["运费说明"] for item in chunks)


def test_recursive_splitter_does_not_treat_headings_as_hard_boundaries():
    markdown = f"# 第一节\n\n{'甲' * 180}\n\n# 第二节\n\n{'乙' * 180}"

    chunks = split_markdown_document(
        markdown,
        "售后政策.md",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) == 1


def test_recursive_splitter_respects_chunk_size_and_overlap():
    markdown = "第一句。第二句。第三句。第四句。第五句。第六句。第七句。第八句。"

    chunks = split_markdown_document(
        markdown,
        "售后政策.md",
        chunk_size=25,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert max(len(item["answer"]) for item in chunks) <= 25
    first = chunks[0]["answer"]
    second = chunks[1]["answer"]
    assert any(
        first[-size:] == second[:size]
        for size in range(1, min(len(first), len(second)) + 1)
    )


def test_recursive_chunks_keep_nearest_heading_metadata():
    markdown = (
        "# 售后政策\n\n"
        "## 运费说明\n\n"
        + "普通商品满 99 元包邮。" * 50
        + "\n\n# 其他政策\n\n"
        + "其他说明。" * 50
    )

    chunks = split_markdown_document(markdown, "售后政策.md")

    assert chunks[0]["category"] == "售后政策"
    assert chunks[0]["section_path"] == "售后政策 > 运费说明"
    assert any(item["section_path"] == "其他政策" for item in chunks)


def test_faq_question_is_kept_on_each_recursive_chunk():
    markdown = (
        "# 商品FAQ\n\n"
        "## 问：智能猫砂盆怎么清理？\n\n"
        + "答：断电后拆下集便仓，用清水冲洗晾干。" * 30
    )

    chunks = split_markdown_document(markdown, "商品FAQ.md")

    assert len(chunks) > 1
    assert all("智能猫砂盆怎么清理" in str(item["questions"]) for item in chunks)


def test_faq_question_is_inherited_from_escaped_question_heading():
    markdown = (
        "# \\# 商品FAQ\n\n"
        "# \\## 问：iHao智能手表如何清洁保养？\n\n"
        + "# 答：使用无绒软布擦拭表身。" * 40
    )

    chunks = split_markdown_document(markdown, "商品FAQ.md")

    assert len(chunks) > 1
    assert all("iHao智能手表如何清洁保养" in str(item["questions"]) for item in chunks)
