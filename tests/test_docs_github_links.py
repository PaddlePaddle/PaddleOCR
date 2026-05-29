import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_docs_github_links_rejects_moving_source_refs(tmp_path):
    checker = _load_tool("check_docs_github_links")
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(
        "[config](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/config.yaml)\n",
        encoding="utf-8",
    )

    violations = checker.find_forbidden_links(
        tmp_path,
        repo_slug="PaddlePaddle/PaddleOCR",
        forbidden_refs={"main", "master"},
    )

    assert len(violations) == 1
    assert violations[0].path == doc_path
    assert violations[0].ref == "main"
    assert violations[0].line_number == 1


def test_check_docs_github_links_allows_versioned_and_placeholder_refs(tmp_path):
    checker = _load_tool("check_docs_github_links")
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(
        "\n".join(
            [
                "[release](https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.5/deploy/config.yaml)",
                "[placeholder](https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/deploy/config.yaml)",
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.find_forbidden_links(
        tmp_path,
        repo_slug="PaddlePaddle/PaddleOCR",
        forbidden_refs={"main", "master"},
    )

    assert violations == []


def test_resolve_doc_github_refs_replaces_placeholders(tmp_path):
    resolver = _load_tool("resolve_doc_github_refs")
    doc_path = tmp_path / "doc.md"
    doc_path.write_text(
        "[config](https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/deploy/config.yaml)\n",
        encoding="utf-8",
    )

    changed = resolver.resolve_placeholders(
        tmp_path,
        placeholder="{{PADDLEOCR_GITHUB_REF}}",
        source_ref="release/3.5",
    )

    assert changed == [doc_path]
    assert (
        doc_path.read_text(encoding="utf-8")
        == "[config](https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.5/deploy/config.yaml)\n"
    )
