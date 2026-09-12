"""Unit seam: `select_corpus` — signal is always full, noise is a fixed-seed sample.

Pure function over a directory tree; no DB, no vector store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from scripts.seed_db import DOCS_DIR, SIGNAL_SUBDIR, select_corpus, stage_noise_corpus


def _write_pdf(path: Path, num_pages: int) -> None:
    """A real, minimal N-page PDF — for testing the page-budget filter."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A throwaway docs tree: 5 signal files, 40 noise files, plus decoys."""
    signal = tmp_path / "true_data"
    noise = tmp_path / "noisy_data"
    signal.mkdir()
    noise.mkdir()
    for i in range(5):
        (signal / f"doc_{i}.txt").write_text("signal")
    for i in range(40):
        (noise / f"noise_{i:02d}.pdf").write_text("noise")
    # decoys that must be ignored
    (signal / ".gitkeep").write_text("")
    (noise / "notes.md").write_text("still counts — .md is supported")
    (noise / "archive.zip").write_text("unsupported suffix, skip")
    return tmp_path


def test_signal_is_always_taken_in_full(corpus_dir: Path) -> None:
    for sample in (0, 3, 999, "all"):
        assert len(select_corpus(sample, docs_dir=corpus_dir).signal) == 5


def test_noise_sample_size_is_respected(corpus_dir: Path) -> None:
    assert len(select_corpus(10, docs_dir=corpus_dir).noise) == 10


def test_noise_zero_takes_nothing(corpus_dir: Path) -> None:
    assert select_corpus(0, docs_dir=corpus_dir).noise == []


def test_noise_all_takes_the_whole_pool(corpus_dir: Path) -> None:
    # 40 .pdf + 1 .md
    assert len(select_corpus("all", docs_dir=corpus_dir).noise) == 41


def test_noise_sample_above_pool_size_clamps(corpus_dir: Path) -> None:
    assert len(select_corpus(10_000, docs_dir=corpus_dir).noise) == 41


def test_noise_sample_is_deterministic(corpus_dir: Path) -> None:
    first = select_corpus(12, docs_dir=corpus_dir).noise
    second = select_corpus(12, docs_dir=corpus_dir).noise
    assert first == second


def test_noise_sample_is_sorted(corpus_dir: Path) -> None:
    noise = select_corpus(12, docs_dir=corpus_dir).noise
    assert noise == sorted(noise)


def test_noise_sample_draws_from_the_pool(corpus_dir: Path) -> None:
    pool = set(select_corpus("all", docs_dir=corpus_dir).noise)
    assert set(select_corpus(7, docs_dir=corpus_dir).noise) <= pool


def test_negative_sample_is_rejected(corpus_dir: Path) -> None:
    with pytest.raises(ValueError, match="noise_sample"):
        select_corpus(-1, docs_dir=corpus_dir)


# --- page-budget filter (protects against multi-hundred-page PDFs blowing up
# --- a docling ingestion run, which analyses every page) --------------------


@pytest.fixture
def page_budget_dir(tmp_path: Path) -> Path:
    """Signal (irrelevant here) + a noise pool with a mix of short and long PDFs."""
    signal = tmp_path / "true_data"
    noise = tmp_path / "noisy_data"
    signal.mkdir()
    noise.mkdir()
    (signal / "doc.txt").write_text("signal")
    _write_pdf(noise / "short.pdf", num_pages=5)
    _write_pdf(noise / "medium.pdf", num_pages=50)
    _write_pdf(noise / "huge.pdf", num_pages=500)
    (noise / "not_really_a_pdf.pdf").write_text("garbage, not a real PDF")
    (noise / "notes.md").write_text("non-PDF, never subject to a page count")
    return tmp_path


def test_noise_pool_excludes_pdfs_over_the_page_budget(page_budget_dir: Path) -> None:
    noise = select_corpus("all", docs_dir=page_budget_dir, max_noise_pages=100).noise
    names = {p.name for p in noise}

    assert "huge.pdf" not in names
    assert "short.pdf" in names
    assert "medium.pdf" in names


def test_max_noise_pages_none_disables_the_filter(page_budget_dir: Path) -> None:
    noise = select_corpus("all", docs_dir=page_budget_dir, max_noise_pages=None).noise

    assert {p.name for p in noise} == {
        "short.pdf",
        "medium.pdf",
        "huge.pdf",
        "not_really_a_pdf.pdf",
        "notes.md",
    }


def test_non_pdf_files_are_never_excluded_by_the_page_budget(page_budget_dir: Path) -> None:
    noise = select_corpus("all", docs_dir=page_budget_dir, max_noise_pages=1).noise
    names = {p.name for p in noise}

    assert "notes.md" in names


def test_an_unreadable_pdf_fails_open_rather_than_being_excluded(page_budget_dir: Path) -> None:
    # A corrupt/fake PDF whose page count can't be determined must still be
    # eligible — the filter should never silently drop something it can't
    # actually inspect.
    noise = select_corpus("all", docs_dir=page_budget_dir, max_noise_pages=1).noise
    names = {p.name for p in noise}

    assert "not_really_a_pdf.pdf" in names


def test_default_max_noise_pages_still_returns_the_untouched_pool(corpus_dir: Path) -> None:
    # The existing fixture's fake .pdf files (plain text, not real PDFs) must
    # still all come through under the *default* budget, unchanged from
    # before this filter existed.
    assert len(select_corpus("all", docs_dir=corpus_dir).noise) == 41


def test_missing_directories_yield_empty(tmp_path: Path) -> None:
    selection = select_corpus("all", docs_dir=tmp_path / "nope")
    assert selection.signal == []
    assert selection.noise == []
    assert selection.total == 0


def test_stage_noise_corpus_symlinks_from_root_staging_folder(tmp_path: Path) -> None:
    staging = tmp_path / "noisy_data 2"
    staging.mkdir()
    for i in range(3):
        (staging / f"paper_{i}.pdf").write_bytes(b"noise body")
    target = tmp_path / "seed" / "docs" / "noisy_data"

    linked = stage_noise_corpus(root=tmp_path)

    assert linked == 3
    files = sorted(target.iterdir())
    assert [f.name for f in files] == ["paper_0.pdf", "paper_1.pdf", "paper_2.pdf"]
    assert all(f.is_symlink() for f in files)
    assert files[0].read_bytes() == b"noise body"  # the symlink resolves


def test_stage_noise_corpus_prefers_noisy_data_2_over_noisy_data(tmp_path: Path) -> None:
    (tmp_path / "noisy_data 2").mkdir()
    (tmp_path / "noisy_data 2" / "fresh.pdf").write_bytes(b"fresh")
    (tmp_path / "noisy_data").mkdir()
    (tmp_path / "noisy_data" / "stale.pdf").write_bytes(b"stale")

    stage_noise_corpus(root=tmp_path)

    target = tmp_path / "seed" / "docs" / "noisy_data"
    assert [f.name for f in target.iterdir()] == ["fresh.pdf"]


def test_stage_noise_corpus_is_a_noop_once_target_has_files(tmp_path: Path) -> None:
    target = tmp_path / "seed" / "docs" / "noisy_data"
    target.mkdir(parents=True)
    (target / "already-here.pdf").write_bytes(b"x")
    (tmp_path / "noisy_data 2").mkdir()
    (tmp_path / "noisy_data 2" / "ignored.pdf").write_bytes(b"y")

    assert stage_noise_corpus(root=tmp_path) == 0
    assert [f.name for f in target.iterdir()] == ["already-here.pdf"]


def test_stage_noise_corpus_is_a_noop_with_no_staging_folder(tmp_path: Path) -> None:
    assert stage_noise_corpus(root=tmp_path) == 0


def test_real_signal_corpus_has_47_files() -> None:
    """The staged `seed/docs/true_data/` is the 47-file signal set."""
    signal = select_corpus(0).signal
    if not (DOCS_DIR / SIGNAL_SUBDIR).is_dir():
        pytest.skip("signal corpus not staged")
    assert len(signal) == 47
