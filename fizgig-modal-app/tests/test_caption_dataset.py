from pathlib import Path

from caption_dataset import caption_candidates, training_caption, write_caption


def test_caption_candidates_select_only_missing_supported_sidecars(tmp_path: Path):
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.webp"
    ignored = tmp_path / "notes.json"
    first.write_bytes(b"jpg")
    second.write_bytes(b"webp")
    ignored.write_text("{}", encoding="utf-8")
    first.with_suffix(".txt").write_text("existing caption\n", encoding="utf-8")

    assert caption_candidates(tmp_path) == [second]
    assert caption_candidates(tmp_path, overwrite=True) == [first, second]


def test_training_caption_normalizes_and_prefixes_trigger():
    assert training_caption("  a   woman in profile  ", "linda") == (
        "linda, a woman in profile"
    )


def test_write_caption_is_a_matching_sidecar(tmp_path: Path):
    image = tmp_path / "portrait.jpg"
    image.write_bytes(b"jpg")
    caption = write_caption(image, "linda, close-up portrait")
    assert caption == tmp_path / "portrait.txt"
    assert caption.read_text(encoding="utf-8") == "linda, close-up portrait\n"
