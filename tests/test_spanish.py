"""Spanish (Kaikki) lookup tests.

Everything here is offline: the fixture synthesises a tiny JSONL gzip and
streams it through the real Kaikki ingest into a temporary dataset directory.
"""

import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from kotoba import datasets, lookup
from kotoba.cli import EXIT_NO_RESULTS, EXIT_OK, EXIT_USAGE, main
from kotoba.langs import kaikki
from kotoba.langs.spanish import LookupError, Spanish
from kotoba.model import Entry, LookupResult, Sense

DATASET_ID = "kaikki-es"
HOMEPAGE = "https://kaikki.org/dictionary/Spanish/"

SPANISH_ROWS = [
    {
        "word": "hola",
        "lang_code": "es",
        "pos": "intj",
        "senses": [{"glosses": ["hello", "hi"]}],
        "sounds": [{"ipa": "/ˈola/"}],
    },
    {
        "word": "gato",
        "lang_code": "es",
        "pos": "noun",
        "senses": [{"glosses": ["cat"]}],
        "forms": [{"form": "gatos", "tags": ["plural"]}],
    },
    {
        "word": "perro",
        "lang_code": "es",
        "pos": "noun",
        "senses": [{"glosses": ["dog"]}],
    },
    {
        "word": "casa",
        "lang_code": "es",
        "pos": "noun",
        "senses": [{"glosses": ["house"]}],
    },
]

# Must be skipped by the ingest: a lang_marker hit is checked against lang_code.
FRENCH_ROW = {
    "word": "chat",
    "lang_code": "fr",
    "pos": "noun",
    "senses": [{"glosses": ["cat"]}],
}


def _write_jsonl_gz(path: Path, rows: list[dict]) -> Path:
    with gzip.open(path, "wb") as fh:
        for row in rows:
            fh.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


@pytest.fixture
def spanish_data(tmp_path, monkeypatch) -> Path:
    """Ingest a synthetic Kaikki dump into a temp dataset directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    dataset_path = datasets.dataset_dir(DATASET_ID)
    dataset_path.mkdir(parents=True)
    archive = _write_jsonl_gz(tmp_path / "kaikki-es.jsonl.gz", [*SPANISH_ROWS, FRENCH_ROW])
    kaikki.ingest_kaikki(archive, dataset_path, datasets.descriptor(DATASET_ID))
    datasets.write_meta(
        dataset_path,
        {
            "name": DATASET_ID,
            "descriptor_id": DATASET_ID,
            "version": "test",
            "license": "CC BY-SA 4.0 and GFDL",
            "attribution": "synthetic test fixture",
            "source_url": HOMEPAGE,
            "retrieved_at": "2026-01-01T00:00:00Z",
        },
    )
    return dataset_path


@pytest.fixture
def empty_data(tmp_path, monkeypatch) -> Path:
    """Point every XDG root at an empty tree: nothing is installed."""
    data_root = tmp_path / "empty-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return data_root


def _fake_api_result(query: str) -> LookupResult:
    return LookupResult(
        query=query,
        source="es",
        entries=(
            Entry(
                headword=query,
                reading=None,
                senses=(Sense(pos=("api",), glosses=("from-api",)),),
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# ingest


def test_ingest_index_row_counts(spanish_data):
    db = datasets.db_path(spanish_data)
    assert db.exists()
    con = sqlite3.connect(str(db))
    try:
        total = con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        headwords = [row[0] for row in con.execute("SELECT headword FROM entries ORDER BY id")]
        senses = con.execute("SELECT SUM(sense_count) FROM entries").fetchone()[0]
        hola = con.execute(
            "SELECT pos, glosses, sounds FROM entries WHERE headword = 'hola'"
        ).fetchone()
        gato_forms = con.execute("SELECT forms FROM entries WHERE headword = 'gato'").fetchone()[0]
    finally:
        con.close()

    assert total == len(SPANISH_ROWS)
    assert headwords == ["hola", "gato", "perro", "casa"]  # French row skipped
    assert senses == len(SPANISH_ROWS)
    assert hola[0] == "intj"
    assert json.loads(hola[1]) == ["hello", "hi"]
    assert json.loads(hola[2]) == [{"type": "ipa", "value": "/ˈola/"}]
    assert json.loads(gato_forms) == [{"form": "gatos", "tags": ["plural"]}]


# --------------------------------------------------------------------------- #
# search


def test_search_exact_headword(spanish_data):
    result = Spanish().search("hola")

    assert isinstance(result, LookupResult)
    assert result.query == "hola"
    assert result.source == "es"
    assert result.url == HOMEPAGE
    assert [entry.headword for entry in result.entries] == ["hola"]

    entry = result.entries[0]
    assert entry.reading is None
    assert entry.source == "es"
    assert entry.senses[0].pos == ("intj",)
    assert entry.senses[0].glosses == ("hello", "hi")


def test_search_prefix(spanish_data):
    result = Spanish().search("hol")

    assert [entry.headword for entry in result.entries] == ["hola"]
    assert result.entries[0].senses[0].glosses == ("hello", "hi")


def test_search_no_hit_is_empty(spanish_data):
    result = Spanish().search("xyznotaword")

    assert isinstance(result, LookupResult)
    assert result.entries == ()
    assert result.query == "xyznotaword"
    assert result.source == "es"


def test_render_extras_single_ipa(spanish_data):
    result = Spanish().search("hola")
    assert Spanish().render_extras(result.entries[0]) == {"ipa": "/ˈola/"}


def test_search_not_installed(empty_data):
    assert datasets.installed_meta(DATASET_ID) is None
    with pytest.raises(LookupError, match="not installed"):
        Spanish().search("hola")


def test_local_search_not_installed_returns_none(empty_data):
    assert Spanish().local_search("hola") is None


# --------------------------------------------------------------------------- #
# lookup mode matrix


def test_local_only_uses_index(spanish_data, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("api should not be called in local-only mode")

    monkeypatch.setattr(lookup, "_api_lookup", boom)
    result = lookup.lookup("es", "hola", 5, mode="local-only")

    assert [entry.headword for entry in result.entries] == ["hola"]
    assert result.source == "es"


def test_local_first_prefers_index(spanish_data, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("api should not be called when the index hits")

    monkeypatch.setattr(lookup, "_api_lookup", boom)
    result = lookup.lookup("es", "hola", 5, mode="local-first")

    assert [entry.headword for entry in result.entries] == ["hola"]


def test_local_first_falls_back_to_api(spanish_data, monkeypatch):
    calls = []

    def fake_api_search(query, *, limit=None):
        calls.append((query, limit))
        return _fake_api_result(query)

    monkeypatch.setattr(lookup.langs.get("es"), "api_search", fake_api_search)
    result = lookup.lookup("es", "xyznotaword", 3, mode="local-first")

    assert result.entries[0].senses[0].glosses == ("from-api",)
    assert calls == [("xyznotaword", 3)]


def test_api_only_skips_index(spanish_data, monkeypatch):
    calls = []

    def fake_local_search(query, limit=None):
        raise AssertionError("local index should not be queried in api-only mode")

    def fake_api_search(query, *, limit=None):
        calls.append(query)
        return _fake_api_result(query)

    monkeypatch.setattr(lookup.langs.get("es"), "local_search", fake_local_search)
    monkeypatch.setattr(lookup.langs.get("es"), "api_search", fake_api_search)
    result = lookup.lookup("es", "hola", 5, mode="api-only")

    assert result.entries[0].senses[0].glosses == ("from-api",)
    assert calls == ["hola"]


def test_offline_flag_is_local_only(spanish_data, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("api should not be called when offline")

    monkeypatch.setattr(lookup, "_api_lookup", boom)
    result = lookup.lookup("es", "gato", 5, offline=True)

    assert [entry.headword for entry in result.entries] == ["gato"]


def test_local_only_without_dataset_uses_api(empty_data, monkeypatch):
    calls = []

    def fake_api_search(query, *, limit=None):
        calls.append(query)
        return _fake_api_result(query)

    monkeypatch.setattr(lookup.langs.get("es"), "api_search", fake_api_search)
    result = lookup.lookup("es", "hola", 5, mode="local-only")

    assert result.entries[0].senses[0].glosses == ("from-api",)
    assert calls == ["hola"]


# --------------------------------------------------------------------------- #
# CLI


def test_cli_es_print(spanish_data, capsys):
    assert main(["es", "hola", "--print"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "hola" in out
    assert "hello" in out


def test_cli_alias_sp_print(spanish_data, capsys):
    assert main(["sp", "hola", "--print"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "hola" in out
    assert "hello" in out


def test_cli_es_no_results(spanish_data, capsys):
    assert main(["es", "xyznotaword", "--print"]) == EXIT_NO_RESULTS
    assert "no results" in capsys.readouterr().err


def test_cli_unknown_language_is_usage_error(spanish_data, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["xx", "hola", "--print"])

    assert exc.value.code == EXIT_USAGE
    assert "error" in capsys.readouterr().err
