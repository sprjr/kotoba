"""Spanish lookups backed by the local Kaikki (Wiktextract) index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from kotoba import datasets
from kotoba.langs.base import Language
from kotoba.model import Entry, LookupResult, Sense

DATASET_ID = "kaikki-es"
DEFAULT_LIMIT = 12

_COLUMNS = "id, headword, lemma, pos, glosses, forms, tags, sense_count"


class LookupError(Exception):
    pass


def _index_path() -> Path:
    dataset_path = datasets.dataset_dir(DATASET_ID)
    path = datasets.db_path(dataset_path)
    if datasets.installed_meta(DATASET_ID) is None or not path.exists():
        raise LookupError("spanish dataset not installed")
    return path


def _connect(*, required: bool) -> sqlite3.Connection | None:
    try:
        path = _index_path()
    except (LookupError, datasets.DictError):
        if required:
            raise
        return None
    return sqlite3.connect(str(path))


def _homepage() -> str | None:
    try:
        return datasets.descriptor(DATASET_ID).homepage or None
    except datasets.DictError:
        return None


def _like_prefix(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _search_ids(con: sqlite3.Connection, query: str, limit: int) -> list[int]:
    tiers = (
        ("headword = ? COLLATE NOCASE", (query,)),
        ("lemma = ? COLLATE NOCASE", (query,)),
        (r"headword LIKE ? COLLATE NOCASE ESCAPE '\'", (_like_prefix(query),)),
    )
    ordered: list[int] = []
    seen: set[int] = set()
    for clause, params in tiers:
        if len(ordered) >= limit:
            break
        rows = con.execute(
            f"SELECT id FROM entries WHERE {clause} ORDER BY sense_count DESC, id ASC LIMIT ?",
            (*params, limit),
        ).fetchall()
        for (row_id,) in rows:
            if row_id in seen:
                continue
            seen.add(row_id)
            ordered.append(row_id)
    return ordered[:limit]


def _entry_from_row(row: tuple, homepage: str | None) -> Entry | None:
    headword, pos, glosses_raw, tags_raw = row[1], row[3], row[4], row[6]
    try:
        glosses = json.loads(glosses_raw or "[]")
    except (TypeError, json.JSONDecodeError):
        glosses = []
    if not glosses:
        return None
    try:
        tags = json.loads(tags_raw or "[]")
    except (TypeError, json.JSONDecodeError):
        tags = []
    if isinstance(tags, str):
        tags = [tags]
    sense = Sense(
        pos=(pos or "",) if pos else (),
        glosses=tuple(str(gloss) for gloss in glosses),
    )
    return Entry(
        headword=headword or "",
        reading=None,
        senses=(sense,),
        tags=tuple(str(tag) for tag in tags),
        url=homepage or "",
        source="es",
    )


class Spanish(Language):
    code = "es"
    name = "Spanish"
    aliases = ("sp", "spanish", "español", "espanol")
    language_code = "es"

    def datasets(self) -> list[str]:
        return [DATASET_ID]

    def local_search(self, query: str, limit: int | None = None) -> LookupResult | None:
        cap = limit or DEFAULT_LIMIT
        try:
            con = _connect(required=False)
        except LookupError:
            raise
        if con is None:
            raise LookupError("spanish dataset not installed")
        homepage = _homepage()
        try:
            ids = _search_ids(con, query, cap)
            entries: list[Entry] = []
            for row_id in ids:
                row = con.execute(
                    f"SELECT {_COLUMNS} FROM entries WHERE id = ?", (row_id,)
                ).fetchone()
                if row is None:
                    continue
                entry = _entry_from_row(row, homepage)
                if entry is not None:
                    entries.append(entry)
        finally:
            con.close()

        return LookupResult(
            query=query,
            source=self.code,
            entries=tuple(entries),
            url=homepage,
        )

    def search(self, query: str, *, limit: int | None = None, offline: bool = False) -> LookupResult:
        result = self.local_search(query, limit)
        if result is None:
            raise LookupError("spanish dataset not installed")
        return result

    def api_search(self, query: str, *, limit: int | None = None) -> LookupResult:
        return self.search(query, limit=limit)
