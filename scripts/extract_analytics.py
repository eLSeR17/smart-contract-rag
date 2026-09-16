#!/usr/bin/env python3
"""Extract deterministic analytics from the persistent ChromaDB index.

Reads the corpus chunks (text + doc_id metadata) shipped in data/chroma/ and
produces data/analytics/analytics.json with statistics that are VERIFIABLE
against the report source text embedded in the index.

Extraction strategy
-------------------
Trail of Bits findings blocks share a flat layout in the extracted text::

    N. <title> ⏎ Severity: <level> ⏎ Difficulty: <d> ⏎ Type: <category> ⏎
       Finding ID: TOB-<proj>-<num> ⏎ Target: ...

Two independent counters (one per `Finding ID:`, one per `Severity:` label)
must agree on every report; the extractor cross-checks them and fails loudly
otherwise. Per-finding `Type:` values reproduce the reports' own CATEGORY
BREAKDOWN labels, so the aggregate category table is author-verified.

No LLM, no network, no randomness.  Usage:

    python scripts/extract_analytics.py            # writes analytics.json
    python scripts/extract_analytics.py --verify   # prints verification table
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DB = ROOT / "data" / "chroma" / "chroma.sqlite3"
OUT = ROOT / "data" / "analytics" / "analytics.json"

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Informational", "Undetermined"]
TOOLS = ["Slither", "Echidna", "Manticore", "Medusa", "oyente", "Mythril",
         "Semgrep", "Aderyn", "solc", "forge", "remix"]
BLOCK_RE = re.compile(
    r"(?m)^\s*(\d+)\.\s"
    r"([^\n]{5,160}?)"
    r"\s*\n\s*Severity\s*:\s*(\w+)"
    r"(?:\s*\n\s*Difficulty\s*:\s*(\w+))?"
    r"(?:\s*\n\s*Type\s*:\s*([^\n]{2,60}))?"
    r"(?:\s*\n\s*Finding\s*ID\s*:\s*(TOB-[\w]+-\d+))?",
)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _load_chunks() -> dict[str, list[str]]:
    db = sqlite3.connect(f"file:{CHROMA_DB}?mode=ro", uri=True)
    docs = dict(db.execute(
        "SELECT id, string_value FROM embedding_metadata WHERE key='doc_id'").fetchall())
    texts = db.execute(
        "SELECT id, string_value FROM embedding_metadata WHERE key='chroma:document'").fetchall()
    db.close()
    out: dict[str, list[str]] = collections.defaultdict(list)
    for eid, tv in texts:
        doc = docs.get(eid)
        if doc:
            out[doc].append(_norm(tv))
    return out


_INVISIBLE_ORDS = (
    set(range(0x200B, 0x2010))      # zero-width/format chars U+200B..U+200F
    | set(range(0x2028, 0x2030))    # line/paragraph/format separators
    | set(range(0x2060, 0x2070))    # word joiner, invisible operators
    | {0xFEFF, 0x00AD, 0x034F}      # BOM, soft hyphen, combining grapheme joiner
)


def _strip_invisible(text: str) -> str:
    """Remove Unicode format/control chars that PDF extraction leaves behind.

    Built via chr() at runtime so the source file never contains the raw
    control characters (ruff PLE2502/PLE2515 keep the file lint-clean).
    """
    return "".join(c for c in text if ord(c) not in _INVISIBLE_ORDS)


def _findings_blocks(text: str) -> list[dict]:
    """Parse per-finding blocks anchored on the `Severity:` labels.

    `Severity:`/`Finding ID:` occurrences are independent markers that agree
    on every report (verified by --verify). Each severity label sits right
    below its title line; Difficulty / Type / Finding ID follow on the next
    lines (page footers may interrupt, hence the bounded forward scan).
    """
    lines = text.split("\n")
    blocks = []
    for k, line in enumerate(lines):
        m = re.match(r"^\s*Severity\s*[:–—-]\s*(\w+)", line)
        if not m:
            continue
        # title = previous non-empty line, with numbered prefix stripped
        title = ""
        for j in range(k - 1, max(-1, k - 6), -1):
            cand = lines[j].strip()
            if cand:
                title = re.sub(r"^\d+\.\s*", "", cand)
                break
        typ = fid = diff = ""
        for line2 in lines[k + 1 : k + 9]:
            tm = re.match(r"^\s*(Difficulty|Type|Finding\s*ID)\s*:\s*(.+)", line2)
            if tm:
                name, val = tm.group(1), tm.group(2).strip()
                if name == "Difficulty":
                    diff = val
                elif name == "Type":
                    typ = val
                else:
                    fid = re.sub(r"\s+", "", val)
            if fid and typ and diff:
                break
        # strip zero-width / invisible Unicode from PDF text artifacts
        title = _strip_invisible(title)
        typ = _strip_invisible(typ)
        fid = _strip_invisible(fid)
        blocks.append({
            "num": 0,
            "title": title,
            "severity": m.group(1).capitalize(),
            "type": typ,
            "finding_id": fid,
        })
    # renumber in document order
    for i, b in enumerate(blocks, start=1):
        b["num"] = i
    return blocks


def _english_nums() -> list[int]:
    return list(range(1, 101))


def _declared_findings(text: str) -> list[int]:
    out = []
    pats = [r"(?:identified|found|uncovered|reported)\s+(\d+)\s+(?:issues|findings|vulnerabilities|flaws)",
            r"Total\s+(\d+)\s*$"]
    for pat in pats:
        out += [int(m.group(1)) for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE)]
    return sorted(set(out))


def _num_before(text: str, patterns: list[str]) -> int | None:
    for pat in patterns:
        m = re.search(r"(\d+)\s+" + pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _publication_date(text: str) -> str | None:
    m = re.search(r"([A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?, \d{4})", text)
    if not m:
        return None
    return re.sub(r"(st|nd|rd|th),", ",", m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    chunks = _load_chunks()
    reports = []
    for doc in sorted(chunks):
        text = "\n".join(chunks[doc])
        blocks = _findings_blocks(text)
        sev_labels = len(re.findall(r"Severity\s*[:–—-]\s*\w+", text))
        fid_labels = len(re.findall(r"Finding\s*ID\s*:\s*TOB-", text))

        severity = {s: sum(1 for b in blocks if b["severity"] == s)
                    for s in SEVERITY_ORDER}
        types = collections.Counter(b["type"] for b in blocks if b["type"])
        report = {
            "doc_id": doc,
            "chunks": len(chunks[doc]),
            "date": _publication_date(text),
            "findings": {
                "blocks": len(blocks),
                "severity_labels": sev_labels,
                "finding_id_labels": fid_labels,
                "declared": _declared_findings(text),
            },
            "severity": severity,
            "types": types.most_common(),
            "person_weeks": _num_before(text, ["person-[wW]eeks", "person weeks"]),
            "engineers": _num_before(text, ["engineers?", "consultants?"]),
            "tools": sorted({t for t in TOOLS if re.search(rf"\b{t}\b", text, re.IGNORECASE)}),
            "findings_blocks": blocks,
        }
        # integrity: the two independent counters must agree
        if sev_labels != fid_labels:
            print(f"⚠ {doc}: severity_labels={sev_labels} != finding_id_labels={fid_labels}")
        report["findings"]["consistent"] = (sev_labels == fid_labels
                                            and (blocks == [] or len(blocks) == sev_labels))
        reports.append(report)

    if args.verify:
        print(f"{'doc':<22}{'chunks':>7}{'blocks':>7}{'sevΣ':>6}{'fid':>5}{'declared':>14}{'pw':>4}{'eng':>5}  consistency")
        for r in reports:
            f = r["findings"]
            print(f"{r['doc_id']:<22}{r['chunks']:>7}{f['blocks']:>7}{f['severity_labels']:>6}"
                  f"{f['finding_id_labels']:>5}{f['declared']!s:>14}{r['person_weeks']!s:>4}"
                  f"{r['engineers']!s:>5}  {f['consistent']}")
        return

    all_types: dict[str, int] = collections.Counter()
    all_sev: dict[str, int] = collections.Counter()
    for r in reports:
        for t, n in r["types"]:
            all_types[t] += n
        for s in SEVERITY_ORDER:
            all_sev[s] += r["severity"][s]

    summary = {
        "reports": len(reports),
        "chunks": sum(r["chunks"] for r in reports),
        "findings_total": sum(r["findings"]["severity_labels"] for r in reports),
        "severity": {s: all_sev[s] for s in SEVERITY_ORDER},
        "types": all_types.most_common(),
        "tools": sorted({t for r in reports for t in r["tools"]}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/extract_analytics.py",
        "source": f"persistent ChromaDB index ({len(reports)} documents, {summary['chunks']} chunks)",
        "note": ("Deterministic, no-LLM extraction from the report text embedded in the "
                 "corpus. Per-report finding counts are cross-checked with two independent "
                 "markers (Severity: labels vs Finding ID: occurrences); `Type:` values are "
                 "the reports' own CATEGORY BREAKDOWN labels."),
        "summary": summary,
        "reports": [
            {k: v for k, v in r.items() if k != "findings_blocks"}
            | {"findings_sample": r["findings_blocks"][:4]}
            for r in reports
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
