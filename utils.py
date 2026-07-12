"""
utils.py: shared utilities for the LLM systematic-review screening pipeline.
"""
import concurrent.futures
import os
import re
import sys
import json
import shutil
import time
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # pymupdf, fallback for PDFs pymupdf4llm cannot parse
import pymupdf4llm
import requests
from urllib.parse import quote
from markdown import Markdown
from tqdm import tqdm
from transformers import AutoTokenizer
import xmltodict


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def normalize_path(p) -> Path:
    """Expand ~ and normalise separators; return a Path object."""
    p_str = str(p).replace("\\\\", "/").replace("\\", "/")
    return Path(p_str).expanduser()


def add_long_path_prefix(p) -> Path:
    """
    On Windows, prepend the \\?\ prefix so paths longer than 260 chars work.
    No-op on other platforms.
    """
    if sys.platform != "win32":
        return Path(p)
    path = Path(p)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    s = str(path)
    if s.startswith("\\\\?\\"):
        return path
    if s.startswith("\\\\"):
        s = "\\\\?\\UNC\\" + s.lstrip("\\")
    else:
        s = "\\\\?\\" + s
    return Path(s)


def adjust_path(p) -> Path:
    """Expand ~ and on Windows apply the long-path prefix."""
    if os.name == "nt":
        return add_long_path_prefix(normalize_path(p))
    return normalize_path(p)


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def load_xml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return xmltodict.parse(f.read())


def write_xml(data: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(xmltodict.unparse(data, pretty=True))


# ---------------------------------------------------------------------------
# Markdown / text helpers
# ---------------------------------------------------------------------------

def _unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        _unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


Markdown.output_formats["plain"] = _unmark_element
_md = Markdown(output_format="plain")
_md.stripTopLevelTags = False


def unmark(text: str) -> str:
    """Strip markdown formatting, returning plain text."""
    return _md.convert(text)


def clean_markdown(md: str) -> str:
    """Remove non-printable chars, collapse whitespace, limit repeated chars."""
    printable = "".join(ch for ch in md if ch.isprintable())
    printable = re.sub(r"\s+", " ", printable)
    printable = re.sub(r"(.)\1{15,}", r"\1" * 15, printable)
    return printable.strip()


def sanitize_filename(title: str, max_chars: int = 150) -> str:
    """Return a filesystem-safe, length-limited version of a paper title."""
    safe = re.sub(r'[\\/*?:"<>|]', "_", title)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_chars]


# ---------------------------------------------------------------------------
# Token-budget helper
# ---------------------------------------------------------------------------

def truncate_to_token_budget(
    text: str, tokenizer: AutoTokenizer, max_body_tokens: int
) -> str:
    """Return the first *max_body_tokens* tokens of *text*, decoded back to str."""
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) <= max_body_tokens:
        return text
    return tokenizer.decode(
        ids[:max_body_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


# ---------------------------------------------------------------------------
# PDF download helpers
# ---------------------------------------------------------------------------

_HEADERS = {"User-Agent": "llm-systematic-review/1.0 (academic research; contact via GitHub)"}


def _is_valid_pdf(path: str) -> bool:
    """Return True if the file begins with the PDF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def _download_file(url: str, out_path: str) -> bool:
    """Streamed download; returns True if a valid PDF was saved."""
    try:
        r = requests.get(url, stream=True, timeout=60, headers=_HEADERS)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if _is_valid_pdf(out_path):
            return True
        os.remove(out_path)  # discard HTML paywall pages
    except Exception as exc:
        print(f"Download error from {url}: {exc}")
    return False


def get_pdf_via_crossref(doi: str) -> Optional[str]:
    """Ask CrossRef for a direct PDF link."""
    url = f"https://api.crossref.org/works/{quote(doi, safe='/')}"
    try:
        resp = requests.get(url, timeout=30, headers=_HEADERS)
        if resp.status_code != 200:
            return None
        for link in resp.json()["message"].get("link", []):
            if link.get("content-type") == "application/pdf":
                return link.get("URL")
    except Exception as exc:
        print(f"CrossRef error for DOI {doi}: {exc}")
    return None


def get_pdf_via_unpaywall(doi: str, email: str) -> Optional[str]:
    """Query Unpaywall for the best open-access PDF. Requires a contact email."""
    url = f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}?email={quote(email)}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            return best["url_for_pdf"]
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception as exc:
        print(f"Unpaywall error for DOI {doi}: {exc}")
    return None


def get_pdf_via_europepmc(doi: str) -> Optional[str]:
    """Search Europe PMC for an open-access full-text PDF."""
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=DOI:{quote(doi, safe='')}&format=json"
    )
    try:
        resp = requests.get(url, timeout=30, headers=_HEADERS)
        if resp.status_code != 200:
            return None
        for result in resp.json().get("resultList", {}).get("result", []):
            pmcid = result.get("pmcid")
            if pmcid and result.get("isOpenAccess") == "Y":
                return (
                    f"https://europepmc.org/backend/ptpmcrender.fcgi"
                    f"?accid={pmcid}&blobtype=pdf"
                )
    except Exception as exc:
        print(f"Europe PMC error for DOI {doi}: {exc}")
    return None


def get_pdf_via_semantic_scholar(doi: str) -> Optional[str]:
    """Query Semantic Scholar for an open-access PDF."""
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper"
        f"/DOI:{quote(doi, safe='/')}?fields=openAccessPdf"
    )
    try:
        resp = requests.get(url, timeout=30, headers=_HEADERS)
        if resp.status_code != 200:
            return None
        oa = resp.json().get("openAccessPdf") or {}
        return oa.get("url")
    except Exception as exc:
        print(f"Semantic Scholar error for DOI {doi}: {exc}")
    return None


def download_pdfs(
    included_entries: List[dict],
    manual_folder: Path,
    out_folder: Path,
    skip_ids: Optional[List[int]] = None,
    sleep_sec: int = 2,
    unpaywall_email: str = "",
) -> List[dict]:
    """
    Download PDFs for INCLUDE-listed papers.

    Sources tried in order for each paper:
      CrossRef → Unpaywall → Europe PMC → Semantic Scholar → manual fallback

    Parameters
    ----------
    included_entries : list of dicts with keys id, title, authors, doi, year
    manual_folder    : directory where pre-downloaded PDFs can be placed manually
    out_folder       : destination directory for downloaded PDFs
    skip_ids         : list of numeric paper IDs to skip (known-unavailable papers)
    sleep_sec        : pause between requests when a download fails
    unpaywall_email  : contact email for the Unpaywall API (skip if empty)

    Returns
    -------
    List of entries whose PDF could not be obtained.
    """
    skip_ids = set(int(x) for x in (skip_ids or []))
    out_root = add_long_path_prefix(out_folder)
    out_root.mkdir(parents=True, exist_ok=True)

    # Deduplicate by paper ID
    seen, uniq = set(), []
    for e in included_entries:
        if e["id"] not in seen:
            seen.add(e["id"])
            uniq.append(e)

    still_missing: List[dict] = []

    for paper in tqdm(uniq, desc="Downloading PDFs", unit="paper"):
        if int(paper["id"]) in skip_ids:
            tqdm.write(f"[SKIP] {paper['id']}: {paper['title']}")
            continue

        filename = f"{paper['id']} - {sanitize_filename(paper['title'])}.pdf"
        target = out_root / filename

        if target.is_file() and target.stat().st_size > 1024:
            tqdm.write(f"[ALREADY EXISTS] {paper['id']}: {paper['title']}")
            continue

        manual = manual_folder / filename
        if manual.is_file() and manual.stat().st_size > 1024:
            shutil.copy2(manual, target)
            tqdm.write(f"[COPIED FROM MANUAL] {paper['id']}: {paper['title']}")
            continue

        downloaded = False
        doi = paper.get("doi", "") or ""
        if doi and doi != "None":
            sources = [
                ("CROSSREF",  lambda d: get_pdf_via_crossref(d)),
                ("UNPAYWALL", lambda d: get_pdf_via_unpaywall(d, unpaywall_email) if unpaywall_email else None),
                ("EUROPEPMC", lambda d: get_pdf_via_europepmc(d)),
                ("S2",        lambda d: get_pdf_via_semantic_scholar(d)),
            ]
            for label, fn in sources:
                pdf_url = fn(doi)
                if pdf_url and _download_file(pdf_url, str(target)):
                    tqdm.write(f"[OK: {label}] {paper['id']}: {paper['title']}")
                    downloaded = True
                    break

        if downloaded:
            continue

        still_missing.append(
            {
                "id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "doi": paper.get("doi", ""),
                "year": paper.get("year", ""),
                "filename": filename,
                "out_path": str(target),
            }
        )
        tqdm.write(f"[FAIL] {paper['id']}: {paper['title']}")
        time.sleep(sleep_sec)

    if still_missing:
        print("\n=== MANUAL PDF ADDITION REQUIRED ===")
        for e in still_missing:
            print(
                f"  • ID {e['id']}: {e['title']}, "
                f"{e['authors']} ({e['year']}), expected filename: {e['filename']}"
            )
        input(
            f"\nPlace missing PDFs in:\n  {manual_folder}\nor directly in:\n  {out_root}"
            "\nPress <Enter> when done…\n"
        )

        still_missing_after = []
        for e in still_missing:
            target = Path(e["out_path"])
            manual = manual_folder / e["filename"]
            if manual.is_file() and manual.stat().st_size > 1024:
                shutil.copy2(manual, target)
                tqdm.write(f"[COPIED AFTER MANUAL] {e['id']}")
                continue
            if target.is_file() and target.stat().st_size > 1024:
                tqdm.write(f"[MANUAL PLACE OK] {e['id']}")
                continue
            still_missing_after.append(e)
            tqdm.write(f"[STILL MISSING] {e['id']}: {e['title']}")

        still_missing = still_missing_after
        if still_missing:
            print("\nPDFs still missing after manual step:")
            for e in still_missing:
                print(f"  • {e['id']}: {e['title']}")
        else:
            print("\nAll missing PDFs have been found.")
    else:
        print("\nAll PDFs downloaded automatically.")

    return still_missing


# ---------------------------------------------------------------------------
# PDF pre-processing
# ---------------------------------------------------------------------------

def _norm_title(s: str) -> str:
    """Lowercase, strip all non-alphanumeric chars, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def truncate_before_title(text: str, title: str) -> Tuple[str, bool]:
    """
    Return (text_from_title_onward, found).

    Three passes in order of strictness:
      1. Strip markdown bold/header markers, exact match.
      2. Strip all non-alphanumeric characters, exact match.
      3. Normalized title is a substring of the line (handles extra surrounding text).
    """
    if not title:
        return text, False

    lines = text.split("\n")

    # Pass 1: strip markdown formatting only
    target_md = re.sub(r"[*#]+", "", title).strip().lower()
    for i, line in enumerate(lines):
        if re.sub(r"[*#]+", "", line).strip().lower() == target_md:
            return "\n".join(lines[i:]), True

    # Pass 2: strip all punctuation / special characters
    target_norm = _norm_title(title)
    for i, line in enumerate(lines):
        if _norm_title(line) == target_norm:
            return "\n".join(lines[i:]), True

    # Pass 3: normalized title appears inside a longer line
    if len(target_norm) >= 15:
        for i, line in enumerate(lines):
            norm_line = _norm_title(line)
            if len(norm_line) >= 15 and target_norm in norm_line:
                return "\n".join(lines[i:]), True

    return text, False


def remove_references(text: str) -> str:
    """Strip everything from the first References-like heading onward."""
    patterns = [
        r"^\s*[#=]*\s*(References|Bibliography|Literature\s+Cited|List\s+of\s+Sources|Sources?)\s*$",
        r"^\s*\*\*(References|Bibliography)\*\*\s*$",
        r"^\s*(?:REF|References|Biblio):?\s*$",
        r"^\s*Reference\(s\).*$",
    ]
    lines = text.split("\n")
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.match(pat, line, re.IGNORECASE):
                return "\n".join(lines[:i])
    return text


def preprocess_pdfs(
    pdf_dir: Path,
    included_json: str,
    short_json_out: str,
    tokenizer_name: str = "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    max_total_tokens: int = 131_072,
    reserved_prompt_tokens: int = 5_000,
    debug: bool = False,
) -> Tuple[dict, dict]:
    """
    Extract, clean, and token-limit each PDF in *pdf_dir*.

    Steps:
      1. Try pymupdf4llm; on failure fall back to fitz page-by-page extraction.
         If both fail, prompt the user to replace the file or press S to skip.
      2. Strip text before the paper title (three-pass fuzzy matching).
      3. Remove the References section.
      4. Truncate body to (max_total_tokens - reserved_prompt_tokens) tokens.
         If the title was not found and the text exceeds the budget, the entry
         is skipped, since the relevant section cannot be reliably identified.

    Returns (short_texts, long_texts).  long_texts is always empty; it is
    kept only to preserve the original API.
    """
    with open(included_json, "r", encoding="utf-8") as f:
        title_by_id = {e["id"]: e["title"] for e in json.load(f)}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_body_tokens = max_total_tokens - reserved_prompt_tokens
    if max_body_tokens <= 0:
        raise ValueError("reserved_prompt_tokens exceeds max_total_tokens")

    short_texts: dict = {}
    long_texts: dict = {}
    pdf_root = Path(pdf_dir).resolve()

    for file in tqdm(sorted(pdf_root.iterdir()), desc="PDF preprocessing"):
        if file.suffix.lower() != ".pdf":
            continue
        try:
            entry_id, _ = file.stem.split(" - ", 1)
        except ValueError:
            if debug:
                tqdm.write(f"[WARN] Unexpected filename format: {file.name}")
            continue

        raw_md = None
        while True:
            try:
                raw_md = pymupdf4llm.to_markdown(str(add_long_path_prefix(file)))
                break
            except Exception as exc:
                tqdm.write(f"\n[ERROR] Cannot read PDF {file.name}: {exc}")
                tqdm.write("[FALLBACK] Trying page-by-page text extraction (120s timeout)...")

                def _fitz_extract(path: str) -> Optional[str]:
                    doc = fitz.open(path)
                    pages: list = []
                    for n in range(len(doc)):
                        try:
                            pages.append(doc[n].get_text())
                        except Exception:
                            pass
                    doc.close()
                    return "\n".join(pages) if pages else None

                fitz_result: Optional[str] = None
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(_fitz_extract, str(add_long_path_prefix(file)))
                        fitz_result = fut.result(timeout=120)
                except concurrent.futures.TimeoutError:
                    tqdm.write("[ERROR] Fallback timed out after 120s.")
                except Exception as exc2:
                    tqdm.write(f"[ERROR] Fallback also failed: {exc2}")

                if fitz_result:
                    raw_md = fitz_result
                    tqdm.write(f"[FALLBACK OK] Extracted text from {file.name}")
                    break

                print(
                    "\nPlace a valid PDF with this exact filename:\n"
                    f"    {file.name}\n"
                    f"in:\n    {add_long_path_prefix(pdf_root)}\n"
                    "Then press <Enter> to retry, or 'S' + <Enter> to skip this PDF."
                )
                choice = input(">>> ").strip().upper()
                if choice == "S":
                    tqdm.write(f"[SKIP] {file.name} skipped, no readable PDF available.")
                    break

        if raw_md is None:
            continue

        text = clean_markdown(raw_md)
        text, title_found = truncate_before_title(text, title_by_id.get(entry_id, ""))
        text = remove_references(text)

        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        if not title_found and n_tokens > max_body_tokens:
            tqdm.write(
                f"[SKIP] {entry_id}: title not found and text exceeds token budget "
                f"({n_tokens:,} tokens), cannot reliably extract the relevant section."
            )
            continue

        if n_tokens > max_body_tokens:
            text = truncate_to_token_budget(text, tokenizer, max_body_tokens)
            if debug:
                tqdm.write(
                    f"[TRUNCATED] {entry_id}: {n_tokens:,} → "
                    f"{len(tokenizer.encode(text, add_special_tokens=False)):,} tokens"
                )

        short_texts[entry_id] = {
            "title": title_by_id.get(entry_id, file.stem),
            "prompt": text,
            "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
        }

    with open(short_json_out, "w", encoding="utf-8") as f:
        json.dump(short_texts, f, indent=2, ensure_ascii=False)

    print(f"\nPre-processing done. Short texts: {len(short_texts)}")
    return short_texts, long_texts


# ---------------------------------------------------------------------------
# Folder helper
# ---------------------------------------------------------------------------

def clear_folder(folder) -> None:
    """Delete all files and sub-directories inside *folder* (keep the folder itself)."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    for entry in folder.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)
        except Exception as exc:
            print(f"Warning: could not delete {entry}: {exc}")


# ---------------------------------------------------------------------------
# Final-includes export
# ---------------------------------------------------------------------------

def export_included(
    entries: dict,
    consensus_key: str,
    pdf_folder: Path,
    out_folder: Path,
    run_idx: int,
) -> None:
    """
    Print a numbered list of finally included papers and copy their PDFs to
    out_folder/included_pdfs/run{run_idx}/.

    Parameters
    ----------
    entries       : parsed XML dict (from load_xml)
    consensus_key : entry field that holds the final decision (e.g. "B2-CR_fulltext_consensus")
    pdf_folder    : folder where the run's downloaded PDFs live
    out_folder    : base output folder (xml_output_base parent)
    run_idx       : 1-based run number, used to name the subfolder
    """
    included = [
        e for e in entries["root"]["entry"]
        if e.get(consensus_key, "").split(" - ")[0].upper() == "INCLUDE"
    ]

    print(f"\n{'='*60}")
    print(f"FINAL INCLUDES: Run {run_idx}  ({len(included)} paper{'s' if len(included) != 1 else ''})")
    print(f"{'='*60}")
    for i, e in enumerate(included, 1):
        print(
            f"{i:>3}. [{e.get('id', '?')}] {e.get('title', '?')}\n"
            f"       {e.get('authors', '?')} ({e.get('year', '?')}), DOI: {e.get('doi', 'n/a')}"
        )
    print(f"{'='*60}\n")

    dest_dir = add_long_path_prefix(Path(out_folder) / "included_pdfs" / f"run{run_idx}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for e in included:
        filename = f"{e['id']} - {sanitize_filename(e.get('title', str(e['id'])))}.pdf"
        src = add_long_path_prefix(Path(pdf_folder) / filename)
        if src.is_file():
            shutil.copy2(src, dest_dir / filename)
            copied += 1
        else:
            print(f"[WARN] PDF not found for ID {e['id']}: {filename}")

    print(f"Copied {copied}/{len(included)} PDFs to:\n  {dest_dir}\n")
