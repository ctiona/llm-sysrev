"""
B1 Framework: single-agent iterative screening framwork.

Architecture
------------
  One agent performs title/abstract screening (INCLUDE / EXCLUDE / UNSURE).
  UNSURE decisions are re-screened to achieve a binary INCLUDE / EXCLUDE decision (forced screening).
  Included papers are downloaded, pre-processed, and then screened at full-text level
  after a validity check.

Usage
-----
  Set environment variables:
    OPENAI_BASE_URL=<your endpoint>
    OPENAI_API_KEY=<your key>

  Run:
    python B1/B1.py --config B1/config.yaml
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Tuple

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) #TODO: is this needed when generalized?
import agent_setup
from agent_setup import Agent
from utils import (
    adjust_path,
    add_long_path_prefix,
    download_pdfs,
    export_included,
    load_xml,
    preprocess_pdfs,
    unmark,
    write_xml,
)


# ---------------------------------------------------------------------------
# Title / abstract screening
# ---------------------------------------------------------------------------

def screen_titles_parallel(
    entries: dict, cfg: dict, max_workers: int = 10
) -> Tuple[int, int, int]:
    """Run T&A screening in parallel; return (include, exclude, unsure) counts."""
    inclusions = exclusions = unsure = 0
    total = len(entries["root"]["entry"])
    gen_kwargs = cfg.get("generation_kwargs", {})

    def process(entry: dict) -> str:
        agent = Agent(
            model=cfg["model"],
            background="Basic",
            task="T&A_Screening",
            criteria="t&a_Include/Exclude/Unsure",
            output_format="T&A_Screening_Include/Exclude/Unsure",
            **gen_kwargs,
        )
        prompt = (
            f"title: {entry['title']}\n"
            f"year: {entry['year']}\n"
            f"journal: {entry['journal']}\n"
            f"authors: {entry['authors']}\n"
            f"abstract: {entry['abstract']}"
        )
        _, decision = agent.reply_to(prompt)
        entry["B1_decision"] = decision
        return decision

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=total, desc="T&A Screening", unit="entry", dynamic_ncols=True) as pbar:
        futures = {executor.submit(process, e): e for e in entries["root"]["entry"]}
        for fut in concurrent.futures.as_completed(futures):
            try:
                decision = fut.result()
                part = decision.split(" - ", 1)[0]
                if part == "INCLUDE":
                    inclusions += 1
                elif part == "EXCLUDE":
                    exclusions += 1
                else:
                    unsure += 1
            except Exception as exc:
                pbar.set_postfix_str(f"ERR {exc}", refresh=True)
            pbar.update(1)
            pbar.set_postfix({"Incl": inclusions, "Excl": exclusions, "Unsure": unsure}, refresh=True)

    print(f"\nT&A results: INCLUDE: {inclusions}, EXCLUDE: {exclusions}, UNSURE: {unsure}")
    return inclusions, exclusions, unsure


# ---------------------------------------------------------------------------
# Forced screening (UNSURE entries only)
# ---------------------------------------------------------------------------

def forced_screen(entries: dict, cfg: dict) -> Tuple[int, int]:
    """Re-screen UNSURE entries with a binary INCLUDE/EXCLUDE prompt."""
    incl = excl = 0
    total = len(entries["root"]["entry"])
    gen_kwargs = cfg.get("generation_kwargs", {})

    def process(entry: dict) -> str:
        if not entry.get("B1_decision", "").upper().startswith("UNSURE"):
            return entry.get("B1_decision", "")
        agent = Agent(
            model=cfg["model"],
            background="Basic",
            task="T&A_Screening",
            criteria="t&a_Include/Exclude",
            output_format="T&A_Screening_Include/Exclude",
            **gen_kwargs,
        )
        prompt = (
            f"title: {entry['title']}\n"
            f"year: {entry['year']}\n"
            f"journal: {entry['journal']}\n"
            f"authors: {entry['authors']}\n"
            f"abstract: {entry['abstract']}"
        )
        _, forced = agent.reply_to(prompt)
        entry["B1_forced"] = forced
        return forced

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.get("max_workers", 10)) as executor, \
            tqdm(total=total, desc="Forced Screening", unit="entry", dynamic_ncols=True) as pbar:
        futures = {executor.submit(process, e): e for e in entries["root"]["entry"]}
        for fut in concurrent.futures.as_completed(futures):
            decision = fut.result() or ""
            part = decision.split(" - ", 1)[0]
            if part == "INCLUDE":
                incl += 1
            elif part == "EXCLUDE":
                excl += 1
            pbar.update(1)
            pbar.set_postfix({"Incl": incl, "Excl": excl}, refresh=True)

    print(f"\nForced screening: INCLUDE: {incl}, EXCLUDE: {excl}")
    return incl, excl


# ---------------------------------------------------------------------------
# Full-text validity check
# ---------------------------------------------------------------------------

def validate_fulltexts(
    short_texts: dict, entries_xml: dict, cfg: dict
) -> Tuple[int, int, int]:
    """Check whether each extracted PDF text is usable; return (valid, invalid, unclear)."""
    valid = invalid = unclear = 0
    gen_kwargs = cfg.get("generation_kwargs", {})

    def process_one(pmid: str, entry: dict) -> str:
        agent = Agent(
            model=cfg["model"],
            background="Basic",
            task="Validity_Check",
            criteria="Validity_Check",
            output_format="Validity_Check",
            **gen_kwargs,
        )
        prompt = f"title: {entry['title']}\ntext: {entry['prompt']}\n"
        _, decision = agent.reply_to(prompt)
        decision = unmark(decision)
        for e in entries_xml["root"]["entry"]:
            if str(e.get("id")) == pmid:
                e["B1_fulltext_validity"] = decision
                break
        return decision.split(" -", 1)[0].strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor, \
            tqdm(total=len(short_texts), desc="Validity check", unit="paper", dynamic_ncols=True) as pbar:
        futures = {executor.submit(process_one, pid, val): pid for pid, val in short_texts.items()}
        for fut in concurrent.futures.as_completed(futures):
            try:
                cat = fut.result()
                if cat == "VALID":
                    valid += 1
                elif cat == "INVALID":
                    invalid += 1
                else:
                    unclear += 1
            except Exception as exc:
                pbar.set_postfix_str(f"ERR {exc}", refresh=True)
            pbar.update(1)
            pbar.set_postfix({"Valid": valid, "Invalid": invalid, "Unclear": unclear}, refresh=True)

    print(f"\nValidity: VALID: {valid}, INVALID: {invalid}, UNCLEAR: {unclear}")
    return valid, invalid, unclear


# ---------------------------------------------------------------------------
# Full-text screening
# ---------------------------------------------------------------------------

def screen_fulltexts(
    short_texts: dict, entries_xml: dict, cfg: dict
) -> Tuple[int, int, int]:
    """Screen each valid full text; return (include, exclude, unclear) counts."""
    incl = excl = unclear = 0
    gen_kwargs = cfg.get("generation_kwargs", {})

    def process_one(pmid: str, data: dict) -> str:
        for e in entries_xml["root"]["entry"]:
            if str(e.get("id")) == pmid and \
                    e.get("B1_fulltext_validity", "").split(" - ")[0] == "INVALID":
                decision = "EXCLUDE - Invalid paper"
                e["B1_fulltext_decision"] = decision
                return decision

        agent = Agent(
            model=cfg["model"],
            background="Basic",
            task="Fulltext_Screening",
            criteria="fulltext_Include/Exclude",
            output_format="fulltext_Include/Exclude",
            **gen_kwargs,
        )
        prompt = f"title: {data['title']}\ntext: {data['prompt']}"
        _, decision = agent.reply_to(prompt)
        decision = unmark(decision)
        for e in entries_xml["root"]["entry"]:
            if str(e.get("id")) == pmid:
                e["B1_fulltext_decision"] = decision
                break
        return decision

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor, \
            tqdm(total=len(short_texts), desc="Full-text screening", unit="paper", dynamic_ncols=True) as pbar:
        futures = {executor.submit(process_one, pid, val): pid for pid, val in short_texts.items()}
        for fut in concurrent.futures.as_completed(futures):
            try:
                decision = fut.result() or ""
                first = decision.split(" -", 1)[0].strip()
                if first == "INCLUDE":
                    incl += 1
                elif first == "EXCLUDE":
                    excl += 1
                else:
                    unclear += 1
            except Exception as exc:
                pbar.set_postfix_str(f"ERR {exc}", refresh=True)
            pbar.update(1)
            pbar.set_postfix({"Incl": incl, "Excl": excl, "Unclear": unclear}, refresh=True)

    print(f"\nFull-text screening: INCLUDE: {incl}, EXCLUDE: {excl}, UNCLEAR: {unclear}")
    return incl, excl, unclear


# ---------------------------------------------------------------------------
# One complete cycle
# ---------------------------------------------------------------------------

def run_one_cycle(cycle_idx: int, cfg: dict) -> dict:
    print(f"\n=== RUN {cycle_idx + 1} / {cfg['n_cycles']} ===\n")

    entries = load_xml(cfg["xml_input"])

    screen_titles_parallel(entries, cfg=cfg, max_workers=cfg["max_workers"])
    forced_screen(entries, cfg=cfg)

    included = [
        {"title": e["title"], "authors": e["authors"], "doi": e.get("doi", ""), "id": e["id"], "year": e.get("year", "")}
        for e in entries["root"]["entry"]
        if e.get("B1_decision", "").lower().startswith("include")
        or e.get("B1_forced", "").lower().startswith("include")
    ]
    with open(cfg["included_json"], "w", encoding="utf-8") as f:
        json.dump(included, f, indent=4, ensure_ascii=False)

    download_pdfs(
        included,
        manual_folder=Path(cfg["manual_folder"]),
        out_folder=Path(cfg["pdf_folder"]),
        skip_ids=cfg.get("skip_ids", []),
        sleep_sec=cfg["sleep_between_downloads"],
        unpaywall_email=cfg.get("unpaywall_email", ""),
    )

    short_texts, _ = preprocess_pdfs(
        pdf_dir=Path(cfg["pdf_folder"]),
        included_json=cfg["included_json"],
        short_json_out=cfg["short_texts_json"],
        tokenizer_name=cfg["tokenizer"],
        max_total_tokens=cfg.get("max_total_tokens", 131_072),
        reserved_prompt_tokens=cfg.get("reserved_prompt_tokens", 5_000),
    )

    included_ids = {
        str(e["id"])
        for e in entries["root"]["entry"]
        if e.get("B1_decision", "").lower().startswith("include")
        or e.get("B1_forced", "").lower().startswith("include")
    }
    short_texts = {k: v for k, v in short_texts.items() if k in included_ids}

    valid_cnt, invalid_cnt, valid_unclear_cnt = validate_fulltexts(short_texts, entries, cfg)
    incl_cnt, excl_cnt, ft_unclear_cnt = screen_fulltexts(short_texts, entries, cfg)

    out_xml = add_long_path_prefix(f"{cfg['xml_output_base']}_run{cycle_idx + 1}.xml")
    write_xml(entries, str(out_xml))
    print(f"\nRun {cycle_idx + 1} done, output: {out_xml}")

    export_included(
        entries,
        consensus_key="B1_fulltext_decision",
        pdf_folder=Path(cfg["pdf_folder"]),
        out_folder=Path(cfg["xml_output_base"]).parent,
        run_idx=cycle_idx + 1,
    )

    return {
        "run": cycle_idx + 1,
        "valid_papers": valid_cnt,
        "invalid_papers": invalid_cnt,
        "unclear_validity": valid_unclear_cnt,
        "fulltext_includes": incl_cnt,
        "fulltext_excludes": excl_cnt,
        "fulltext_unclear": ft_unclear_cnt,
        "xml_path": str(out_xml),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B1: single-agent systematic review framework")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    agent_setup.load_syscontext(raw.get("syscontext", "syscontext"))

    for key in ("xml_input", "xml_output_base", "pdf_folder", "manual_folder"):
        raw[key] = str(adjust_path(raw[key]))

    Path(raw["xml_output_base"]).parent.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for i in range(raw["n_cycles"]):
        all_stats.append(run_one_cycle(i, raw))

    with open(raw["summary_path"], "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print("\nAll runs completed. Summary:", raw["summary_path"])


if __name__ == "__main__":
    main()
