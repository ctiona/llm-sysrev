"""
B2-CR Framework: single basic agent with conflict resolution.

Architecture
------------
  A single "Basic" agent performs title/abstract screening twice per paper
  (acting as two independent reviewers).  Where the two decisions disagree or
  either votes UNSURE, a multi-turn conversation is performed until consensus
  is reached.  The same two-pass pattern is applied at full-text level after
  a validity check.

Usage
-----
  Set environment variables:
    OPENAI_BASE_URL=<your endpoint>
    OPENAI_API_KEY=<your key>

  Run:
    python B2-CR/B2-CR.py --config B2-CR/config.yaml
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_setup
from agent_setup import Agent
from utils import (
    adjust_path,
    add_long_path_prefix,
    clear_folder,
    download_pdfs,
    export_included,
    load_xml,
    preprocess_pdfs,
    unmark,
    write_xml,
)


# ---------------------------------------------------------------------------
# Title / abstract screening: two random agents per entry
# ---------------------------------------------------------------------------

def screen_titles_parallel(
    entries: dict, cfg: dict, max_workers: int = 10
) -> None:
    """
    For each entry run T&A screening twice with the same Basic agent,
    storing decisions as reviewer_1_decision and reviewer_2_decision.
    """
    gen_kwargs = cfg.get("generation_kwargs", {})

    def process(entry: dict) -> None:
        for reviewer_id in ("reviewer_1", "reviewer_2"):
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
            entry[f"{reviewer_id}_decision"] = decision

    total = len(entries["root"]["entry"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=total, desc="T&A Screening", unit="entry", dynamic_ncols=True) as pbar:
        futures = {executor.submit(process, e): e for e in entries["root"]["entry"]}
        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)


# ---------------------------------------------------------------------------
# Conflict / UNSURE resolution
# ---------------------------------------------------------------------------

def resolve_conflicts(
    entries: dict,
    decision_prefix: str,
    consensus_key: str,
    short_texts: Optional[dict],
    cfg: dict,
    max_rounds: int = 10,
) -> dict:
    """
    For every entry where the two independent decisions disagree or contain UNSURE,
    run a multi-turn conversation between the two agents until they reach consensus.

    Parameters
    ----------
    decision_prefix : prefix of the two decision fields (e.g. "" for T&A,
                      "B2-CR_fulltext_" for full-text)
    consensus_key   : XML field name to store the final decision
    short_texts     : pre-processed PDF texts (required for full-text conflicts)
    """
    gen_kwargs = cfg.get("generation_kwargs", {})
    transcript_store: dict = {}
    unresolved = []

    for entry in tqdm(entries["root"]["entry"], desc="Resolving conflicts", unit="entry"):
        decision_fields = [
            k for k in entry
            if k.startswith(decision_prefix) and k.endswith("_decision")
        ]
        if len(decision_fields) != 2:
            continue

        dec_a = entry[decision_fields[0]]
        dec_b = entry[decision_fields[1]]

        need_resolve = (
            dec_a.split(" - ")[0] != dec_b.split(" - ")[0]
            or "UNSURE" in dec_a.upper()
            or "UNSURE" in dec_b.upper()
        )
        if not need_resolve:
            entry[consensus_key] = dec_a.split(" - ")[0]
            continue

        reviewer_names = [
            k[len(decision_prefix):-len("_decision")] for k in decision_fields
        ]
        criteria_key = (
            "t&a_Include/Exclude/Unsure" if decision_prefix == "" else "fulltext_Include/Exclude"
        )

        reviewers = [
            Agent(
                model=cfg["model"],
                background="Basic",
                task="Conflict_resolving",
                criteria=criteria_key,
                output_format="Conflict_resolving",
                **gen_kwargs,
            )
            for _ in range(2)
        ]

        if decision_prefix == "":
            transcript = (
                f"This is a conversation between two reviewers {reviewer_names} for a "
                "Systematic Review. They are trying to achieve a unanimous final decision "
                "for references where their independent decisions were not unanimous.\n\n"
                f"Paper ID: {entry.get('id', '?')}\n"
                f"Title   : {entry.get('title', '?')}\n"
                f"Year    : {entry.get('year', '?')}\n"
                f"Journal : {entry.get('journal', '?')}\n"
                f"Authors : {entry.get('authors', '?')}\n"
                f"Abstract: {entry.get('abstract', '?')}\n\n"
                f"Independent decision of {reviewer_names[0]}: {dec_a}\n"
                f"Independent decision of {reviewer_names[1]}: {dec_b}\n"
            )
        else:
            sid = str(entry["id"])
            transcript = (
                f"This is a conversation between two reviewers {reviewer_names} for a "
                "Systematic Review. They are trying to achieve a unanimous final decision "
                "for references where their independent decisions were not unanimous.\n\n"
                f"Title: {short_texts[sid]['title']}\n"
                f"Text: {short_texts[sid]['prompt']}\n\n"
                f"Independent decision of {reviewer_names[0]}: {dec_a}\n"
                f"Independent decision of {reviewer_names[1]}: {dec_b}\n"
            )

        final_decision = None
        for round_idx in range(max_rounds):
            for rev_idx, (reviewer, name) in enumerate(zip(reviewers, reviewer_names)):
                transcript += f"\n{name}: "
                _, resp = reviewer.reply_to(transcript)
                resp = unmark(resp).strip()
                transcript += resp + "\n"

                if round_idx == 0 and rev_idx == 0:
                    continue

                lower = resp.lower()
                if lower.startswith(("include -", "exclude -")):
                    final_decision = resp.split(" - ")[0].upper()
                elif "final decision:" in lower:
                    rest = lower[lower.find("final decision:") + len("final decision:"):].lstrip()
                    word = rest.split(maxsplit=1)[0] if rest else ""
                    if word in ("include", "exclude"):
                        final_decision = word.upper()
                elif "agreed" in lower:
                    if "include" in lower and "exclude" not in lower:
                        final_decision = "INCLUDE"
                    elif "exclude" in lower and "include" not in lower:
                        final_decision = "EXCLUDE"

                if final_decision:
                    break
            if final_decision:
                break

        if final_decision:
            entry[consensus_key] = final_decision
            transcript += "\n=== CONSENSUS REACHED ===\n"
        else:
            unresolved.append((entry, transcript, reviewer_names, consensus_key))
            transcript += "\n=== NO AUTOMATIC CONSENSUS ===\n"
        transcript_store[entry["id"]] = transcript

    if unresolved:
        print(f"\n{len(unresolved)} conflict(s) require manual resolution:")
        for entry, transcript, reviewer_names, cons_key in unresolved:
            print(f"\n--- ID: {entry.get('id', '?')} ---")
            print(transcript)
            while True:
                choice = input("Enter 'I' (Include) or 'E' (Exclude): ").strip().upper()
                if choice in ("I", "E"):
                    entry[cons_key] = "INCLUDE" if choice == "I" else "EXCLUDE"
                    break

    return transcript_store


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
                e["B2-CR_fulltext_validity"] = decision
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
# Full-text initial screening: two random agents per paper
# ---------------------------------------------------------------------------

def screen_fulltexts_initial(
    short_texts: dict, entries_xml: dict, cfg: dict
) -> None:
    """Two-pass full-text screening; decisions stored as B2-CR_fulltext_{reviewer_id}_decision."""
    gen_kwargs = cfg.get("generation_kwargs", {})

    def process_one(pmid: str, data: dict) -> None:
        entry = next(
            (e for e in entries_xml["root"]["entry"] if str(e.get("id")) == pmid), None
        )
        if entry is None:
            return
        if entry.get("B2-CR_fulltext_validity", "").split(" - ")[0] == "INVALID":
            for reviewer_id in ("reviewer_1", "reviewer_2"):
                entry[f"B2-CR_fulltext_{reviewer_id}_decision"] = "EXCLUDE - Invalid paper"
            return

        for reviewer_id in ("reviewer_1", "reviewer_2"):
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
            entry[f"B2-CR_fulltext_{reviewer_id}_decision"] = unmark(decision)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor, \
            tqdm(total=len(short_texts), desc="Full-text screening", unit="pdf") as pbar:
        futures = {executor.submit(process_one, pid, val): pid for pid, val in short_texts.items()}
        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)


# ---------------------------------------------------------------------------
# One complete cycle
# ---------------------------------------------------------------------------

def run_one_cycle(cycle_idx: int, cfg: dict) -> dict:
    print(f"\n=== RUN {cycle_idx + 1} / {cfg['n_cycles']} ===\n")

    entries = load_xml(cfg["xml_input"])
    clear_folder(cfg["pdf_folder"])

    screen_titles_parallel(entries, cfg=cfg, max_workers=cfg["max_workers"])

    ta_transcripts = resolve_conflicts(
        entries, decision_prefix="", consensus_key="B2-CR_ta_consensus",
        short_texts=None, cfg=cfg,
    )
    ta_counts = {}
    for e in entries["root"]["entry"]:
        k = e.get("B2-CR_ta_consensus", "")
        if k:
            ta_counts[k] = ta_counts.get(k, 0) + 1
    print(f"T&A consensus: INCLUDE: {ta_counts.get('INCLUDE', 0)}, EXCLUDE: {ta_counts.get('EXCLUDE', 0)}")
    transcript_dir = Path(cfg.get("transcript_dir", "transcripts")) / "ta"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    with open(transcript_dir / f"ta_transcript_B2-CR_run{cycle_idx + 1}.json", "w", encoding="utf-8") as f:
        json.dump(ta_transcripts, f, indent=2, ensure_ascii=False)

    included = [
        {"title": e["title"], "authors": e["authors"], "doi": e.get("doi", ""), "id": e["id"], "year": e.get("year", "")}
        for e in entries["root"]["entry"]
        if e.get("B2-CR_ta_consensus", "").upper() == "INCLUDE"
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

    valid_cnt, invalid_cnt, unclear_cnt = validate_fulltexts(short_texts, entries, cfg)
    screen_fulltexts_initial(short_texts, entries, cfg)

    ft_transcripts = resolve_conflicts(
        entries, decision_prefix="B2-CR_fulltext_", consensus_key="B2-CR_fulltext_consensus",
        short_texts=short_texts, cfg=cfg,
    )
    ft_dir = Path(cfg.get("transcript_dir", "transcripts")) / "fulltext"
    ft_dir.mkdir(parents=True, exist_ok=True)
    with open(ft_dir / f"ft_transcript_B2-CR_run{cycle_idx + 1}.json", "w", encoding="utf-8") as f:
        json.dump(ft_transcripts, f, indent=2, ensure_ascii=False)

    final_counts = {"INCLUDE": 0, "EXCLUDE": 0, "UNCLEAR": 0}
    for e in entries["root"]["entry"]:
        key = e.get("B2-CR_fulltext_consensus", "").split(" - ")[0].upper()
        if key:
            final_counts[key] = final_counts.get(key, 0) + 1

    out_xml = add_long_path_prefix(f"{cfg['xml_output_base']}_run{cycle_idx + 1}.xml")
    write_xml(entries, str(out_xml))
    print(f"\nRun {cycle_idx + 1} done, output: {out_xml}")

    export_included(
        entries,
        consensus_key="B2-CR_fulltext_consensus",
        pdf_folder=Path(cfg["pdf_folder"]),
        out_folder=Path(cfg["xml_output_base"]).parent,
        run_idx=cycle_idx + 1,
    )

    return {
        "run": cycle_idx + 1,
        "valid_papers": valid_cnt,
        "invalid_papers": invalid_cnt,
        "unclear_validity": unclear_cnt,
        "fulltext_includes": final_counts["INCLUDE"],
        "fulltext_excludes": final_counts["EXCLUDE"],
        "fulltext_unclear": final_counts["UNCLEAR"],
        "xml_path": str(out_xml),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="B2-CR: single basic agent with conflict resolution")
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
