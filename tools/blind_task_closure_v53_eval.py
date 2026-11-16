#!/usr/bin/env python3
"""V5.3 read-only Task Closure Y/N evaluation using repository ChatGPT OAuth bridge.

The model receives ONLY conversation turns and (optionally) source SOP text.
It never receives review decisions, injection metadata, recorded explanations,
original/uninjected turns, or the expected label.

Requires: Python 3.10+, openai, and the existing repository OAuth bridge.
Usage examples are in README_AR.md.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MODEL = "gpt-6-luna"

# Mandatory external V5.3-derived judge rubric. No embedded legacy judge prompt.
JUDGE_PROMPT = ""


def canonical_hash(payload: object) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as fh:
        rows = []
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: bad JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def read_human_labels(html: Path, labels_csv: Path | None) -> dict[str, str]:
    """CSV exported from Golden Lens takes precedence over embedded HTML defaults."""
    if labels_csv:
        with labels_csv.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError("No rows in labels CSV")
        raw = {str(r.get("conversation_id", "")).strip():
               str(r.get("review_status", "")).strip().lower() for r in rows}
    else:
        page = html.read_text(encoding="utf-8-sig")
        match = re.search(r"\bconst\s+initialReviews\s*=\s*(\{.*?\})\s*;", page, re.DOTALL)
        if not match:
            raise ValueError("Cannot find initialReviews in HTML. Export CSV from Golden Lens and use --labels-csv.")
        obj = json.loads(match.group(1))
        raw = {key.removeprefix("golden:"): str(value.get("decision", "")).strip().lower()
               for key, value in obj.items() if key.startswith("golden:")}

    decisions = {"approved": "Y", "rejected": "N"}
    bad = {k: v for k, v in raw.items() if v not in decisions}
    if bad:
        raise ValueError(f"Missing/nonbinary human decisions for IDs: {list(bad)[:10]}; complete labels first")
    return {key: decisions[status] for key, status in raw.items()}


def select_records(dataset: Path, labels: dict[str, str], ids: set[str] | None) -> list[dict[str, Any]]:
    items = load_jsonl(dataset)
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for record in items:
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise ValueError("A record has no conversation_id")
        if cid in seen:
            raise ValueError(f"Duplicate conversation_id: {cid}")
        seen.add(cid)
        if ids is not None and cid not in ids:
            continue
        turns = record.get("injected_conversation")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{cid}: missing injected_conversation")
        clean_turns = []
        for i, t in enumerate(turns, 1):
            if not isinstance(t, dict) or t.get("speaker") not in ("CUSTOMER", "AGENT", "ACTION"):
                raise ValueError(f"{cid}: invalid speaker at turn {i}")
            if type(t.get("turn_id")) is not int or not isinstance(t.get("content"), str):
                raise ValueError(f"{cid}: invalid turn at position {i}")
            clean_turns.append({"turn_id": t["turn_id"], "speaker": t["speaker"], "content": t["content"]})
        if cid not in labels:
            raise ValueError(f"{cid}: human label missing")
        # Do not copy metadata, violations, judge_evaluations, or known label into model input.
        selected.append({"conversation_id": cid, "turns": clean_turns, "human_label": labels[cid]})
    missing = (ids or set()) - seen
    if missing:
        raise ValueError(f"Unknown conversation IDs: {sorted(missing)}")
    return selected


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def meaning(value: str) -> str:
    stop = {"due", "to", "faq", "how", "other", "general", "one", "item"}
    words = re.findall(r"[a-z0-9]+", str(value).casefold())
    return "".join(w for w in words if w not in stop).rstrip("0123456789")


def load_sops(abcd_file: Path, guideline_file: Path, aliases_file: Path) -> dict[str, Any]:
    """Recover source SOP by conversation ID, without reading or sending scenario truth."""
    source = json.loads(abcd_file.read_text(encoding="utf-8-sig"))
    source_guidelines = json.loads(guideline_file.read_text(encoding="utf-8-sig"))
    aliases = json.loads(aliases_file.read_text(encoding="utf-8-sig")) if aliases_file.is_file() else {}
    lookups: dict[str, Any] = {}
    for group_name, group in source_guidelines.items():
        for sub_name, sop in group.get("subflows", {}).items():
            for key in (normalize(sub_name), meaning(sub_name)):
                lookups.setdefault(key, sop)
    sops = {}
    for subset in source.values():
        for record in subset:
            scenario = record.get("scenario", {})
            flow, sub = str(scenario.get("flow", "")), str(scenario.get("subflow", ""))
            key = normalize(sub)
            found = lookups.get(key)
            if found is None:
                alt = aliases.get(flow, {}).get(sub)
                if alt is None:
                    flow_aliases = next((v for k, v in aliases.items() if normalize(k) == normalize(flow)), {})
                    alt = next((v for k, v in flow_aliases.items() if normalize(k) == key), None)
                if alt:
                    found = lookups.get(normalize(alt))
            if found is None:
                found = lookups.get(meaning(sub))
            if found is not None:
                sops[str(record["convo_id"])] = found
    return sops


def model_messages(turns: list[dict[str, Any]], sop: Any = None, judge_prompt: str | None = None) -> list[dict[str, str]]:
    # Do NOT pass id, human_label, injection_metadata, judge_evaluations, raw response reason,
    # or the original conversation to the remote model.
    payload: dict[str, Any] = {"conversation": turns}
    if sop is not None:
        payload["applicable_sop"] = sop
    return [
        {"role": "system", "content": judge_prompt if judge_prompt is not None else JUDGE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_model_json(response: str, allowed_ids: set[int]) -> dict[str, Any]:
    text = response.strip()
    # Some model responses are wrapped in fenced code blocks.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object in judge response")
        decoder = json.JSONDecoder()
        try:
            obj, rest = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("Unparseable judge JSON") from exc
        if text[:start].strip() or rest.strip():
            raise ValueError("Judge returned non-JSON extra text")
    if not isinstance(obj, dict) or obj.get("label") not in ("Y", "N"):
        raise ValueError("Judge label must be Y or N")
    reason = obj.get("reason", "")
    turn_ids = obj.get("evidence_turn_ids", [])
    if not isinstance(reason, str) or not isinstance(turn_ids, list):
        raise ValueError("Invalid reason or evidence_turn_ids")
    if any(type(tid) is not int or tid not in allowed_ids for tid in turn_ids):
        raise ValueError("Invalid evidence turn ID")
    return {"label": obj["label"], "reason": reason.strip(), "evidence_turn_ids": turn_ids}


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def summary_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("status") == "ok" and r.get("luna_label") in ("Y", "N")]
    confusion = Counter((r["human_label"], r["luna_label"]) for r in valid)
    tp, tn, fp, fn = (confusion[("Y", "Y")], confusion[("N", "N")],
                      confusion[("N", "Y")], confusion[("Y", "N")])
    safe_div = lambda x, y: round(x / y, 4) if y else None
    mismatches = [r["conversation_id"] for r in valid if r["human_label"] != r["luna_label"]]
    return {
        "model": MODEL,
        "total_rows": len(rows), "evaluated": len(valid),
        "errors": sum(r.get("status") == "error" for r in rows),
        "human_Y": sum(r["human_label"] == "Y" for r in rows),
        "human_N": sum(r["human_label"] == "N" for r in rows),
        "agree": tp + tn, "disagree": fp + fn,
        "accuracy": safe_div(tp + tn, len(valid)),
        "precision_Y": safe_div(tp, tp + fp),
        "recall_Y": safe_div(tp, tp + fn),
        "f1_Y": safe_div(2 * tp, 2 * tp + fp + fn),
        "confusion_matrix": {"human_Y_luna_Y": tp, "human_Y_luna_N": fn,
                             "human_N_luna_Y": fp, "human_N_luna_N": tn},
        "mismatch_ids": mismatches,
        "important_note": "Compares Luna against human Y/N labels. It does not establish the semantic truth of every case independently.",
    }


def save_comparison(output_dir: Path, source: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for record in source:
        cid = record["conversation_id"]
        result = results.get(cid, {})
        predicted = result.get("luna_label", "") if result.get("status") == "ok" else ""
        rows.append({
            "conversation_id": cid,
            "human_label": record["human_label"],
            "luna_label": predicted,
            "agreement": "YES" if predicted == record["human_label"] else ("NO" if predicted else ""),
            "status": result.get("status", "not_run"),
            "sop_included": result.get("sop_included", ""),
            "reason": result.get("reason", ""),
            "evidence_turn_ids": ";".join(str(x) for x in result.get("evidence_turn_ids", [])),
            "error": result.get("error", ""),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comparison.csv"
    with output_file.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["conversation_id"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def import_bridge(repo: Path, oauth_script: Path):
    path = repo / "src" / "dataset_generation"
    if not (path / "chatgpt_oauth_bridge.py").is_file():
        raise FileNotFoundError(f"Missing repository bridge: {path / 'chatgpt_oauth_bridge.py'}")
    sys.path.insert(0, str(path))
    from chatgpt_oauth_bridge import ChatGPTOAuthResponsesClient  # noqa: PLC0415
    bridge = ChatGPTOAuthResponsesClient(oauth_script)
    bridge.check_login()
    return bridge


async def run(args: argparse.Namespace, client: Any = None) -> dict[str, Any]:
    prompt_file = getattr(args, "judge_prompt_file", None)
    if not prompt_file:
        raise ValueError("Provide --judge-prompt-file with the V5.3 judge-only rubric")
    prompt_text = prompt_file.read_text(encoding="utf-8-sig").strip()
    if not prompt_text:
        raise ValueError("Empty judge prompt file")
    prompt_id = str(prompt_file)
    prompt_hash = canonical_hash(prompt_text)
    labels = read_human_labels(args.html, args.labels_csv)
    selected_ids = set(args.ids.split(",")) if args.ids else None
    selected = select_records(args.dataset, labels, selected_ids)
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No cases selected")

    sops: dict[str, Any] = {}
    if args.with_sop:
        if not (args.abcd and args.guidelines and args.aliases):
            raise ValueError("--with-sop requires --abcd --guidelines --aliases")
        sops = load_sops(args.abcd, args.guidelines, args.aliases)
        missing = [x["conversation_id"] for x in selected if x["conversation_id"] not in sops]
        if missing:
            raise ValueError(f"SOP missing for {len(missing)} cases (examples: {missing[:8]})")

    if args.dry_run:
        print(f"DRY RUN: {len(selected)} cases, labels: {dict(Counter(x['human_label'] for x in selected))}")
        print(f"Model: {MODEL}; SOP included: {bool(args.with_sop)}")
        print(f"Judge rubric: {prompt_id}; prompt fingerprint: {prompt_hash}")
        print("Only messages: system (judge rubric), user (injected conversation + optional SOP)")
        print("No OAuth call, no model verdicts, no accuracy results.")
        return {"dry_run": True, "n": len(selected)}

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw_out = output / "luna_results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if raw_out.is_file():
        for row in load_jsonl(raw_out):
            cid = str(row.get("conversation_id"))
            if cid in existing:
                raise ValueError(f"Duplicate result for case {cid}; do not mix result files")
            existing[cid] = row

    expected_prompt_hash = prompt_hash
    for rec in selected:
        cid = rec["conversation_id"]
        if cid not in existing:
            continue
        prior = existing[cid]
        expected_input_hash = canonical_hash(model_messages(rec["turns"], sops.get(cid) if args.with_sop else None, prompt_text))
        if prior.get("model") != MODEL or prior.get("prompt_hash") != expected_prompt_hash or prior.get("input_hash") != expected_input_hash:
            raise ValueError(f"Case {cid} has previous results with a different prompt/model/input. Use another output dir.")

    if client is None:
        if not args.oauth_script:
            raise ValueError("Provide --oauth-script or CHATGPT_OAUTH_SCRIPT; don't provide tokens")
        client = import_bridge(args.repo, args.oauth_script)

    for index, rec in enumerate(selected, 1):
        cid = rec["conversation_id"]
        if cid in existing and existing[cid].get("status") == "ok":
            print(f"[{index}/{len(selected)}] {cid}: resumed ({existing[cid]['luna_label']})", flush=True)
            continue
        if cid in existing and existing[cid].get("status") == "error":
            # Avoid append duplicate IDs: re-run failures in a new output directory.
            print(f"[{index}/{len(selected)}] {cid}: previous error; use a new output dir to retry", flush=True)
            continue
        sop = sops.get(cid) if args.with_sop else None
        messages = model_messages(rec["turns"], sop, prompt_text)
        valid_ids = {t["turn_id"] for t in rec["turns"]}
        result: dict[str, Any] = {
            "conversation_id": cid,
            "human_label": rec["human_label"],
            "model": MODEL,
            "prompt_hash": expected_prompt_hash,
            "prompt_file": prompt_id,
            "input_hash": canonical_hash(messages),
            "sop_included": args.with_sop,
        }
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = await client.generate(messages, MODEL, args.timeout)
                judged = parse_model_json(response, valid_ids)
                result.update(status="ok", luna_label=judged["label"],
                              reason=judged["reason"], evidence_turn_ids=judged["evidence_turn_ids"])
                break
            except Exception as exc:
                if attempt == args.max_attempts:
                    result.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
                else:
                    await asyncio.sleep(min(12.0, 2 ** (attempt - 1) + random.random()))
        write_jsonl(raw_out, result)
        existing[cid] = result
        if result["status"] == "ok":
            match = "MATCH" if result["luna_label"] == rec["human_label"] else "MISMATCH"
            print(f"[{index}/{len(selected)}] {cid}: human={rec['human_label']} luna={result['luna_label']} {match}", flush=True)
        else:
            print(f"[{index}/{len(selected)}] {cid}: ERROR (see results file)", flush=True)
        save_comparison(output, selected, existing)

    save_comparison(output, selected, existing)
    relevant = [{"conversation_id": rec["conversation_id"], "human_label": rec["human_label"],
                 **existing.get(rec["conversation_id"], {})} for rec in selected]
    overview = summary_rows(relevant)
    print("\nSUMMARY", json.dumps(overview, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {output / 'comparison.csv'} and {output / 'summary.json'}", flush=True)
    return overview


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind Task Closure evaluation with OAuth gpt-6-luna")
    parser.add_argument("--dataset", type=Path, required=True, help="golden_dataset_injected.jsonl")
    parser.add_argument("--html", type=Path, required=True, help="Golden Lens HTML with initialReviews")
    parser.add_argument("--judge-prompt-file", type=Path, required=True, help="Required V5.3 evaluation-only rubric file")
    parser.add_argument("--labels-csv", type=Path, help="Optional current CSV export overrides HTML labels")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to dynamic-multi-turn-agent-evaluation")
    parser.add_argument("--oauth-script", type=Path, default=Path(os.environ['CHATGPT_OAUTH_SCRIPT']) if os.getenv('CHATGPT_OAUTH_SCRIPT') else None, help="Path to existing chatgpt_plus_oauth_fixed.py")
    parser.add_argument("--output-dir", type=Path, default=Path("blind_luna_results"))
    parser.add_argument("--ids", help="Comma-separated conversation IDs")
    parser.add_argument("--limit", type=int, help="Evaluate first N cases")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs, labels and SOP coverage without OAuth")
    parser.add_argument("--with-sop", action="store_true", help="Send applicable source SOP alongside conversation (still blind)")
    parser.add_argument("--abcd", type=Path, help="Source data/abcd_v1.1.json for optional SOP")
    parser.add_argument("--guidelines", type=Path, help="Source data/guidelines.json for optional SOP")
    parser.add_argument("--aliases", type=Path, help="Repo config/guideline_aliases.json for optional SOP")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
