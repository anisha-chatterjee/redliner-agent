"""
Evaluate a redliner's output against an answer key.

  1. Match snippets   - code: a fuzzy score says whether two snippets quote the same text.
  2. Sort redlines    - code: match / misattributed / missed / not in expected,
                        then precision, recall and F1.
  3. Grade fixes      - LLM: is each matched fix ideal, acceptable or still a red flag?
  4. Closer look      - LLM: is each unmatched redline a fair flag anyway?

Standard library only. Steps 3 and 4 are skipped with --no-llm or without ANTHROPIC_API_KEY.

Usage:
    python evaluate.py --output redline_output/original-v4.json [--details] [--no-llm]

The answer key is found from the file name (original-v4 -> eval_data/original/expected_output.json)
and the report is saved to eval_report_output/original-v4.json.
"""

import argparse
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher

JUDGE_MODEL = "claude-sonnet-5-5"  # not the redliner's model, so it isn't grading its own work
MATCH_CUTOFF = 85                  # fuzzy score (0-100) at which two snippets count as the same text
REPORT_DIR = "eval_report_output"


# --- 1. Match snippets -------------------------------------------------------

def clean(text):
    """Lowercase, straighten curly quotes and dashes, collapse whitespace."""
    text = (text or "").translate(str.maketrans("“”‘’–— ", "\"\"''-- "))
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_score(a, b):
    """0-100: how well the shorter text fits somewhere inside the longer one.

    For each stretch the two texts share, line the short text up against an equally long
    window of the long text and compare them. The best window wins, so quoting one sentence
    of a paragraph still scores 100."""
    short, long_ = sorted((clean(a), clean(b)), key=len)
    if not short:
        return 0.0
    best = 0.0
    for i, j, _ in SequenceMatcher(None, short, long_, autojunk=False).get_matching_blocks():
        window = long_[max(0, j - i):max(0, j - i) + len(short)]
        best = max(best, SequenceMatcher(None, short, window, autojunk=False).ratio())
    return round(best * 100, 1)


# --- 2. Sort redlines --------------------------------------------------------

def sort_redlines(predicted, expected):
    """Put every redline from both lists into one bucket. Returns a list of rows."""
    same_text = sorted(((fuzzy_score(p["text_snippet"], e["text_snippet"]), pi, ei)
                        for pi, p in enumerate(predicted) for ei, e in enumerate(expected)), reverse=True)
    same_text = [(pi, ei) for score, pi, ei in same_text if score >= MATCH_CUTOFF]
    used_p, used_e, rows = set(), set(), []

    def pair_up(bucket, same_clause):
        # Strongest scores first; each redline is used at most once.
        for pi, ei in same_text:
            clause_matches = clean(predicted[pi]["playbook_clause_reference"]) == \
                clean(expected[ei]["playbook_clause_reference"])
            if pi not in used_p and ei not in used_e and clause_matches == same_clause:
                used_p.add(pi)
                used_e.add(ei)
                rows.append({"bucket": bucket, "predicted": predicted[pi], "expected": expected[ei]})

    pair_up("match", same_clause=True)
    pair_up("misattributed", same_clause=False)
    rows += [{"bucket": "missed", "predicted": None, "expected": e}
             for ei, e in enumerate(expected) if ei not in used_e]
    rows += [{"bucket": "not in expected", "predicted": p, "expected": None}
             for pi, p in enumerate(predicted) if pi not in used_p]
    return rows


def scores(rows, n_predicted, n_expected):
    """Precision, recall, F1. None when there is nothing to divide by (e.g. a clean document)."""
    matches = sum(1 for r in rows if r["bucket"] == "match")
    precision = matches / n_predicted if n_predicted else None
    recall = matches / n_expected if n_expected else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --- 3 & 4. LLM judge --------------------------------------------------------

def ask_claude(api_key, prompt, schema):
    """One Messages API call that must answer in the given JSON schema. Retries rate limits / server errors."""
    body = json.dumps({
        "model": JUDGE_MODEL,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": schema}},
        "fallbacks": "default",
    }).encode("utf-8")
    headers = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2026-10-21",
               "anthropic-beta": "server-side-fallback-2026-07-01"}
    for attempt in range(3):
        try:
            request = urllib.request.Request("https://api.anthropic.com/v1/messages", body, headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read())
            break
        except urllib.error.HTTPError as e:
            if attempt == 2 or not (e.code == 429 or e.code >= 500):
                raise RuntimeError("API error {}: {}".format(e.code, e.read().decode("utf-8", "replace")))
        except (urllib.error.URLError, OSError) as e:  # OSError covers read timeouts (socket.timeout)
            if attempt == 2:
                raise RuntimeError("Network error: {}".format(getattr(e, "reason", e)))
        time.sleep(2 ** (attempt + 1))
    if result.get("stop_reason") != "end_turn":
        raise RuntimeError("judge stopped early: {}".format(result.get("stop_reason")))
    return json.loads(next(b["text"] for b in result["content"] if b["type"] == "text"))


def json_object(**properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


FIX_SCHEMA = json_object(
    verdict={"type": "string", "enum": ["ideal", "acceptable", "red_flag"]},
    reason={"type": "string"},
)
FAIR_SCHEMA = json_object(fair={"type": "boolean"}, reason={"type": "string"})

# Some playbook fields are empty or were copied from another clause (e.g. Notices' acceptable
# position is the Privacy Policy's), so the judge gets every field and is told what to do then.
PLAYBOOK_NOTE = ("If a playbook field is empty or clearly about a different topic, ignore it and judge "
                 "against the clause's other fields, including its example ideal clause.")


def playbook_clause(clause):
    """The playbook fields the judge needs for one clause, as tagged sections."""
    fields = [("review_instruction", "review_instruction"), ("ideal", "ideal_position"),
              ("acceptable", "acceptable_position"), ("red_flag", "red_flags"),
              ("example_ideal_clause", "example_ideal_clause")]
    return "\n\n".join("<{0}>\n{1}\n</{0}>".format(tag, (clause.get(key) or "(none)").strip())
                       for key, tag in fields)


def grade_fix(api_key, clause, predicted, expected):
    """Step 3: grade a matched redline's suggested fix against the playbook clause."""
    prompt = (
        "You are a senior contracts lawyer grading a proposed fix to a contract against a negotiation "
        "playbook. Judge whether it fixes the problem according to the playbook, not whether it uses the "
        "same words as the reference fix. {}\n"
        "verdict: 'ideal' if it meets the ideal position, 'acceptable' if it meets the acceptable position, "
        "'red_flag' if it still matches a red flag or leaves the problem unfixed.\n"
        "reason: one or two sentences.\n\n"
        "{}\n\n<original_text>\n{}\n</original_text>\n\n"
        "<reference_fix>\n{}\n</reference_fix>\n\n<proposed_fix>\n{}\n</proposed_fix>"
    ).format(PLAYBOOK_NOTE, playbook_clause(clause),
             expected["text_snippet"], expected["suggested_fix"], predicted["suggested_fix"])
    return ask_claude(api_key, prompt, FIX_SCHEMA)


def closer_look(api_key, clause, predicted):
    """Step 4: is an unmatched redline a genuine issue under the clause it cites?"""
    prompt = (
        "You are a senior contracts lawyer. The text below was flagged as violating the playbook clause "
        "'{}'. Is it a fair flag: does the text genuinely match one of the red flags, or fall short of the "
        "acceptable position? Be strict: text that is already acceptable, or that belongs to a different "
        "clause, is not a fair flag. {} Give a one- or two-sentence reason.\n\n"
        "{}\n\n<flagged_text>\n{}\n</flagged_text>"
    ).format(predicted["playbook_clause_reference"].strip(), PLAYBOOK_NOTE, playbook_clause(clause),
             predicted["text_snippet"])
    return ask_claude(api_key, prompt, FAIR_SCHEMA)


def run_judge(api_key, rows, playbook):
    clauses = {clean(c["clause"]): c for c in playbook}
    for r in rows:
        try:
            if r["bucket"] == "match":
                clause = clauses.get(clean(r["expected"]["playbook_clause_reference"]), {})
                r["fix_grade"] = grade_fix(api_key, clause, r["predicted"], r["expected"])
            elif r["bucket"] in ("misattributed", "not in expected"):
                clause = clauses.get(clean(r["predicted"]["playbook_clause_reference"]), {})
                r["closer_look"] = closer_look(api_key, clause, r["predicted"])
        except RuntimeError as err:
            print("WARNING: judge failed on a {} redline: {}".format(r["bucket"], err))


# --- Output ------------------------------------------------------------------

def pct(x):
    return "n/a" if x is None else "{:.1f}%".format(100 * x)


def clause_of(r):
    return (r["expected"] or r["predicted"])["playbook_clause_reference"].strip()


def print_results(rows, metrics, n_predicted, n_expected, details):
    def reason(judgement):
        if details and judgement:
            for line in textwrap.wrap(judgement["reason"], width=88):
                print("      " + line)

    print("\n=== Results ===")
    print("Precision {}   Recall {}   F1 {}".format(pct(metrics["precision"]), pct(metrics["recall"]),
                                                    pct(metrics["f1"])))
    grades = [r["fix_grade"] for r in rows if r.get("fix_grade")]
    if grades:
        counts = [sum(1 for g in grades if g["verdict"] == v) for v in ("ideal", "acceptable", "red_flag")]
        print("Fixes     {} ideal, {} acceptable, {} still a red flag".format(*counts))
    if n_expected == 0:
        print("Clean document: nothing to find, so the ideal output is 0 redlines. Redliner flagged {}.".format(
            n_predicted))

    if grades:
        print("\nFix grade (by the LLM judge, against the playbook clause): ideal = meets the ideal position |")
        print("  acceptable = meets the acceptable (fallback) position | red flag = still has the problem")
    if not details and any(r.get("fix_grade") or r.get("closer_look") for r in rows):
        print("Add --details to see the judge's reasons.")

    headings = {
        "match": ("Match: same text, same clause", "Fix grade"),
        "misattributed": ("Misattributed: same text, different clause", "Real issue under the clause used?"),
        "missed": ("Missed: in the answer key, never flagged", ""),
        "not in expected": ("Not in expected: flagged, but not in the answer key", "Real issue anyway?"),
    }
    for bucket, (title, column) in headings.items():
        in_bucket = [r for r in rows if r["bucket"] == bucket]
        if not in_bucket:
            continue
        print("\n{} ({})".format(title, len(in_bucket)))
        if column and any(r.get("fix_grade") or r.get("closer_look") for r in in_bucket):
            print("  {:<40} {}".format("", column))
        for r in in_bucket:
            grade, look = r.get("fix_grade"), r.get("closer_look")
            if grade:
                verdict = grade["verdict"].replace("_", " ")
            elif look:
                verdict = "yes" if look["fair"] else "no"
            else:
                verdict = ""
            print("  {:<40} {}".format(clause_of(r), verdict))
            if bucket == "misattributed":
                print("    flagged as: {}".format(r["predicted"]["playbook_clause_reference"].strip()))
            reason(grade or look)


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate redline output against an answer key.")
    parser.add_argument("--output", required=True, help="redliner output, e.g. redline_output/original-v4.json")
    parser.add_argument("--expected", help="answer key (default: eval_data/<case>/expected_output.json)")
    parser.add_argument("--playbook", default="playbook.json")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM judge (steps 3 and 4)")
    parser.add_argument("--details", action="store_true", help="print the judge's reasons")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.output))[0]          # e.g. original-v4
    expected_path = args.expected or os.path.join("eval_data", re.sub(r"-v\d+$", "", name), "expected_output.json")
    with open(args.output, encoding="utf-8") as f:
        predicted = json.load(f)
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)
    with open(args.playbook, encoding="utf-8") as f:
        playbook = json.load(f)
    print("{}: {} redlines vs. {} expected ({})".format(name, len(predicted), len(expected), expected_path))

    rows = sort_redlines(predicted, expected)
    order = {clean(c["clause"]): i for i, c in enumerate(playbook)}
    rows.sort(key=lambda r: order.get(clean(clause_of(r)), len(order)))  # list in playbook order
    metrics = scores(rows, len(predicted), len(expected))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and not args.no_llm:
        print("Running the judge ({})...".format(JUDGE_MODEL))
        run_judge(api_key, rows, playbook)
    else:
        print("No LLM judge: fix grades and the closer look are skipped.")

    print_results(rows, metrics, len(predicted), len(expected), args.details)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, name + ".json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"judge_model": JUDGE_MODEL, "match_cutoff": MATCH_CUTOFF, "scores": metrics,
                   "redlines": [dict(r, clause=clause_of(r)) for r in rows]}, f, indent=2, ensure_ascii=False)
    print("\nReport saved to {}".format(report_path))


if __name__ == "__main__":
    main()
