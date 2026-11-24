"""
Generate an eval case, in one of two ways:
  - default: plant red flags from the playbook into a perfect NDA (drafted by Claude once).
  - --from-original: reword the original NDA so it looks different but keeps the same problems.
    It still shares the original's problems and clause mix, so use it for one or two cases only.

Note: this is a development tool, not part of the redliner. Unlike redline-script.py and
eval-script.py (standard library only, Python 3.8+), it uses the Anthropic Python SDK, which
needs Python 3.10+ and `pip install anthropic`.

Usage:
    python generate-eval-data.py --out eval_data/case-02                        # 5-8 issues (random)
    python generate-eval-data.py --issues 6 --out eval_data/case-03             # exactly 6 issues
    python generate-eval-data.py --issues 6 --seed 3 --out eval_data/case-03    # same clauses every run
    python generate-eval-data.py --from-original --out eval_data/reworded-01    # reworded original
"""

import argparse
import json
import random
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """You create test data for an AI tool that reviews NDAs on behalf of the \
Discloser against a negotiation playbook. You are given an NDA that fully complies with the \
playbook, and some playbook clauses. For each clause, pick one of its red flags and rewrite the \
NDA so that it contains that red flag, the way a lawyer acting for the Recipient might draft it.

Rules:
- Exactly one problem per clause, made in one place: the section that clause belongs to. Change \
only what is needed; copy every other part of the NDA exactly.
- Make each problem realistic: wording a real counterparty might propose, not an obviously \
absurd term. Do not reuse the red flag's own wording.
- Introduce only those problems, nothing else the playbook would object to.
- Never delete a whole section. If a red flag says something is missing, weaken or remove that \
protection inside the section instead.

Return:
- document: the full rewritten NDA.
- issues: exactly one entry per clause, with:
  - text_snippet: the sentence(s) a reviewer would redline, copied character-for-character from \
your document. If a protection was removed, quote the sentence(s) where it should have been.
  - playbook_clause_reference: the playbook clause name, exactly as given.
  - suggested_fix: contract-ready text that should replace text_snippet. Not commentary."""

ISSUE = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("text_snippet", "playbook_clause_reference", "suggested_fix")},
    "required": ["text_snippet", "playbook_clause_reference", "suggested_fix"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {"document": {"type": "string"}, "issues": {"type": "array", "items": ISSUE}},
    "required": ["document", "issues"],
    "additionalProperties": False,
}


PERFECT_SYSTEM_PROMPT = """You are a contracts lawyer acting for the Discloser. Draft a complete, \
ready-to-sign NDA between "Discloser" and "Recipient" that fully meets the ideal position of every \
clause in the negotiation playbook you are given.

Rules:
- Include one numbered section for every playbook clause, in playbook order, headed with the \
clause name. Meet that clause's ideal position and avoid every one of its red flags.
- Use each clause's example ideal clause as your starting point. The examples are templates taken \
from other agreements, so make them correct for this document: fill in any placeholders for \
terms (periods, amounts, countries) with realistic values, fix any typos, and make every \
cross-reference point to the right section of this document.
- Do not invent company names or contact details. Use bracketed placeholders for party details \
(e.g. [Discloser legal name], [Recipient email address], [Discloser street address]).
- Use the defined terms consistently throughout.
- Write contract text only: no commentary, notes, or explanations.

Return the full NDA as `document`."""

PERFECT_SCHEMA = {
    "type": "object",
    "properties": {"document": {"type": "string"}},
    "required": ["document"],
    "additionalProperties": False,
}

PERFECT_FIELDS = ("clause_definition", "review_instruction", "ideal", "red_flag", "example_ideal_clause")


def ask_claude(system: str, user: str, schema: dict, effort: str) -> dict:
    """One streamed Claude call whose answer must match `schema`; returns the parsed JSON."""
    with anthropic.Anthropic().beta.messages.stream(
        model="claude-opus-5-5",
        max_tokens=64000,
        system=system,
        messages=[{"role": "user", "content": user}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason != "end_turn":
        raise SystemExit(f"Claude stopped early: {response.stop_reason}")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def build_perfect_document(playbook: list) -> str:
    """The clean NDA every case starts from, drafted by Claude to meet every clause's ideal position.

    Check it before relying on it: read it, and run redline-script.py on it, which should find
    no issues."""
    clauses = [{"clause": c["clause"].strip(), **{k: c[k] for k in PERFECT_FIELDS}} for c in playbook]
    user = f"<playbook>\n{json.dumps(clauses, indent=2, ensure_ascii=False)}\n</playbook>"
    return ask_claude(PERFECT_SYSTEM_PROMPT, user, PERFECT_SCHEMA, effort="high")["document"].strip() + "\n"


REWORD_SYSTEM_PROMPT = """You create test data for an AI tool that reviews NDAs against a \
negotiation playbook. You are given an NDA and its answer key: the problems a reviewer found in it. \
Write a different-looking NDA that contains exactly the same problems, to test whether the tool \
relies on the original's exact wording and layout.

Rules:
- Reword every sentence, reorder the sections, and use different section headings and numbering. \
You may merge or split sections. Keep the same parties and defined terms.
- Keep the legal effect of every listed problem exactly: each problem must still be there, just \
worded differently.
- Do not fix any other part of the NDA and do not add new problems.

Return:
- document: the full rewritten NDA.
- issues: one entry per listed problem, in the same order, with:
  - text_snippet: the sentence(s) in your document containing that problem, copied \
character-for-character from your document.
  - playbook_clause_reference: unchanged from the answer key.
  - suggested_fix: the answer key's fix, adjusted only so it fits your wording."""


def reword_original() -> dict:
    """A reworded copy of the original NDA with the same problems; its answer key carries over."""
    nda = Path("eval_data/original/document.txt").read_text(encoding="utf-8")
    answer_key = Path("eval_data/original/expected_output.json").read_text(encoding="utf-8")
    user = f"<nda>\n{nda}</nda>\n\n<answer_key>\n{answer_key}\n</answer_key>"
    return ask_claude(REWORD_SYSTEM_PROMPT, user, SCHEMA, effort="medium")


def plant_red_flags(playbook: list) -> dict:
    """The perfect NDA with red flags from randomly picked playbook clauses planted in it."""
    perfect = Path("eval_data/perfect/document.txt")
    if not perfect.exists():  # built once; delete the file to rebuild it
        print("No perfect NDA yet; asking Claude to draft one from the playbook...")
        perfect.parent.mkdir(parents=True, exist_ok=True)
        perfect.write_text(build_perfect_document(playbook), encoding="utf-8")
        Path("eval_data/perfect/expected_output.json").write_text("[]\n", encoding="utf-8")  # no issues
        print(f"Wrote {perfect}. Check it (read it, and redline it: it should have no issues) before using it.")
    nda = perfect.read_text(encoding="utf-8")
    clauses = [
        {"clause": c["clause"], "is_required": c["is_required"], "red_flags": c["red_flag"]}
        for c in random.sample(playbook, args.issues or random.randint(5, 8))  # random clauses so cases vary
    ]
    user = f"<nda>\n{nda}</nda>\n\n<clauses>\n{json.dumps(clauses, indent=2, ensure_ascii=False)}\n</clauses>"
    return ask_claude(SYSTEM_PROMPT, user, SCHEMA, effort="medium")


parser = argparse.ArgumentParser()
parser.add_argument("--issues", type=int, help="Number of red flags to inject (default: random 5-8)")
parser.add_argument("--out", required=True, help="Output folder, e.g. eval_data/case-01")
parser.add_argument("--seed", type=int, help="Random seed, so the same clauses are picked every run")
parser.add_argument("--from-original", action="store_true",
                    help="Reword the original NDA, keeping its problems and answer key (use for 1-2 cases only)")
args = parser.parse_args()
random.seed(args.seed)  # None (the default) means a different random pick each run

playbook = json.loads(Path("playbook.json").read_text(encoding="utf-8"))
result = reword_original() if args.from_original else plant_red_flags(playbook)

# Check the answer key follows the same rules as the redliner's output; don't save a broken case.
errors, seen = [], set()
for issue in result["issues"]:
    clause, snippet = issue["playbook_clause_reference"], issue["text_snippet"]
    if not snippet.strip() or not issue["suggested_fix"].strip():
        errors.append(f"empty snippet or fix for {clause!r}")
    elif snippet not in result["document"]:
        errors.append(f"snippet not found verbatim in document: {snippet[:80]!r}")
    if clause in seen:
        errors.append(f"more than one issue for clause {clause!r}")
    seen.add(clause)
if errors:
    raise SystemExit("Generated case not saved:\n  " + "\n  ".join(errors) + "\nRun the command again.")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "document.txt").write_text(result["document"], encoding="utf-8")
(out / "expected_output.json").write_text(json.dumps(result["issues"], indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(result['issues'])} issue(s) to {out}")
