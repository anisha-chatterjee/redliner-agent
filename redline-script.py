"""
Redline a legal document against a rule-based playbook.

Runs on Python 3.8+ with only the standard library (no pip install needed).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python redline-script.py --document eval_data/original/document.txt --output redline_output/original-v4.json
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2026-10-21"
MODEL = "claude-opus-5-5"
MAX_RETRIES = 3

def build_output_format(clause_names):
    """JSON schema the model's answer must match (enforced by structured outputs).
    The enum limits playbook_clause_reference to the exact clause names in the playbook."""
    return {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text_snippet": {"type": "string"},
                        "playbook_clause_reference": {"type": "string", "enum": clause_names},
                        "suggested_fix": {"type": "string"},
                    },
                    "required": ["text_snippet", "playbook_clause_reference", "suggested_fix"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["issues"],
        "additionalProperties": False,
    }

SYSTEM_PROMPT = """You are reviewing a legal document, using the company's negotiation playbook.

For each playbook clause, follow its review instruction and compare the document's existing text \
against its ideal, acceptable, and red-flag positions. Flag text that matches a red flag or falls \
short of an acceptable position.

Rules for which issues to report:
- At most ONE issue per playbook clause. If a clause has several problems, report only the most \
serious one.
- Only flag problems the playbook covers: text that matches a red flag or falls short of an \
acceptable position. Do not report anything the playbook does not cover, even if you would \
otherwise consider it a problem.
- Do not report a playbook clause that is entirely absent from the document. But if a clause is \
present and lacks a protection its playbook position requires, that is an issue: quote the \
sentence(s) where the protection belongs, and write a fix that adds it.
- Each piece of document text belongs to at most one issue; never report the same text under two \
clauses.
- Choosing the clause: base it on what the provision is and where it sits, using the document's \
section heading together with the playbook's description of what each clause covers and where it \
is usually found. Do not choose a clause just because one of its red flags uses similar wording.
- Keep one problem together: if consecutive sentences in the same section work together to create \
one problem, report them as a single issue with one snippet and one fix, under the clause that \
section belongs to, even if part of the passage also touches another clause's concerns.
- Do not report text that is already acceptable.

For every issue:
- text_snippet: an EXACT, verbatim quote copied character-for-character from the document, \
including its punctuation and quote marks. Quote the complete sentence(s) containing the problem \
(or, for a missing protection, where it belongs), without the section number.
- playbook_clause_reference: the EXACT clause name from the playbook, unchanged.
- suggested_fix: contract-ready replacement text for text_snippet, not commentary. Start from the \
clause's example ideal clause and adapt it to this document: use its defined terms (e.g. \
"Discloser", "Recipient", "Representatives"), keep any parts of the original that are already \
acceptable, and address every problem you found in the snippet. If no example fits, draft new \
language that meets the ideal position.
  - Keep it concise: include only the language needed to fix the problem, roughly the length of \
the example ideal clause, not an expanded version of it.
  - Format to match the content: when the fix sets out several separate obligations, conditions, \
or options, use a short numbered or bulleted list (as the playbook examples often do); otherwise \
use a single sentence or short paragraph.
  - Stay within the referenced clause: only include language that belongs to that playbook \
clause, not material that belongs to other clauses."""

# Playbook fields shown to the model for each clause, in this order, with their labels.
PLAYBOOK_FIELDS = [
    ("clause_definition", "What this clause covers"),
    ("review_instruction", "Review instruction"),
    ("ideal", "Ideal position"),
    ("acceptable", "Acceptable position"),
    ("red_flag", "Red flags"),
    ("example_ideal_clause", "Example ideal clause"),
    ("example_fallback_clause", "Example fallback clause"),
]


def load_inputs(document_path, playbook_path):
    with open(document_path, encoding="utf-8") as f:
        document = f.read()
    with open(playbook_path, encoding="utf-8") as f:
        playbook = json.load(f)
    return document, playbook


def format_playbook(playbook):
    """Render each playbook clause as a labeled section, which is easier to follow than raw JSON."""
    sections = []
    for entry in playbook:
        required = "Required" if entry.get("is_required") else "Optional"
        # Keep the name exactly as stored (some have trailing spaces) so the model copies it verbatim.
        lines = ['<clause name="{}" status="{}">'.format(entry["clause"], required)]
        for field, label in PLAYBOOK_FIELDS:
            value = (entry.get(field) or "").strip()
            if value:
                lines.append("## {}\n{}".format(label, value))
        lines.append("</clause>")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_user_message(document, playbook):
    return (
        "<playbook>\n" + format_playbook(playbook) + "\n</playbook>\n\n"
        "<document>\n" + document + "\n</document>\n\n"
        "Review the document against the playbook and list every issue."
    )


def call_claude(api_key, body):
    """POST to the Messages API, retrying on rate limits and server errors."""
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            # Lets the API retry on a fallback model if a safety classifier declines.
            "anthropic-beta": "server-side-fallback-2026-07-01",
        },
        method="POST",
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            retryable = e.code == 429 or e.code >= 500
            if not retryable or attempt == MAX_RETRIES:
                sys.exit("API error {}: {}".format(e.code, error_body))
            problem = "API error {}".format(e.code)
        except urllib.error.URLError as e:
            if attempt == MAX_RETRIES:
                sys.exit("Network error: {}".format(e.reason))
            problem = "Network error ({})".format(e.reason)
        wait = 2 ** attempt
        print("{}; retrying in {}s (attempt {} of {})...".format(problem, wait, attempt + 1, MAX_RETRIES))
        time.sleep(wait)


def find_issues(api_key, document, playbook):
    body = {
        "model": MODEL,
        "max_tokens": 16000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_message(document, playbook)}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": "high",
            "format": {"type": "json_schema", "schema": build_output_format(clause_names(playbook))},
        },
        "fallbacks": "default",
    }
    print("Sending document and playbook to {} (this can take a few minutes)...".format(MODEL))
    start = time.time()
    response = call_claude(api_key, body)
    usage = response.get("usage", {})
    print("Response received in {:.0f}s from {} ({} input tokens, {} output tokens).".format(
        time.time() - start, response.get("model"), usage.get("input_tokens"), usage.get("output_tokens")))

    if response.get("stop_reason") == "refusal":
        sys.exit("Model refused the request: {}".format(response.get("stop_details")))
    if response.get("stop_reason") == "max_tokens":
        sys.exit("Response hit max_tokens before finishing; output would be incomplete.")

    text = next(block["text"] for block in response["content"] if block["type"] == "text")
    return json.loads(text)["issues"]


def clause_names(playbook):
    return [entry["clause"] for entry in playbook]


def validate_output(issues, playbook, document):
    """Return a list of error messages: a snippet that isn't an exact quote from the document,
    an unknown clause name, or a clause used more than once."""
    valid = set(clause_names(playbook))
    errors = []
    seen = set()
    for issue in issues:
        if issue["text_snippet"] not in document:
            errors.append("snippet not found verbatim in document: {!r}".format(issue["text_snippet"][:80]))
        clause = issue["playbook_clause_reference"]
        if clause not in valid:
            errors.append("clause reference not in playbook: {!r}".format(clause))
        elif clause in seen:
            errors.append("more than one issue for clause: {!r}".format(clause))
        seen.add(clause)
    return errors


def main():
    parser = argparse.ArgumentParser(description="Redline a document against a playbook with Claude.")
    parser.add_argument("--document", required=True, help="e.g. eval_data/original/document.txt")
    parser.add_argument("--playbook", default="playbook.json")
    parser.add_argument("--output", required=True, help="e.g. redline_output/original-v4.json")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY first, e.g.  export ANTHROPIC_API_KEY=sk-ant-...")

    document, playbook = load_inputs(args.document, args.playbook)
    print("Loaded {} ({} characters) and {} ({} clauses).".format(
        args.document, len(document), args.playbook, len(playbook)))

    issues = find_issues(api_key, document, playbook)
    print("Found {} issues:".format(len(issues)))
    for issue in issues:
        print("  - {}".format(issue["playbook_clause_reference"]))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    print("Wrote {} issues to {}".format(len(issues), args.output))

    errors = validate_output(issues, playbook, document)
    if errors:
        for error in errors:
            print("ERROR: " + error, file=sys.stderr)
        sys.exit("Validation failed with {} error(s).".format(len(errors)))
    print("Validation passed: all {} issues quote the document exactly and use valid, distinct "
          "playbook clause names.".format(len(issues)))


if __name__ == "__main__":
    main()
