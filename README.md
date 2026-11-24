# Redliner Agent

Reviews an NDA against a negotiation playbook (`playbook.json`) with one Claude call and outputs
every issue as JSON:

```json
[
  {
    "text_snippet": "EXACT QUOTE",
    "playbook_clause_reference": "EXACT CLAUSE NAME",
    "suggested_fix": "EXACT REPLACEMENT SUGGESTION"
  }
]
```

## Deliverables

| Deliverable | File |
|---|---|
| Redlining script | `redline.py` |
| Redlining output (JSON) | `redline_output.json` (the provided `bad_document.txt`) |
| Evaluation script | `evaluate.py` |
| Evaluation summary | `evaluation.txt` (the provided `bad_document.txt` vs. its `expected_output.json`) |

## Files

| File | What it does | Requirements |
|---|---|---|
| `redline.py` | The redliner: document + playbook in, issues JSON out | Python 3.8+, standard library only |
| `evaluate.py` | Scores a redliner output against an answer key | Python 3.8+, standard library only |
| `generate-eval-data.py` | Dev tool: builds synthetic test cases from the playbook | Python 3.10+, `pip install anthropic` |
| `eval_data/<case>/` | Test cases: `document.txt` + `expected_output.json`. `eval_data/original/` holds the provided `bad_document.txt` (renamed `document.txt`) and its `expected_output.json` | |
| `redline_output/` | Redliner outputs for every case, named `<case>-v<N>.json` | |
| `eval_report_output/` | Full eval reports per case (`<case>-v<N>.json`: every redline, bucket and judge reason) and `all-v4.txt`, the printed summary of all five cases | |

All scripts read the API key from the `ANTHROPIC_API_KEY` environment variable; it is never in the code.

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Redline the provided document (bad_document.txt)
python3 redline.py --document eval_data/original/document.txt --output redline_output.json

# Score it against the provided expected_output.json
# (add --details for the judge's reasoning, --no-llm to skip the judge)
python3 evaluate.py --output redline_output.json --expected eval_data/original/expected_output.json

# For the other cases, the answer key is found from the file name:
# redline_output/case-01-v4.json -> eval_data/case-01/expected_output.json
python3 evaluate.py --output redline_output/case-01-v4.json

# Generate a new synthetic test case (5-8 planted red flags; drafts the perfect NDA first if it is missing)
python3 generate-eval-data.py --seed 1 --out eval_data/case-01
```

## How the redliner works

1. **Load** the document and playbook.
2. **Format the playbook** clause by clause in a fixed order (what the clause covers, review
   instruction, ideal, acceptable, red flags, example clauses). The content is unchanged;
   `provision_definition` is left out because it is identical to `clause_definition` for every clause.
3. **One Claude call** (`claude-opus-5-5`, adaptive thinking, high effort) with structured outputs.
   The JSON schema restricts `playbook_clause_reference` to the exact playbook clause names, so the
   model cannot return a name that isn't in the playbook.
4. **Validate** the response, then write the file. Validation fails (non-zero exit) if a snippet
   is not an exact quote from the document, a clause name is not in the playbook, or a clause is
   used more than once.

### Prompt rules, and why

Each rule came from a failure seen in an earlier run, written to apply to any NDA rather than to one sample:

| Rule | Failure it addresses |
|---|---|
| At most one issue per playbook clause (the most serious) | Several issues under bundled clauses such as General |
| Only flag problems the playbook covers | Drafting errors (e.g. wrong cross-references) reported as issues |
| Don't report clauses entirely absent from the document | New clauses attached to unrelated sentences |
| Do report a present clause that lacks a protection the playbook requires, quoting where it belongs | Removed protections (e.g. deleted data-security obligations) going unflagged |
| Choose the clause by what the provision is and which section it sits in, not by similar red-flag wording | The same sentence labeled with a different clause than its section |
| Keep consecutive sentences that form one problem together | One problem split across two clauses |
| Concise fixes, lists for multi-part obligations, stay within the clause | Long, prose-only fixes that pulled in other clauses' material |

## How the evaluation works

- **Test cases:**
  - `original`: the provided NDA (`bad_document.txt`) with its human-written answer key.
  - `perfect` (generated): an NDA Claude drafts from the playbook to meet every clause's ideal
    position. Its answer key is empty, so it checks that the redliner doesn't invent problems.
  - `case-NN` (generated): the perfect NDA with 5-8 red flags planted by Claude, and an answer
    key listing them (`generate-eval-data.py`).
  - `reworded-NN` (generated, 1-2 cases): the original NDA reworded and reordered by Claude with
    the same problems, so the human answer key carries over (`--from-original`). It checks whether
    the prompt rules depend on the original's exact wording and layout.
- **Matching:** each predicted snippet is fuzzy-matched to the answer key, then sorted into
  match / misattributed (right text, wrong clause) / missed / not in expected. Precision, recall
  and F1 are computed from that.
- **LLM judge** (`claude-sonnet-5-5`, a different model from the redliner): grades each matched
  fix as ideal / acceptable / still a red flag, and checks whether unmatched flags are fair anyway.
  It sees the clause's review instruction, positions, red flags and example ideal clause, and is
  told to ignore playbook fields that are empty or about a different topic (see Limitations).
- **Generated answer keys are checked** before a case is saved: one issue per clause, no empty
  fields, and every snippet quoted exactly from the generated document.

## Results (v4)

| Case | Precision | Recall | F1 |
|---|---|---|---|
| original (human answer key) | 63.6% | 87.5% | 73.7% |
| perfect (should flag nothing) | 0 flags | | |
| case-01 | 100% | 100% | 100% |
| case-02 | 100% | 100% | 100% |
| case-03 | 100% | 83.3% | 90.9% |

Run `evaluate.py --details` on each output for the judge's fix grades and reasons.

## Limitations

- **Synthetic cases are easier than real ones.** The same model writes the perfect NDA, plants
  the problems and finds them; each planted problem is a single, clearly located change; and
  fuzzy matching counts a longer predicted snippet as a match if it contains the expected one.
  The human-labeled `original` case is the more meaningful score.
- **The playbook has data errors.** The acceptable positions for Notices and for Breach, dispute
  and termination are copies of the Privacy Policy clause's, and Breach's ideal position is empty.
- **One human-labeled document.** The original's answer key reflects one reviewer's judgment;
  some "not in expected" flags are defensible under the playbook.
- **The playbook is read literally.** For example, on the original NDA the redliner flags a
  different Notices sentence than the answer key, because the playbook's red flags don't
  describe the problem the reviewer saw.
- **The judge sees only the flagged text, not the whole document.** It can grade a correct
  one-sentence fix as a red flag when the protection it wants is in a neighboring sentence (two
  Disclosure required by Law fixes in v4). Passing the surrounding text would fix this, at extra cost.
- **One issue per clause** means secondary problems in the same clause go unreported.
