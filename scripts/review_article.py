#!/usr/bin/env python3
"""
LLM-based editorial review for BESS article series.

Extracts reader-facing text from notes/*/app.py and sends it to Claude
for scoring on clarity, engagement, and editorial quality.

Uses `claude` CLI with the user's existing subscription — no API key needed.

Exit code 0 = pass (all scores >= threshold), 1 = fail.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Minimum score (1-10) to pass. Articles below this on any dimension block commit.
THRESHOLD = 6

CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")

STYLE_GUIDE_PATH = Path(__file__).resolve().parent.parent / "notes" / "STYLE_GUIDE.md"


def _load_style_guide() -> str:
    """Read the project style guide, or return empty string if missing."""
    try:
        return STYLE_GUIDE_PATH.read_text()
    except (OSError, UnicodeDecodeError):
        return ""


REVIEW_PROMPT_HEAD = """\
You are an editorial reviewer for a series of data-driven articles about \
battery energy storage (BESS) merchant economics, published as interactive \
Streamlit pages. The target audience is asset owners, investors, and \
commercial teams in the European energy storage industry.

The audience is energy industry professionals — asset owners, traders, \
investors. Terms like FCR, aFRR, FEC, DA, ID, SoC, BESS are standard \
vocabulary and do NOT need explanation.

Score the article text below on five dimensions (1-10 each). Be strict — \
a 7 means "good, publishable", a 5 means "needs work".

Dimensions:
1. **clarity** — Is the argument easy to follow? Are claims supported \
immediately, or do they hang? Are abbreviations explained?
2. **engagement** — Does the opening hook the reader? Is there a clear \
"so what" for the target audience? Would someone share this?
3. **structure** — Does it follow a logical progression? Are sections \
self-contained and headings informative? Is there an intro before data?
4. **rigour** — Are projections hedged appropriately? Are caveats stated? \
Is there a clear boundary between data and interpretation?
5. **conciseness** — Is every paragraph earning its place? Any filler, \
repetition, or over-qualification?
6. **rhythm** — Does the text read at a steady pace, or do specific \
sentences/paragraphs make the reader stumble? Rhythm breaks are local \
tripwires distinct from structure (which is about ordering): hunt for \
the patterns below. These are the fixes a rhythm-focused editor would \
make on a first read — imagine reading aloud and note every place the \
flow stalls.

Rhythm-break patterns to flag (every instance; do NOT merge):
- **Parenthetical/em-dash definition buried inside an impact sentence.** \
If the bold takeaway also carries a side-definition ("X — the Y-per-Z \
bill — dominates"), the sentence has two jobs and hits weakly. Split: \
define first, hit second.
- **Speed-bump caveats right after a list or right before a chart.** Two \
"one caveat…" / "one exception…" paragraphs stacked before the visual \
kill momentum. Push caveats into methodology or fold into the caption.
- **Inconsistent parallelism in numbered findings.** If the text \
announces "three things jump out" and then items 1–2 are findings but \
item 3 is a FAQ, sidebar, or definition, the rhythm breaks. Either \
restore parallelism or move the odd item out of the list.
- **Forward spoilers.** A finding that deflates the next section's \
reveal ("like DoD" just before the DoD section opens). Remove the tell.
- **Reopened cases after a conclusion.** A section that re-explains the \
claim before giving the mechanism ("this is counterintuitive…", "many \
people think…", "but actually…") pads the top. Start with the mechanism.
- **Redundant bridge sentences before a heading.** "And then there's X \
— which deserves its own section." The heading is the bridge.
- **Filler adverbs and meta-commentary.** "So", "really", "actually", \
"it's worth noting that" — cut unless they carry load.

List EVERY specific issue you see across all six dimensions — do not \
cap the count, do not filter by score, do not merge related issues into \
one. Keep each suggestion under 40 words. Err on the side of including \
more comments rather than fewer — the reader will triage.

Rules for suggestions (read carefully — prior reviews had a high false-\
positive rate on these):

1. **Quote the exact problematic text verbatim in every suggestion.** If \
you cannot find the quoted text by a literal substring search of the \
article, DO NOT INCLUDE THE SUGGESTION — you are hallucinating. Never \
invent sections, placeholders, empty divs, or other content that isn't in \
the text you were given.

2. **Respect the two-layer structure.** Articles have:
   - A **main body** for a reader who wants the finding fast.
   - A **methodology / expander section** for a reader who wants the \
physics and citations.
   Restatement between layers is INTENTIONAL, not duplication. Do not \
flag a main-body finding that reappears in a methodology expander, or \
vice versa. Only flag true *within-layer* duplication.

3. **Do not demand sources for standard industry knowledge.** These \
figures are taken as given for this audience and do NOT need citation: \
180 €/kWh pack CAPEX, 25 °C reference temperature, 0.5C baseline rate, \
8760 h/yr, 70% SoH warranty floor, typical EU day-ahead spreads, standard \
degradation exponents in the 0.5–1.5 range.

4. **Illustrative examples do not need calibration sources.** Phrases \
like "one pack hits EOL at year 10, the other at 14" are intuition \
pumps, not model outputs — the baseline point is disclosed elsewhere. \
Do not flag them as "lacking operating assumptions".

5. **Mathematical notation belongs in methodology.** Symbols like z_cyc, \
Ea, DoD^1.5, Arrhenius exponents are appropriate inside a methodology or \
equations section. Only flag them if they appear in the opening narrative \
before the kernel has been introduced.

6. **Do not critique template boilerplate.** Series cross-links ("Next in \
this series"), footers, takeaway callouts, and standard section headings \
are harness elements. Leave their placement and wording alone unless the \
*content* is wrong.

7. **Forward references are a deliberate device.** A brief main-body \
mention with "details in the methodology below" is working as intended. \
Do not flag them as "forward reference without context".

8. **Apply the project style guide below** to every suggestion. The style \
guide is the source of truth on voice, structure, non-native-reader \
language, interactive-UI rules, and editorial stance. Cite the relevant \
section in your suggestion when the fix is driven by a rule (e.g. \
"[style §7: drop 'every']"). If a construction in the article contradicts \
the style guide, flag it — don't let it through.

9. **Every suggestion MUST include a concrete replacement.** Each item in \
the suggestions array has three fields: `dimension`, `before` (verbatim \
text from the article, copy-pasted), and `after` (your proposed rewrite \
that fixes the issue). If you cannot write a concrete `after`, you do \
not have a publishable suggestion — drop it. Do not return vague \
guidance like "consider clarifying" without a rewrite. `after` can be an \
empty string ONLY when you are proposing to delete `before` entirely.

---

PROJECT STYLE GUIDE (source of truth — supersedes any conflicting rule):

{STYLE_GUIDE}

---

Now score the article on all five dimensions and list every issue that \
survives these rules, with before/after on each.

Return ONLY valid JSON, no other text.\
"""


def build_review_prompt() -> str:
    style = _load_style_guide() or "(style guide file missing — apply project conventions inferred from the article)"
    return REVIEW_PROMPT_HEAD.replace("{STYLE_GUIDE}", style)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "clarity": {"type": "integer", "minimum": 1, "maximum": 10},
        "engagement": {"type": "integer", "minimum": 1, "maximum": 10},
        "structure": {"type": "integer", "minimum": 1, "maximum": 10},
        "rigour": {"type": "integer", "minimum": 1, "maximum": 10},
        "conciseness": {"type": "integer", "minimum": 1, "maximum": 10},
        "rhythm": {"type": "integer", "minimum": 1, "maximum": 10},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["dimension", "before", "after", "rationale"],
            },
        },
    },
    "required": ["clarity", "engagement", "structure", "rigour", "conciseness", "rhythm", "suggestions"],
}


def extract_article_text(filepath: Path) -> str:
    """Extract reader-facing text from app.py (st.markdown, render_* calls)."""
    source = filepath.read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    target_funcs = {
        "st.markdown", "st.write", "st.caption",
        "render_standfirst", "render_takeaway", "render_chart_title",
        "render_chart_caption", "render_annotation", "render_footer_note",
        "render_closing", "render_header",
    }

    # Streamlit UI controls — emit a placeholder marker so the reviewer sees
    # that a section between two markdown blocks has interactive content, not
    # an empty gap. Without these markers the reviewer flags sections like
    # "Build your own schedule" as empty because the widgets (buttons,
    # sliders, chart) never pass through st.markdown.
    ui_markers = {
        "st.button": "[interactive element: button]",
        "st.slider": "[interactive element: slider]",
        "st.radio": "[interactive element: radio toggle]",
        "st.selectbox": "[interactive element: dropdown]",
        "st.checkbox": "[interactive element: checkbox]",
        "st.number_input": "[interactive element: number input]",
        "st.plotly_chart": "[interactive element: plotly chart]",
        "st.pyplot": "[interactive element: matplotlib chart]",
        "st.altair_chart": "[interactive element: altair chart]",
        "st.metric": "[interactive element: metric display]",
        "st.dataframe": "[interactive element: data table]",
        "st.table": "[interactive element: static table]",
    }

    # Collect (source_line, text) so the extracted reader view follows source
    # order — markdown blocks and UI placeholders interleave the way the page
    # actually reads.
    items: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                name = f"{func.value.id}.{func.attr}"
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name in target_funcs:
                for arg in node.args:
                    s = _extract_string(arg)
                    if s:
                        items.append((lineno, s))
                for kw in node.keywords:
                    s = _extract_string(kw.value)
                    if s:
                        items.append((lineno, s))
            elif name in ui_markers:
                items.append((lineno, ui_markers[name]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Prose-length string constants from anywhere in the module —
            # picks up descriptive fields embedded in data dicts (preset
            # descriptions, tooltip copy) that reach the UI indirectly. Filter
            # keeps out CSS classnames, dict keys, short labels, and similar.
            s = node.value
            if len(s) >= 40 and " " in s:
                items.append((lineno, s))

    items.sort(key=lambda x: x[0])

    # Dedupe while preserving order. Collapse consecutive identical UI markers
    # (e.g. five preset buttons in a row → one marker) but keep different
    # markers distinct so the reader sees the mix.
    texts: list[str] = []
    seen: set[str] = set()
    last_marker: Optional[str] = None
    for _, text in items:
        is_marker = text.startswith("[interactive element:")
        if is_marker:
            if text == last_marker:
                continue
            last_marker = text
        else:
            last_marker = None
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)

    return "\n\n".join(texts)


def _extract_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        # For f-strings, keep only the literal-constant parts and drop the
        # interpolated expressions entirely. Emitting "{...}" for the dynamic
        # bits trips up the reviewer, which reads it as an unfilled template
        # placeholder and raises false "empty section" flags.
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
        return "".join(parts)
    return None


def review_with_claude(text: str) -> dict:
    """Send article text to Claude CLI for review, return parsed JSON."""
    prompt = f"{build_review_prompt()}\n\n---\n\nARTICLE TEXT:\n\n{text}"

    cmd = [
        CLAUDE_BIN,
        "-p",
        "--output-format", "json",
        "--json-schema", json.dumps(JSON_SCHEMA),
        "--max-turns", "3",
        "--model", "opus",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("\n  Review timed out — skipping (won't block commit)", file=sys.stderr)
        return {}

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"\n  claude CLI failed (rc={result.returncode}) — skipping", file=sys.stderr)
        return {}

    # structured_output may be populated even when the CLI reports is_error
    # (e.g. 400 on a trailing turn after structured output was returned).
    structured = response.get("structured_output")
    if structured and isinstance(structured, dict):
        return structured

    if response.get("is_error") or result.returncode != 0:
        msg = response.get("result", f"rc={result.returncode}")
        print(f"\n  claude error: {msg} — skipping", file=sys.stderr)
        return {}

    # Fallback: try parsing "result" as JSON
    result_text = response.get("result", "")
    if isinstance(result_text, str) and result_text.strip().startswith("{"):
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass

    print(f"\n  No structured output in response — skipping", file=sys.stderr)
    return {}


def format_review(filepath: Path, review: dict) -> tuple[str, bool]:
    """Format review output. Returns (text, passed)."""
    if not review:
        return f"  {filepath.name}: review skipped (CLI unavailable)\n", True

    dims = ["clarity", "engagement", "structure", "rigour", "conciseness", "rhythm"]
    scores = {d: review.get(d, 0) for d in dims}
    avg = sum(scores.values()) / len(scores)
    passed = all(s >= THRESHOLD for s in scores.values())

    lines = [f"\n  {'Score':14s}"]
    for dim in dims:
        score = scores[dim]
        bar = "█" * score + "░" * (10 - score)
        flag = " ✗" if score < THRESHOLD else ""
        lines.append(f"  {dim:14s} {bar} {score}/10{flag}")

    lines.append(f"  {'':14s} {'─' * 10}")
    lines.append(f"  {'average':14s} {' ' * 10} {avg:.1f}/10")

    suggestions = review.get("suggestions", [])
    if suggestions:
        lines.append("")
        width = len(str(len(suggestions)))
        for i, s in enumerate(suggestions, 1):
            dim = s.get("dimension", "?")
            before = (s.get("before") or "").strip()
            after = (s.get("after") or "").strip()
            rationale = (s.get("rationale") or s.get("suggestion") or "").strip()
            # Back-compat for older reviews that only emit `suggestion`:
            if not before and not after and rationale:
                lines.append(f"  {i:>{width}}. [{dim}] {rationale}")
                continue
            header = f"  {i:>{width}}. [{dim}] {rationale}" if rationale else f"  {i:>{width}}. [{dim}]"
            lines.append(header)
            if before:
                lines.append(f"  {'':>{width}}  BEFORE: {before[:200]}{'…' if len(before) > 200 else ''}")
            after_disp = after if after else "(delete)"
            lines.append(f"  {'':>{width}}  AFTER:  {after_disp[:200]}{'…' if len(after_disp) > 200 else ''}")

    if not passed:
        lines.append(f"\n  ✗ Blocked: score below {THRESHOLD} on {', '.join(d for d, s in scores.items() if s < THRESHOLD)}")

    return "\n".join(lines), passed


def main():
    files = sys.argv[1:] if sys.argv[1:] else []

    if not files:
        root = Path(__file__).resolve().parent.parent / "notes"
        files = [str(p) for p in root.glob("*/app.py") if p.parent.name != "_template"]

    if not Path(CLAUDE_BIN).exists():
        print("  claude CLI not found — skipping editorial review", file=sys.stderr)
        return 0

    exit_code = 0
    for filepath_str in files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            continue

        text = extract_article_text(filepath)
        if not text.strip():
            continue

        print(f"\n{filepath.relative_to(filepath.parent.parent.parent)}:")
        print("  Reviewing with Claude...", end="", flush=True)

        review = review_with_claude(text)
        output, passed = format_review(filepath, review)

        print("\r" + " " * 40 + "\r", end="")  # clear "Reviewing..." line
        print(output)

        if not passed:
            exit_code = 1

    if exit_code == 0 and files:
        print("\nEditorial review: all articles pass.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
