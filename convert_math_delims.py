#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


FENCE_RE = re.compile(r"^\s*(```|~~~)")  # fenced code blocks
# A cheap currency heuristic: $ followed by a digit -> do not treat as math start
CURRENCY_RE = re.compile(r"^\$\d")


def split_inline_code(line: str):
    """
    Splits a line into alternating [non_code, code, non_code, code, ...]
    using backticks. Not perfect for nested backticks, but works well
    for typical markdown.
    """
    return line.split("`")


def convert_inline_math_in_text(text: str) -> str:
    """
    Convert $...$ to \\(...\\) in a chunk that is NOT inside inline code.
    Keeps $$ untouched here (handled separately).
    Avoids escaped dollars and currency.
    """
    out = []
    i = 0
    in_math = False

    while i < len(text):
        ch = text[i]

        # keep escaped dollar
        if ch == "\\" and i + 1 < len(text) and text[i + 1] == "$":
            out.append("\\$")
            i += 2
            continue

        if ch == "$":
            # if it's $$, leave for display handler
            if i + 1 < len(text) and text[i + 1] == "$":
                out.append("$$")
                i += 2
                continue

            # if starting $ looks like currency, keep it literal
            if not in_math and CURRENCY_RE.match(text[i:]):
                out.append("$")
                i += 1
                continue

            # toggle inline math
            out.append("\\(" if not in_math else "\\)")
            in_math = not in_math
            i += 1
            continue

        out.append(ch)
        i += 1

    # if we ended inside math due to unmatched $, revert (fail-safe)
    if in_math:
        # replace the last '\(' back to '$'
        joined = "".join(out)
        joined = joined[::-1].replace(")\\", "$", 1)[::-1]  # reverse trick
        return joined

    return "".join(out)


def convert_display_blocks(md: str) -> str:
    """
    Convert $$...$$ to \\[...\\] across the whole file,
    but skip fenced code blocks.
    """
    lines = md.splitlines(keepends=True)
    out = []
    in_fence = False
    fence_marker = None
    in_display = False
    display_buf = []

    for line in lines:
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker == marker:
                in_fence = False
                fence_marker = None

            # flush any open display (very unlikely, but safe)
            if in_display:
                out.append("$$\n")
                out.extend(display_buf)
                display_buf = []
                in_display = False

            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        # display delimiter line: $$ alone on the line (ignoring whitespace)
        if line.strip() == "$$":
            if not in_display:
                in_display = True
                display_buf = []
            else:
                # close display block -> emit \[ ... \]
                out.append("\\[\n")
                out.extend(display_buf)
                # ensure ending newline before \]
                if len(out) > 0 and not out[-1].endswith("\n"):
                    out.append("\n")
                out.append("\\]\n")
                in_display = False
                display_buf = []
            continue

        if in_display:
            display_buf.append(line)
        else:
            out.append(line)

    # if file ended with unmatched $$, revert safely
    if in_display:
        out.append("$$\n")
        out.extend(display_buf)

    return "".join(out)


def convert_inline_everywhere(md: str) -> str:
    """
    Convert inline $...$ to \\(...\\), skipping fenced blocks,
    and skipping inline code spans.
    """
    lines = md.splitlines(keepends=True)
    out = []
    in_fence = False
    fence_marker = None

    for line in lines:
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker == marker:
                in_fence = False
                fence_marker = None
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        parts = split_inline_code(line)
        for k in range(0, len(parts), 2):  # only non-code parts
            parts[k] = convert_inline_math_in_text(parts[k])
        out.append("`".join(parts))

    return "".join(out)


def main():
    # convert all .md files recursively, excluding typical build dirs
    skip_dirs = {"_site", ".git", ".venv", "node_modules"}

    for p in Path(".").rglob("*.md"):
        if any(part in skip_dirs for part in p.parts):
            continue

        original = p.read_text(encoding="utf-8")
        step1 = convert_display_blocks(original)
        step2 = convert_inline_everywhere(step1)

        if step2 != original:
            p.write_text(step2, encoding="utf-8")
            print(f"Converted: {p}")
        else:
            print(f"Unchanged: {p}")


if __name__ == "__main__":
    main()
