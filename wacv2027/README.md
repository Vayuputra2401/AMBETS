# WACV 2027 submission — "Aligned but Inert"

Official author kit copied from `LAST/Paper/WACV2027/wacv-2027-author-kit-template`.
Rules reference: `LAST/Paper/WACV2027/WACV2027_RULES.md`.

## Build

    pdflatex main; bibtex main; pdflatex main; pdflatex main

## Hard rules (desk-reject triggers)

1. **8 pages max excluding references.** References unlimited.
2. **Do not alter margins or font sizes.** Overriding the body font is an explicit
   desk-reject trigger. Genuine 10pt only -- do NOT port the 8pt override used in older
   drafts for other venues.
3. Broken anonymity; incomplete COI; citations to non-existent material; author-list
   changes after enrollment.

**Round 2 has NO rebuttal and NO revision.** Reviews are released as the decision. Every
objection must be pre-empted inside the paper; declared limitations beat discovered ones.

## Deadlines (AoE)

| Milestone | Date |
|---|---|
| **Paper enrollment** | **Aug 21, 2026** |
| Paper submission | Aug 28, 2026 |
| Supplementary | Aug 30, 2026 |
| Decisions | Oct 9, 2026 |

## Track

`\usepackage[review,applications]{wacv}` -- biomedical is listed explicitly under
Applications in the CFP. A track option is MANDATORY; omitting it prints a giant `:-(`
and raises a LaTeX error.

Set `\def\wacvPaperID{...}` once enrolled (currently `*****`).

## Numbers discipline

Values written plainly are measured and traceable to `evals/` in this repo. Anything not
yet measured uses `\pending{...}`, which renders as loud red, and `\pendingcount` prints
the remaining count. **Never replace a `\pending` with a plausible-looking value** -- fill
it from a results file or leave it. A stripped placeholder is how a fabricated result
enters a paper.

## Status

- `sec/0_abstract.tex` -- DONE, written from final scope-matched numbers (1971 chars)
- `sec/1_intro` .. `sec/6_discussion` -- stubs; port from `../wacv paper/main_target.tex`,
  which has the full LNCS draft with Acts 1 and 2 already backed by data
