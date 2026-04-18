# Withdrawal gate (critic gate 3 M1)

This document defines the quantitative decision criteria for whether to
continue developing aa-animator past the v0.0.1 demo phase.

## Timeline

1. v0.0.1 mini PoC output sent to Discord (internal channel)
2. 30-second demo GIF posted to X (Twitter) and Hacker News Show HN
3. **48 hours later**: apply the decision matrix below

## Decision matrix

| Condition | Decision |
|---|---|
| X impressions < 100 **OR** likes < 3 | **Formal withdrawal** — archive repo, reclaim engineering time |
| impressions < 500 **OR** GitHub stars < 5 | **Scope reduction** — demote to minor personal tool, no PyPI release, no v0.1 full implementation |
| impressions >= 500 **AND** (likes >= 10 **OR** stars >= 20) | **Proceed to v0.1** — full implementation per architecture.md |

## A/B blind evaluation

In addition to public metrics, conduct an internal blind evaluation:

- Present 3 evaluators with two unlabelled clips:
  - Clip A: v0.0.1 aa-animator output
  - Clip B: `aa_animator.py` slime_dance preset (existing tool)
- If 2 of 3 evaluators prefer Clip A, this is a GO signal even if
  public metrics are borderline

## Rationale

The "DeepAA trap": DeepAA reached 1569 stars but has near-zero
production adoption. Ghostty's viral +boo animation was driven by
Mitchell Hashimoto's personal brand and the simultaneous Ghostty v1
release — not by the underlying technique. This gate prevents building
v0.1 full infrastructure against a demand signal that does not exist.

## What withdrawal looks like

- Tag v0.0.1, push to GitHub (public)
- Close all open issues with label `withdrawn`
- Update README with "development paused" banner
- No PyPI publish
- Memory file updated: `aa-animator-project.md` marked WITHDRAWN

## What proceeding looks like

- Begin Day 1 of the 20-hour TDD plan (architecture.md)
- Create GitHub milestones for v0.1.0 and v0.2.0
- Open PyPI Trusted Publishing configuration
