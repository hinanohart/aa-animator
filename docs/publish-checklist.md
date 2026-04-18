# Pre-publish checklist — aa-animator v0.1.0

Run through every item in order. Do not skip.
Items marked **(user)** require manual confirmation before proceeding.

---

## Local verification

- [ ] `pytest tests/ -v` → all 69 pass
- [ ] `rg -i "$(whoami)|/home/$(whoami)|/mnt/c/Users/$(whoami)" src/ tests/ scripts/ docs/ .github/` → 0 matches
- [ ] `git log --all -p | grep -iE "$(whoami)|/home/$(whoami)|/mnt/c/Users/$(whoami)|WSL2"` → 0 matches (git history clean)
- [ ] `python3 -m build --outdir /tmp/aa-animator-dist/` → wheel + sdist generated without errors
- [ ] `twine check /tmp/aa-animator-dist/*` → all PASSED

Build artifact sizes (Day 4, 2026-04-18):
- wheel: `aa_animator-0.1.0.dev1-py3-none-any.whl` — **30 KB**
- sdist:  `aa_animator-0.1.0.dev1.tar.gz`         — **39 KB**

---

## CLI smoke test (requires runtime deps installed)

- [ ] `aa-animator --help` → help text printed, exit 0
- [ ] `aa-animator animate <test.jpg> -o /tmp/test-aa.mp4` → mp4 generated
- [ ] **(user)** Open `/tmp/test-aa.mp4` and confirm motion + glyph quality looks acceptable

---

## n=3 validation rerun

- [ ] Run `pytest tests/ -v -k validation` (or full suite) → `consistent_pass=true` maintained across 3 seeds
- [ ] Flicker std <= 0.01, fg_entropy >= 3.0 in all 3 runs

---

## Version bump (before final publish)

- [ ] Change `version = "0.1.0.dev1"` → `version = "0.1.0"` in `pyproject.toml`
- [ ] Confirm `src/aa_animator_v2/_version.py` matches (or is derived from pyproject.toml)
- [ ] `python3 -m build --outdir /tmp/aa-animator-dist/` → rebuild with final version
- [ ] `twine check /tmp/aa-animator-dist/*` → PASSED again

---

## GitHub

- [ ] **(user)** Review all commits one final time: `git log --oneline`
- [ ] **(user)** Switch repo visibility: Settings → Danger Zone → Make public
- [ ] Confirm GitHub Actions CI passes on the public repo (Ubuntu + macOS matrix)

---

## PyPI publish

- [ ] `git tag v0.1.0 && git push origin v0.1.0` → triggers `release.yml` GitHub Action
- [ ] **(user)** Monitor GitHub Actions → Release workflow → confirm upload to PyPI succeeds
- [ ] **(user)** Visit https://pypi.org/project/aa-animator/ and confirm package page looks correct

---

## Post-publish

- [ ] `pip install aa-animator==0.1.0` in a clean venv → installs cleanly
- [ ] Run quick smoke test in clean venv: `aa-animator --help`
- [ ] Post announcement (Discord, X, or both)
- [ ] Update Demo section of README.md with real output GIF/link

---

## Rollback plan

If a critical issue is found after publish:
1. `pip` does not support deletion; use PyPI yank: https://pypi.org/help/#yanked
2. Yank via: Settings → Manage project → Edit release → Yank release
3. Fix the issue, bump to `0.1.1`, republish
4. Reference: https://pypi.org/help/#yanked
