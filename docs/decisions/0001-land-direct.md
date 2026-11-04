# 0001 Land direct to main: no pull requests, one coordinator (`scripts/land-direct`)

Status: accepted (the fleet's decision). Copied and adapted from the fleet template (`direct-landing-template/vbm-json`).

## Decision
Prospecta has **no per-change pull requests**. Workers share one git common dir, so the single coordinator lives there. A worker commits locally (never pushes a branch), then runs
`scripts/land-direct`, which lands the commits on `origin/main` itself: hold check, red-check, fetch, rebase in a scratch worktree, **Tier 1 locally** on that exact combination, fetch again
(main moved: rebase and re-run, 3 tries), plain fast-forward push (never forced). One `flock` on `<common>/land.lock` (exit 15 busy). Exit codes and the JSON last line are in the script header.

- **Tier 1** = `scripts/tier1.sh`: the lander self-test (`scripts/test_land_direct.sh`, local bare remote) plus the pytest suite with `PROSPECTA_TIER1=1`, which skips the Postgres-backed tests
  (`tests/conftest.py`). Under 60 s. The Rust TUI keeps its own workflow (`tui-ci.yml`) and is not in Tier 1.
- **Tier 2** = the full suite with Postgres testcontainers, in CI on a 6 h schedule and on dispatch. It never gates landing.
- **Red** (the only definition): the `tier1` job of main's latest push run of `ci.yml` failed, even while the run is going. Only a `fix(ci):` / `fix(tests):` / revert commit lands on red.
- **Hold**: `scripts/land-direct --hold "<reason>"` writes `<common>/prospecta-land-hold` and stops every landing; `--release` lifts it.
- **Budget**: over 150 changed lines under `prospecta/` and `tui/src/` needs a `Diff-Budget: <note>` trailer.
- No regeneration command (`regen=true`), no protected-path set, no Tier 2 revert watchdog (add it only after Tier 2 runs on schedule on main).

## Consequences
After landing, the branch holds pre-rebase commits: start the next task from a fresh worktree off `origin/main`. No remote worker branches exist. Read the push CI result after every landing.
Edits to the lander, `tier1.sh` and CI need the owner as independent judge. Repo setting for the owner: `delete_branch_on_merge=true`.
