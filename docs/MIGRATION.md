# Monorepo Migration Notes

Date: 2026-03-31

## What Changed

- The repository root at this workspace was initialized as the canonical git repo.
- The nested git metadata at `top_reco/.git` was moved to `top_reco/.git_nested_backup`.
- The `top_reco/` directory now behaves as a normal subproject folder inside one repo.

## Why

This removes nested git boundaries so all pipeline components can be versioned together:

- root orchestration and wrappers
- `top_reco/` canonical package
- `data_processing/`
- `analysis/`
- `automation/`
- `docs/`

## Safety and Recovery

The previous nested history is preserved locally in `top_reco/.git_nested_backup`.
Nested backup HEAD at cutover: `237fc21`.

To inspect nested-repo state from before cutover:

```bash
cd top_reco
GIT_DIR=.git_nested_backup GIT_WORK_TREE=. git log --oneline -n 20
GIT_DIR=.git_nested_backup GIT_WORK_TREE=. git status --short
```

To temporarily restore the nested repo layout (if ever needed):

```bash
cd top_reco
mv .git_nested_backup .git
```

## Recommended Next Steps

1. Create an initial root commit promptly so the unified repo has a safety checkpoint.
2. Decide whether to preserve old nested commit history in the root history (for example via subtree/filter-repo) or treat this as a clean monorepo baseline.
3. Configure the root remote and push from the unified root once validated.
