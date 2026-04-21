---
name: gitbutler
version: 0.19.8
description: "GitButler virtual-branch version control — commit, branch, push, squash, reorder commits, manage stacked PRs, and resolve conflicts using `but` instead of raw `git`. Use when user says: commit my changes, create a branch, check what changed, push, make a PR, view diff, stage files, amend a commit, squash commits, undo commits. GitButler manages multiple active branches simultaneously (virtual branches / stacked PRs) without requiring you to switch contexts."
trigger: "but gitbutler"
author: GitButler Team + Hermes integration
---

# GitButler CLI Skill

GitButler replaces raw `git` for all write operations. It manages **virtual branches** — multiple branches active simultaneously in your working tree, enabling stacked PRs without context-switching.

## Core Concept

Instead of one checked-out branch at a time, GitButler keeps ALL branches "applied" (active) simultaneously. Each file change belongs to exactly one virtual branch. This is ideal for coding agents because you never lose work on other features when switching contexts.

**Key difference from vanilla git:** You do NOT `git checkout -b` to create a branch, then switch back to main, etc. Everything stays applied at once.

## Non-Negotiable Rules

1. **Use `but` for all write operations.** Never run `git add`, `git commit`, `git push`, `git checkout`, `git merge`, `git rebase`, or `git stash`. If the user asks for a git command, translate it to `but`.
2. **Always use `--status-after`** on mutation commands so you can verify the result.
3. **Get fresh IDs before every mutation** — run `but status -fv` first to get current file/branch/commit IDs.
4. **`--changes` takes comma-separated IDs or repeated flags.** Not space-separated.

## Read Operations (allowed as-is)

These read-only git commands are fine to use directly:
- `git log`, `git blame`, `git show`, `git diff --stat`

For everything else, prefer `but status -fv`, `but diff`, and `but show`.

## Command Map

| git | but |
|---|---|
| `git status` | `but status -fv` |
| `git add + commit` | `but commit ... --changes <id>` |
| `git checkout -b <name>` | `but branch new <name>` |
| `git push` | `but push` |
| `git rebase -i` | `but move`, `but squash`, `but reword` |
| `git cherry-pick` | `but pick` |
| `git stash` | `but unapply <branch>` (or discard with `but discard`) |

## Essential Commands

### Check workspace state
```bash
but status -fv   # Full verbose — shows file IDs, branch names, commit IDs, stack order
```

### Create a new branch for new work
```bash
but branch new feature/my-feature     # Creates and switches to new virtual branch
# or:
but commit <branch> -c -m "msg" --changes <id>  # -c creates branch if it doesn't exist
```

### Commit changes
```bash
# 1. Get current IDs
but status -fv

# 2. Commit (using file IDs from status output)
but commit my-branch -m "Add feature X" --changes a1,b2 --status-after
```

### Push all branches
```bash
but push           # Pushes ALL applied virtual branches, creates stacked PRs on the forge
```

### Amend into existing commit
```bash
# Get IDs first
but status -fv
# Then amend file change into an earlier commit (rebases dependent commits automatically)
but amend <file-id> <commit-id> --status-after
```

### Reorder / restack commits
```bash
# Move a commit to a different position in the stack
but move c3 c5 --status-after   # uses COMMIT IDs like c3, c5 (from status output)

# Restack one branch on top of another
but move feature/frontend feature/backend --status-after  # uses BRANCH names
```

### Squash commits
```bash
but squash <commit-id> --status-after
```

### Undo last operation
```bash
but undo --status-after   # Reverts to the previous snapshot (careful — can revert more than intended in complex stacks)
```

## Stacked Branches

GitButler's killer feature: multiple branches applied simultaneously.

To stack `feature/logging` on top of `feature/auth`:
```bash
but move feature/logging feature/auth --status-after
```

To make a branch independent again (unstack):
```bash
but move feature/logging zz --status-after   # zz = unassigned / root level
# or:
but branch move --unstack feature/logging
```

## Conflict Resolution

**Never use `git add`, `git checkout --theirs`, etc. during conflict resolution.**

1. `but status -fv` — find commits marked as conflicted
2. `but resolve <commit-id>` — enters resolution mode, puts `<<<<<<<` markers in files
3. Edit the files to remove conflict markers manually
4. `but resolve finish` — finalize (do NOT skip step 3)

## Dependency Locks

If a file was committed on branch A but you want to change it on branch B:
1. Check which branch owns it with `git log --oneline <file>`
2. Stack your branch: `but move feature/b-feature feature/a-feature --status-after`
3. Now commit — the file is now assignable

## JSON Output

For scripting or parsing in agent workflows, use `-j`:
```bash
but status -fv -j    # Returns machine-readable JSON with IDs
```
