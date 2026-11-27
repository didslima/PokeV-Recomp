# Feature Reference Repositories

## Required prior-art review

Before proposing an architecture or implementing a **new feature or substantial structural change** in PokeV-Recomp, inspect the following projects for existing implementations and lessons that may apply:

1. [Witch-Time](https://github.com/Zexyen/Witch-Time)
2. [SavageWindRecomp](https://github.com/Revivedx/SavageWindRecomp/)
3. [Super Odyssey PC Recomp](https://github.com/Simon3Sarkis/super-odyssey-pc-recomp)
4. [LASRecomp](https://github.com/peiikai/LASRecomp/)

These are the project's priority reference repositories. Treat them as a first stop for prior art, not as the only possible sources.

## Review workflow

For each feature, first define the technical problem and identify the relevant areas to inspect in each repository. Consider all four repositories, but keep the investigation proportional: a repository may be marked not applicable when its architecture or scope offers no useful precedent. Do not assume that similarly named features work the same way.

Look beyond README descriptions. When relevant, inspect:
- Source implementation and architecture.
- Tests, fixtures, and CI workflows.
- Recent commits, closed and open pull requests, and issue discussions that explain design decisions or known limitations.
- Packaging, platform-specific code, migration paths, and failure handling.

Compare applicable approaches and report:
- What each relevant project does, with links to the exact files, commits, PRs, or issues.
- Which approach best fits PokeV-Recomp and why.
- What can be reused or adapted, what must be redesigned, and any compatibility, licensing, or maintenance concerns.
- Test cases and failure modes that PokeV-Recomp should adopt.
- Which repositories were checked and why any were considered not applicable.

Do not copy code blindly. Verify its current state, license, dependencies, platform assumptions, and compatibility with PokeV-Recomp before reuse. Prefer adapting proven patterns over inventing a new solution, but do not force an unsuitable pattern merely because it exists in a reference project.

## Expected feature proposal

Before implementation begins, include a concise **Prior-art review** in the feature plan or AGY task report. If no applicable precedent is found, say so explicitly and explain what was searched. Cite evidence with stable GitHub links rather than relying on unlinked summaries.

This requirement applies to new features and meaningful architectural changes. It does not need to block trivial typo fixes, routine maintenance, or narrowly scoped bug fixes unless the bug clearly relates to an existing reference implementation.

## Existing work exception

This policy is prospective. It does **not** require reopening or redesigning the current Windows auto-updater work (including PR #63). Do not apply this review retroactively to that implementation unless explicitly requested.
