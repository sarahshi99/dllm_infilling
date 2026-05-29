# Main Bilingual Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the validated A6000/result-archive documentation on `main` and add Chinese counterparts for all current English Markdown files.

**Architecture:** Merge the experiment and archive branches into `main`, keep the English Markdown files unchanged as source records, and add same-path `.zh.md` translations. Update `AGENTS.md` so future Markdown work follows the English-first, Chinese-translation policy.

**Tech Stack:** Git branches/worktrees, Markdown, local unit tests, `git diff --check`, GitHub push to `main`.

---

## Task 1: Merge validated documentation branches

**Files:**
- Merge branch: `exp/a6000-midcons-longrescue`
- Merge branch: `docs/result-archive`

- [x] Fetch remote `main`.
- [x] Confirm local `main` is up to date.
- [x] Merge `exp/a6000-midcons-longrescue` into `main`.
- [x] Merge `docs/result-archive` into `main`.

## Task 2: Add bilingual policy

**Files:**
- Modify: `AGENTS.md`
- Create: `AGENTS.zh.md`

- [x] Explain PR vs direct merge behavior.
- [x] Add future Markdown rule: write English first, then add `.zh.md` Chinese version.
- [x] Preserve technical paths, commands, model names, metrics, and citations in translation.

## Task 3: Add Chinese counterparts

**Files:**
- Create `.zh.md` next to every current English Markdown file.

- [x] Translate project rules and setup docs.
- [x] Translate scoreboards and result reports.
- [x] Translate result registries and archive notes.
- [x] Translate Superpowers specs and plans as reader-oriented Chinese versions that preserve reproducibility-critical commands and decisions.

## Task 4: Verify and push

**Files:**
- All Markdown files.

- [x] Verify every English `.md` has a same-path `.zh.md`.
- [x] Run unit tests.
- [x] Run `git diff --check`.
- [ ] Commit and push `main`.
