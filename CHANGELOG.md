# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Changelog automation via release-plz: versions, `CHANGELOG.md`, tags, and GitHub Releases are generated from conventional commit subjects (`feat:` → Added, `fix:` → Fixed) on merges to `main`.
- Required commit/PR subject format (`type: description`), enforced by the Commit format workflow and the lefthook `commit-msg` hook.
- Registry metadata: PyPI `[project.urls]` (Repository, Changelog) and crate `repository` field.

### Removed

- Per-PR manual `CHANGELOG.md` edits and their CI gate (superseded by the commit-format requirement).
