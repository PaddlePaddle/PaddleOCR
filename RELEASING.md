# PaddleOCR Release SOP

[中文](./RELEASING_cn.md)

## Scope

This document describes the standard release process for PaddleOCR.

## Release Types

The current process supports the following two release types:

- `bump patch`: for example, `3.4.0 -> 3.4.1`
- `bump minor`: for example, `3.4.x -> 3.5.0`

For `bump major`:

- The current process does not directly support this scenario
- A major release process should be discussed and designed separately, for example by introducing an additional branch or a new preparation flow

## Release Principles

- Daily development happens on `main`
- Official releases are made only from `release/X.Y`
- Official tags must use `vX.Y.Z`
- Do not create official release tags on `main`

## Standard Release Process

### 1. Confirm the release target

First confirm which release line this release belongs to and what the target version is.

Examples:

- Release `3.4.1` on `release/3.4`
- Release `3.5.0` on `release/3.5`

### 2. Create or switch to the release branch

If the release line does not exist yet, create the corresponding `release/X.Y` branch from `main`.  
If it already exists, switch to that branch and continue release preparation there.

Requirements:

- One minor release line corresponds to one fixed `release/X.Y` branch
- That branch should contain only the release content and patch fixes for that line

### 3. Pick release content from main

Based on the release scope, `cherry-pick` the required commits from `main` into `release/X.Y` until the release content is ready.

Requirements:

- Only pick what is needed for the current release
- Avoid bringing unrelated new features into the release branch
- If there are release-specific fixes on the release branch, keep them limited to the current release scope

### 4. Complete pre-release checks

Complete pre-release checks on `release/X.Y`.

At minimum, this should include:

- The current branch is correct
- The working tree is clean
- The version is as expected
- Key functionality is verified
- Required tests, builds, packaging, and regression checks have passed
- Release notes are ready

### 5. Create the official tag

Once the release is ready, create the official tag on `release/X.Y`:

- The tag format must be `vX.Y.Z`

Examples:

- `v3.4.1`
- `v3.5.0`

Requirements:

- The tag must be created on the release branch for the current official release
- Do not use development tags as official release tags

### 6. Publish the GitHub Release

Create a GitHub Release based on the official tag for this release.

### 7. Update dependency constraints and release notes

If this release is the first release of a new minor line, or if it changes PaddleX dependency requirements, update the related release materials before or after the official tag is published.

At minimum, this should include:

- Checking whether the `paddlex` dependency constraints in `pyproject.toml` match the target release
- Checking whether installation docs, upgrade notes, and release notes are aligned with the release version

Completion criteria:

- The `paddlex` dependency constraints match the target release
- The version information in the documentation matches the released version

### 8. Sync the release branch lineage back to main

After the first official release of a new `release/X.Y` line is completed, sync the lineage of that `release/X.Y` branch back to `main`.

This is a fixed step in the current workflow and should be done at least once for each new minor release line.

Purpose:

- Ensure `main` correctly reflects that the release line has produced an official version
- Keep subsequent development and release cadence aligned

Requirements:

- Perform this once after the first official release of each new `release/X.Y`
- For later patch releases on the same `release/X.Y`, it is usually not necessary to repeat it

### 9. Move on to the next development cycle or patch release

After the release:

- `main` continues with ongoing development
- `release/X.Y` continues to maintain that release line

If more patch releases are needed later on the same line:

- Continue preparing patches on `release/X.Y`
- `cherry-pick` from `main` as needed
- Repeat the relevant steps in this SOP

## How to Handle Different Bump Types

### Patch Release

Applicable scenarios:

- Fixing production issues
- Small compatibility fixes
- Documentation, dependency, or stability patches

How to handle it:

- Continue preparing changes on the existing `release/X.Y`
- Create the next patch tag, for example `v3.4.2`

### Minor Release

Applicable scenarios:

- Starting a new release line
- Releasing the next stable version, for example `3.5.0`

How to handle it:

- Create a new `release/X.Y` from `main`
- Prepare the release following the standard process
- Create the first official tag, for example `v3.5.0`
- Update `paddlex` dependency constraints if needed
- Sync the lineage of that release branch back to `main` after the release

### Major Release

Current conclusion:

- It is not included in this SOP for now
- A separate process needs to be discussed and designed later

Before a dedicated solution is defined, do not directly reuse the current minor/patch process for a major release.

## Release Checklist

Before each release, confirm the following:

- The target version and corresponding release branch have been confirmed
- The content on the release branch is ready
- The release scope has been finalized
- Key tests and regression checks have passed
- Release notes are ready
- The official tag uses the `vX.Y.Z` format
- The GitHub Release has been created
- If this is the first official release on `release/X.Y`, its lineage has been synced back to `main`

## Daily Maintenance Recommendations

- New features should go to `main` first
- Official releases should always be performed through `release/X.Y`
- Patch releases should always be maintained on the corresponding `release/X.Y`
- For each new `release/X.Y`, sync its lineage once after the first official release

If the current process changes, this document should be updated accordingly.
