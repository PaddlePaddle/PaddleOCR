# Release

## Versioning

The repository is prepared for a Changesets-based release flow.

The main release target is `packages/paddleocr-js`.

## Typical Release Flow

1. Merge changes into the default branch.
2. Add or update changesets that describe user-facing changes.
3. Let the release workflow open or update the versioning pull request.
4. Merge the release pull request.
5. Publish to npm from the automated release workflow.

## Release Requirements

- `NPM_TOKEN` configured in the repository secrets
- GitHub Actions enabled
- package metadata validated before publishing
