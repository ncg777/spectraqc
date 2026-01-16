# Release Process

## Release Steps

1. **Update versions**
   - Bump the project version in `pyproject.toml`.
   - Update any version references in documentation, changelog, or CLI help output.

2. **Regenerate derived artifacts**
   - Rebuild any generated files (schemas, docs, or data artifacts) so they reflect the new version.
   - Ensure generated files are committed alongside the release changes.

3. **Run tests and validations**
   - Run the full test suite.
   - Execute schema validation and any other automated checks.

4. **Build distributions**
   - Build source and wheel distributions.
   - Verify the artifacts exist in the `dist/` directory.

5. **Tag the release**
   - Create an annotated tag using the release version (e.g., `vX.Y.Z`).

6. **Publish to PyPI**
   - Upload the build artifacts to PyPI.
   - Confirm the release is visible on PyPI.

## Release Validation Checklist

- [ ] Version updated in `pyproject.toml` and any docs/changelog references.
- [ ] Derived artifacts regenerated and committed.
- [ ] `spectraqc --version` reports the expected release version.
- [ ] Full test suite passes.
- [ ] Schema validations pass.
- [ ] Distributions built successfully (`dist/` contains sdist and wheel).
- [ ] Git tag `vX.Y.Z` created and pushed.
- [ ] Release artifacts published to PyPI and verified.
