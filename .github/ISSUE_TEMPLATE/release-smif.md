---
name: Release smif
about: Make a new release
title: Release smif vX.X.X
labels: ''
assignees: ''

---

Release checklist:
- [ ] create a release branch
- [ ] update CHANGELOG with notes on features and fixes since last release.
- [ ] update AUTHORS if necessary
- [ ] create and push an annotated tag
- [ ] open a pull request
- [ ] wait (~15 minutes) for tests to pass
- [ ] check that deploy stage ran as expected, new version available on https://pypi.org/project/smif/#history
- [ ] create a GitHub release for the tag
- [ ] wait (~1 day) for conda bot to open PR on https://github.com/conda-forge/smif-feedstock
- [ ] check dependencies/versions/lints and merge to release conda package

See more notes in https://smif.readthedocs.io/en/latest/developers.html#releases
