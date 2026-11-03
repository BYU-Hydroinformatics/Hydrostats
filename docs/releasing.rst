Releasing
=========

Hydrostats releases are prepared as GitHub drafts and published after review.
Publishing a release starts the GitHub Actions workflow that builds and
publishes the package to PyPI.

Prerequisites
-------------

- Install `uv <https://docs.astral.sh/uv/>`_.
- Install and authenticate the `GitHub CLI <https://cli.github.com/>`_ with
  permission to create releases in the Hydrostats repository.
- Commit and push the release version to GitHub before creating the release.
  The selected commit must contain the same version as the working
  ``pyproject.toml``.

Preparing a release
-------------------

Set the ``version`` in ``pyproject.toml`` to the version being released. Use
``MAJOR.MINOR.PATCH`` for stable releases, or add ``aN``, ``bN``, or ``rcN``
for prereleases; for example, ``2.0.0`` or ``2.0.0rc1``.

Preview the draft details from the repository root:

.. code-block:: bash

   uv run python scripts/create_release.py --dry-run

By default, the script targets ``HEAD``. Use ``--target <ref>`` to select a
different local branch or commit. The selected commit must already be pushed
to GitHub.

When the preview is correct, create the draft:

.. code-block:: bash

   uv run python scripts/create_release.py

The script uses the version as the release title and tag, adds the matching
Read the Docs release-notes link and a comparison link to the previous
published release, and marks alpha, beta, and release-candidate versions as
prereleases. It rejects a version when its GitHub release or tag already
exists.

Review the draft on GitHub, then publish it to trigger the PyPI publishing
workflow.
