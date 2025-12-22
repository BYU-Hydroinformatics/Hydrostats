Contributing to Hydrostats
==========================

We welcome contributions from the community! Whether you're fixing bugs, adding new metrics,
improving documentation, or enhancing the codebase, your contributions help make Hydrostats better.

Getting Started
^^^^^^^^^^^^^^^

Before you begin, ensure you have the following installed:

- `Git <https://git-scm.com/>`_
- `Git LFS <https://git-lfs.com/>`_ (required for downloading large sample data and image files)
- `Python 3.10+ <https://www.python.org/>`_
- `uv <https://docs.astral.sh/uv/>`_

Setting Up Your Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Fork and Clone the Repository**

   .. code-block:: bash

       git clone https://github.com/BYU-Hydroinformatics/Hydrostats.git
       cd Hydrostats
       git lfs install
       git lfs pull

2. **Install Dependencies**

   .. code-block:: bash

       uv sync

Development Workflow
^^^^^^^^^^^^^^^^^^^^

1. **Create a Feature Branch**

   Use a descriptive branch name for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name
       # or
       git checkout -b fix/issue-number

2. **Make Your Changes**

   When implementing new features or metrics:

   - Write clear, modular code
   - Include comprehensive docstrings using the `NumPy docstring format <https://numpydoc.readthedocs.io/>`_
   - Include relevant references to scholarly papers in docstrings

3. **Write Tests**

   Add test cases for any new functionality in the ``tests/`` directory:

   .. code-block:: bash

       # Run tests to ensure nothing is broken
       uv run pytest

   - Aim for high code coverage
   - Test edge cases (NaN, Inf, negative, and zero values)
   - Test with different data structures (lists, NumPy arrays, Pandas Series/DataFrames)

4. **Ensure Code Quality**

   - Follow PEP 8 style guidelines
   - Run the full test suite before committing

5. **Commit and Push**

   .. code-block:: bash

       git add .
       git commit -m "Clear description of your changes"
       git push origin feature/your-feature-name

6. **Create a Pull Request**

   - Go to `GitHub <https://github.com/BYU-Hydroinformatics/Hydrostats>`_ and create a new pull request
   - Provide a clear title and description
   - Link any related issues
   - Wait for continuous integration pipeline to pass

Contributing Metrics
^^^^^^^^^^^^^^^^^^^^

When adding new metrics or analysis functions:

- Ensure the metric is mathematically sound and well-documented
- Include references to the scholarly paper or source where the metric is defined
- Add usage examples in the docstring
- Test with various hydrologic datasets
- Document the metric's advantages and limitations

Contributing to the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation can be built locally with this command:

   .. code-block:: bash

       uv run --only-group docs sphinx-build docs/ docs/_build

It will generate the documentation files in the ``docs/_build`` directory. You can open that directory
and open the ``index.html`` file to locally view the docs.

Important Guidelines
^^^^^^^^^^^^^^^^^^^^

- **Code Style:** Follow PEP 8 guidelines. Code should be clean and well-commented.
- **Documentation:** All functions and classes must include NumPy-style docstrings.
- **Testing:** Write tests for new features. Ensure all tests pass before submitting a PR.
- **Pull Requests:** Always create pull requests before merging to the main branch. This allows continuous integration testing and code review.
- **Communication:** If you're working on a large feature, open an issue first to discuss it with the maintainers.

For BYU-Hydroinformatics Members
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please make pull requests before merging changes to the master branch. This ensures continuous
integration testing and maintains code quality standards.
