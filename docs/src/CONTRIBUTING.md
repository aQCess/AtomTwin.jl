AtomTwin is a high-level design environment and simulation framework built on top of a dedicated performance-focused simulation engine, **Dynamiq**. Dynamiq is developed as an internal Julia submodule of AtomTwin so that engine development can be deliberate and performance-driven, while AtomTwin focuses on user-facing workflows and features. Dynamiq should remain conceptually independent (with its own benchmarks, tests, and internal APIs), but it is stored in the same Git repository as AtomTwin to simplify cloning and package installation.

***

### Scope, authorship, and expectations

- AtomTwin is currently a **single-author** package maintained in limited spare time.  
- Contributions are very welcome, but active supervision and turnaround on reviews may be slow.  
- Substantial changes are best proposed and discussed in an issue or draft MR before significant development effort.  

For now, contributors should be reasonably experienced with Julia and Git and comfortable making well-scoped, well-tested changes.

***

### Project structure and philosophy

- **AtomTwin**: High-level, stable user-facing API, orchestration, and examples.  
- **Dynamiq**: Low-level, performance-oriented engine code, organized as a Julia submodule (e.g. under `src/Dynamiq/`), optimized with benchmarks and targeted tests.  
- **Examples and docs**:
  - Example scripts in `examples/` are intended to demonstrate AtomTwin functionality and serve as regression tests.  
  - Ground truth files live in `test/examples_src`, generated via `Documenter.jl` and `Literate.jl`.  

If in doubt, put user-facing features in AtomTwin and performance-critical internals in Dynamiq.

Even though AtomTwin and Dynamiq share a Git repository, the **AtomTwin** and **Dynamiq** modules are developed with a clear separation of concerns: AtomTwin as the high-level interface, Dynamiq as the engine.

***

### Cloning AtomTwin

AtomTwin and Dynamiq live in a single Git repository. A typical clone flow is:

```bash
git clone <AtomTwin-repo-url>
cd AtomTwin
```

No extra submodule commands are required; the Dynamiq engine code is part of this repository under the `src/Dynamiq/` directory.

***

### Setting up the Julia environment

From the AtomTwin root:

```bash
julia --project=.
```

In the Julia REPL:

```julia
using Pkg
Pkg.instantiate()
```

This installs dependencies for AtomTwin, Dynamiq, and the documentation/examples toolchain.

To run the tests:

```bash
ATOMTWIN_RUN_EXAMPLES=true julia --project=. -e 'using Pkg; Pkg.test("AtomTwin")'
```

This should exercise AtomTwin’s tests, including checks tied to `test/examples_src`.

***

### Working on AtomTwin (high-level API and examples)

AtomTwin owns the public user-facing interface.

Typical changes:

- Add or refine atomic physics models or instructions.  
- Extend or add **example scripts** to demonstrate features.  
- Improve documentation and narrative examples.

Guidelines:

- Every major new user-visible feature or feature set should be covered by:
  - An example script in `examples/`, and  
  - A corresponding ground-truth test under `test/examples_src` for testing and documentation consistency.  

***

### Working on Dynamiq (engine and performance)

Dynamiq is the performance-focused backend, implemented as an internal Julia module but developed with its own benchmarks and engine-level design in mind.

Typical changes:

- Improvements to integrators, solvers, or memory structures.  
- New detectors or modifiers.  
- General improvements to numerical stability.

Guidelines:

- Treat changes as deliberate engine work; avoid mixing them with unrelated AtomTwin UI/API changes.  
- Use benchmarks and targeted tests to validate improvements and catch regressions.  
- If a change may affect AtomTwin behavior, coordinate updates to examples and tests before merging.

**Engine independence**: Even though Dynamiq lives in the same repository, it should be written so that its core concepts and APIs are reasonably self-contained, making it feasible to extract or mirror Dynamiq as a standalone package in the future if needed.

***

### Branches and release policy

- **`main`**: Stable branch.  
  - Intended for users who want the most recent *stable* version of AtomTwin and Dynamiq.  
  - Only well-tested, reviewed changes should be merged here.  

- **`devel`**: Latest development branch (use at your own risk).  
  - Integrates new features and larger refactors before they are considered stable.  
  - May occasionally break; suitable for contributors and early adopters, not for production use.

Typical flow: feature and fix branches are merged into `devel` first; once they are tested and considered stable on `devel`, they are merged or cherry-picked into `main`.

***

### Feature and bugfix branches

When contributing, please work on short-lived topic branches rather than pushing directly to `main` or `devel`.

Recommended naming:

- **Features**:  
  - `feat-short-descriptor`  
  - Example: `feat-new-integrator`, `feat-improved-api`.

- **Bug fixes**:  
  - `fix-short-descriptor`  
  - Example: `fix-qme-loop`, `fix-doc-typo`.

- **Other work (optional conventions)**:  
  - `refactor-short-descriptor` for refactoring.  
  - `perf-short-descriptor` for performance work in Dynamiq.

Workflow:

1. Branch off from **`devel`** (or `main` for very small/urgent fixes):

   ```bash
   git checkout devel
   git pull
   git checkout -b feat-short-descriptor
   ```

2. Commit changes on your topic branch.
3. Open a merge request targeting `devel`.
4. After changes are tested and reviewed on `devel`, they may be merged or cherry-picked into `main` when stable.

***

### Examples, tests, and documentation

- **Examples**:
  - Multiple example scripts are included with AtomTwin and serve as primary demonstrations of functionality.
- **Ground truth in `test/examples_src`**:
  - These files represent the expected outputs used by both example scripts and documentation.
  - They are generated via the documentation tooling and should be updated only when behavior intentionally changes.
- **Tests**:
  - AtomTwin’s tests include checks against these ground truth files to keep examples and documentation consistent with actual behavior.

When modifying public behavior:

- Update the relevant example(s).
- Regenerate and verify the ground truth files, if applicable.
- Ensure the test suite passes.

***

### Building the documentation

The documentation is maintained in a dedicated Julia environment under `docs/`.

To build the docs locally:

```bash
julia --project=docs docs/make.jl
```

This compiles the documentation and writes the generated HTML into the docs build directory `docs/build`.

To publish the updated documentation:

1. Ensure `docs/make.jl` has completed successfully.
2. Check out the `pages` branch of AtomTwin.
3. Inspect and copy the generated build files (e.g. from `docs/build/`) into the root of the `pages` branch.
4. Commit and push the changes on `pages` so that the hosted documentation is updated.

***

### Contribution workflow

1. Prefer opening an issue or draft MR for non-trivial changes, especially engine (Dynamiq) work.
2. Create a feature branch from the main development branch (typically `devel`).
3. Make focused commits with clear messages (separating AtomTwin and Dynamiq concerns when possible).
4. Confirm:
   - `Pkg.test("AtomTwin")` passes.
   - Examples and documentation still build and remain consistent with the ground truth.
5. Submit a merge request with:
   - A concise summary of the change.
   - Any benchmark results (for Dynamiq changes).
   - Notes on updated examples or documentation.

Because maintainer time is limited, small, well-scoped contributions with good tests and clear rationales are especially appreciated.

### Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report security issues by emailing the lead maintainer directly:

**Shannon Whitlock** — whitlock@unistra.fr

Include in your report:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a minimal proof-of-concept.
- Any suggested fix, if you have one.
