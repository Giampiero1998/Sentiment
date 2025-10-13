<!-- Auto-generated: initial Copilot instructions for this workspace. Edit to add project-specific details. -->
# Copilot instructions (project: Progetto 10)

Purpose: give AI coding agents the minimal, actionable knowledge to be productive in this repository. If you add or change major components, update this file.

- Short summary
  - This repository currently appears empty. If source code exists elsewhere, clone or copy it into this workspace before continuing.

- Where to look first (when code is present)
  - `README.md` — overall project goals and setup. If present, read fully.
  - `package.json`, `pyproject.toml`, `requirements.txt`, `Pipfile`, `setup.py`, `Cargo.toml`, `go.mod` — language and dependency managers.
  - `Dockerfile`, `docker-compose.yml` — container dev or production setup.
  - `src/`, `app/`, `lib/`, `backend/`, `frontend/` — main source trees; open top-level entrypoints (e.g., `index.js`, `main.py`, `app.tsx`).

- Architecture and high-level checks (when present)
  - Identify the runtime (Node, Python, .NET, Java, Go) from manifests above.
  - Find the app entrypoint (search for `start`, `main`, `__main__`, or `if __name__ == '__main__'`).
  - Map folder boundaries: which folders are services, libraries, infra, and tests.

- Developer workflows to surface
  - Build: prefer `npm install && npm run build` (Node) or `pip install -r requirements.txt` then `python -m build` (Python) — confirm by reading manifests and scripts.
  - Tests: look for `scripts.test` in `package.json`, `pytest.ini`, `tox.ini`, or `tests/` folder.
  - Run: prefer invoking the package's documented start script or `docker-compose up` if docker files exist.

- Project-specific patterns to document (examples — replace when repo files are available)
  - Example: if `src/api/` contains `routes/*.py`, the project uses a REST-first design where handlers return dicts serialized by a central `app.py`.
  - Example: if `frontend/src/components/` uses `*.module.css`, prefer locally-scoped CSS modules and functional React components.

- Integration points and external deps
  - Check `.env`, `config/*.yaml` or `secrets/` for service URLs (databases, auth, 3rd-party APIs). Do not exfiltrate secrets; ask user for credentials.

- How Copilot should help (actionable rules)
  - Before editing, search for existing tests and run them locally. If none exist, create a focused unit test for any new behavior.
  - Prefer small, single-file pull requests that update or add one feature and related tests.
  - When adding dependencies, update the relevant lockfile (`package-lock.json`, `poetry.lock`, etc.) and ensure builds still succeed.
  - Use existing code style: match existing linter config (`.eslintrc`, `pyproject.toml[tool.black]`, etc.). If none, follow common conventions for the language.

- If you need more context
  - Ask the repository owner for the primary language and any missing setup steps.
  - Request access to internal docs or the upstream repository if this is a placeholder workspace.

---
Please review these instructions and paste any project-specific commands or files you'd like included; I'll merge them into this document.
