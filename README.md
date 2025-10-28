# ernest

## Hugging Face authentication

Place your Hugging Face access token in a local `.env` file to avoid running
`huggingface-cli login`. Supported keys include `HF_TOKEN`,
`HUGGINGFACEHUB_API_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`. Example:

```
HF_TOKEN=hf_xxx
```

The application automatically loads the `.env` file on startup so the token is
available when downloading models.

## Paginated migrations and iterative refinement

Large projects can be migrated in multiple passes to keep LLM prompts within a
safe context window.  Use the new pagination and refinement controls to split
each source artefact into smaller "pages" and optionally polish every page with
extra model calls.

### CLI

```bash
python christophe.py your_project.zip --target-framework "modern service stack" \
  --page-size 5 --refine-passes 1
```

`--page-size` defines how many source chunks are grouped per page.  Setting it
to `0` (default) disables pagination.  `--refine-passes` runs additional
polishing rounds for each page to remove artefacts left by the first
translation.

### Web UI

The Flask interface now exposes *Pagination size* and *Refinement passes*
inputs next to the file uploader, so you can experiment with progressive
refinements directly from the browser.

## Safe mode fallback guardrails

Safe mode is enabled by default and adds an automatic safety harness to the
translation pipeline.  Each chunk is inspected for merge markers, TODOs, or
legacy syntax: suspicious results are reissued with stricter fallback prompts
before touching the destination project.  This significantly reduces the chance
of ending up with conflicted files during large migrations (e.g. legacy COBOL
modules moving to modern service frameworks) without requiring manual tuning.

### CLI

The command line exposes `--safe-mode/--no-safe-mode` so you can toggle the
guardrails explicitly.  Example:

```bash
python christophe.py your_project.zip --target-framework "modern service stack" \
  --no-safe-mode
```

### Web UI

The web form includes a *Safe mode* checkbox (checked by default).  Uncheck it
to force raw translations without fallback retries.

## Authenticated workspaces and project tracking

Launching the Flask server now presents a lightweight authentication screen.
Enter any passphrase to create your personal workspace; subsequent logins with
the same passphrase reopen the same workspace.  After authentication the portal
shows:

* your workspace identifier and API token (displayed in the header);
* an upload form for new migrations;
* a list of active/failed runs, including error messages when something goes
  wrong;
* a catalogue of completed projects with direct download links to the generated
  archives and cost/token breakdowns.

Each migration is recorded asynchronously, so long-running runs remain visible
under *Active uploads* until completion.  Completed migrations move to the
dedicated section and remain downloadable as long as the output archive exists
on disk.

Use the *Log out* link in the header to clear the current session and return to
the passphrase prompt.

## REST API for automated migrations

When the Flask server is running you can trigger migrations programmatically
by calling `POST /api/migrate` with a multipart form payload:

* `archive` (file, required): the ZIP archive of the legacy project.
* `target_framework` (string, required): descriptor for the desired
  destination platform.
* `target_lang` (string, optional): programming language expected in the
  migrated project. Defaults to `java`.
* `src_lang` (string, optional): declare the legacy language to skip the LLM
  stack detection pass.
* `src_framework` (string, optional): descriptor for the legacy framework.
* `safe_mode` (boolean-like string, optional): set to `false`, `0`, `off`, or
  `no` to disable fallback retries; any other value (or omission) keeps safe
  mode enabled.

The response returns the migration plan, the detected (or user-provided)
stack, token usage, and an estimated hardware receipt.

## Serving the Flask application

The CLI doubles as the entry point for the authenticated Flask portal. Install
the dependencies and launch the server with the `--serve` flag:

```bash
pip install -r requirements.txt
python christophe.py --serve --host 0.0.0.0 --port 5000
```

`--host` and `--port` are optional; adjust them to expose the UI on a different
interface or port. When the server starts you can open `http://<host>:<port>` in
the browser, enter a passphrase to create or resume your workspace, and begin
uploading archives from the dashboard. Use `Ctrl+C` in the terminal to stop the
server when you are done.


### Obtaining API tokens

Before calling `/api/migrate`, exchange your passphrase for a bearer token via
`POST /api/auth`:

```bash
curl -X POST http://localhost:5000/api/auth \
  -H 'Content-Type: application/json' \
  -d '{"passphrase": "my secret"}'
```

The response contains a `token` field and the `user_id` of the associated
workspace.  Include the token in subsequent calls via `Authorization: Bearer
<token>` or the `auth_token` query parameter.  Tokens are tied to each
workspace; attempting to access another workspace’s data returns `401`.

### Listing and downloading projects via API

Two additional endpoints expose the migration catalogue:

* `GET /api/projects` returns every tracked migration for the authenticated
  workspace, including status, metadata, and download URLs for completed
  archives.
* `GET /api/projects/<project_id>/download` streams the ZIP archive for a
  completed migration.  Requests with missing/invalid tokens or mismatched
  workspace IDs receive `401`, and attempting to download a project owned by a
  different workspace returns `404`.

## Token accounting and cost receipt

All migration runs now track prompt and completion tokens per LLM profile.
The CLI, web UI, and REST API include a breakdown under `token_usage` plus an
`H100 cost estimate` section that approximates GPU time and cost assuming an
NVIDIA H100-class instance.

## Dependency planning and alternative evaluation

The migration workflow now inspects common dependency manifests (e.g.
`package.json`, `pom.xml`, `requirements.txt`) and summarises the declared
packages before translation begins.  For each dependency the platform asks the
LLM to:

* provide ready-to-use `wget` URLs or build instructions;
* list compatible alternatives native to the destination language or framework;
* outline a translation schedule when no viable replacement exists, complete
  with multi-day milestones when required.

The resolver attempts to download each dependency under the generated project's
`third_party/` directory and records the exit status.  Both the dependency
snapshot and the resolution report are exposed in the CLI output, web UI, and
REST API responses.  All additional prompts reuse the shared pricing counter so
token and cost totals automatically include dependency analysis overhead.
