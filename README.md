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

## Architecture-aware target layouts

`ArchitecturePlanner` analyses representative source snippets before any
translation occurs and produces a canonical destination layout for every legacy
artefact. The LLM-backed proposal (cached for repeat runs) captures destination
paths, Java package names, and optional notes aligned with the selected target
language/framework. The planner falls back to deterministic heuristics so Java
migrations always land under `src/main/java` with sanitised package hierarchies
even when the LLM response is noisy.

The resulting architecture map is stored alongside each migration, exposed via
the API, and visible in the dashboard so you can audit how source files were
reorganised without opening the generated archive.

## Compatibility research for legacy imports

`CompatibilitySearchAgent` inspects legacy manifests and import statements,
queries DuckDuckGo via LangChain when available, and asks the LLM for
target-language friendly alternatives. The resulting guidance lists suggested
libraries/APIs, recommended actions, and confidence notes tailored to the
requested destination stack. Each migration records the compatibility payload
alongside the architecture map so you can review web-sourced hints directly
from the dashboard or API before adopting third-party packages in the target
project.

## Scaffolding blueprint & manual hand-off stubs

`ScaffoldingBlueprintAgent` analyses representative legacy files, the generated
architecture mapping, and known dependencies to identify modules that cannot be
fully migrated automatically. For every detected gap it proposes a contract
(class name, destination path, responsibilities) and the orchestrator writes
TODO stubs inside the generated project tree. These stubs include inline
guidance so target developers can finish the migration while preserving the
expected behaviour. Blueprint metadata and stub creation logs surface via the
CLI output, REST responses, and authenticated dashboard.

## Deterministic dependency extraction fallbacks

While the default behaviour relies on LLM prompts, `DependencyAnalysisAgent`
now parses common manifest formats (e.g. `requirements.txt`, `pyproject.toml`,
`package.json`) locally whenever the model omits entries. This guarantees
critical packages such as `transformers` or `torch` appear in the dependency
snapshot before compatibility research or blueprint planning occurs.

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
NVIDIA H100-class instance.  The receipt also factors in the running
infrastructure bill (defaulting to **$10.58** for the primary resource) and
applies a configurable markup (45 % by default) so you can quote profitable
migration prices.  The summary now exposes:

* `token_generation_cost` – projected GPU spend derived from prompt/completion
  tokens and the configured hourly rate.
* `resource_cost` – the baseline infrastructure spend you specify (defaults to
  $10.58).
* `subtotal_cost` – the sum of resource cost and GPU spend before markup.
* `markup_rate` / `markup_percentage` – the gain applied to the subtotal.
* `suggested_price` – recommended billable amount after markup.
* `projected_profit` – the expected profit given the configured margin.

Two environment variables let you tailor the defaults without touching the
codebase:

```bash
# Override the default $10.58 infrastructure spend.
export CHRISTOPHE_RESOURCE_COST=14.25

# Provide a custom markup as a decimal or percentage string.
export CHRISTOPHE_COST_MARKUP=0.6     # 60 %
# …or…
export CHRISTOPHE_COST_MARKUP=60%

# Optional contextual hints shown in the receipt.
export CHRISTOPHE_RESOURCE_TIME_LEFT="5h 43m left at current spend rate"
export CHRISTOPHE_RESOURCE_CONTEXT="Includes tokens generation fee"
```

If `CHRISTOPHE_COST_MARKUP` is not set the application falls back to
`CHRISTOPHE_MARKUP_RATE`; both default to a 45 % gain.  Clearing
`CHRISTOPHE_RESOURCE_COST` or setting it to `0` removes the baseline spend from
future receipts.

## Suggested machine capabilities

The bundled configuration keeps memory usage reasonable by default—all LLM
profiles now target the 7B-parameter *Mistral Instruct* family so a single
process can run comfortably on workstation-class hardware.  The platform can
execute entirely on CPU-only machines, but the combined translation, dependency
analysis, and asynchronous background workers benefit from additional
resources:

* **Memory:** 16 GB of RAM is the practical minimum for the default 7B models;
  allocate 32 GB or more to absorb concurrent migrations.  Loading
  significantly larger checkpoints (e.g. Mixtral 8x22B) requires 300 GB+ of RAM
  or sharded GPU VRAM—if you override the defaults (see below) ensure the host
  meets the vendor’s published requirements.
* **CPU:** Eight modern vCPUs keep chunking, archive extraction, and dependency
  downloads responsive. Heavy workloads with concurrent migrations benefit from
  16+ vCPUs.
* **GPU (optional):** A CUDA-capable GPU with 24 GB of VRAM or more (e.g.,
  NVIDIA H100-class hardware) shortens inference time considerably. If a GPU is
  present but CUDA drivers/toolkit are missing, the application logs guidance on
  installing `nvcc` or updating drivers at startup.
* **Storage:** Reserve at least 20 GB of free disk space for temporary ZIP
  uploads, generated output archives, cached models, and dependency downloads.

When GPU acceleration is unavailable, the service falls back to CPU inference,
which increases latency but does not block migrations.

### Customising model profiles

You can override any profile via environment variables without modifying the
source code.  For each profile (`classify`, `analyze`, `translate`, `adapt`,
`scaffold`, `dependency`) the application reads:

* `MIGRATION_PROFILE_<NAME>_ID` – Hugging Face model identifier to load.
* `MIGRATION_PROFILE_<NAME>_MAX_TOKENS` – optional integer overriding the
  maximum generation tokens for that profile.
* `MIGRATION_PROFILE_<NAME>_TEMP` – optional float to adjust sampling
  temperature.

For example, to experiment with a larger translation model only when running on
an appropriately sized machine:

```bash
export MIGRATION_PROFILE_TRANSLATE_ID=mistralai/Mixtral-8x22B-Instruct-v0.1
export MIGRATION_PROFILE_TRANSLATE_MAX_TOKENS=4096
```

Remember to scale memory, GPU, and disk resources accordingly before switching
to larger checkpoints.

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
