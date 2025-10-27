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

The response returns the migration plan, the detected (or user-provided)
stack, token usage, and an estimated hardware receipt.

## Token accounting and cost receipt

All migration runs now track prompt and completion tokens per LLM profile.
The CLI, web UI, and REST API include a breakdown under `token_usage` plus an
`H100 cost estimate` section that approximates GPU time and cost assuming an
NVIDIA H100-class instance.
