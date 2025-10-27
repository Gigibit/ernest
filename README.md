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
python christophe.py your_project.zip --target-framework "Spring Boot" \
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
