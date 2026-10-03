# Official metadata review

The framing-stage review of the portal's official metadata, moved here from the capstone folder
(`ml-capstone-project/2_understanding_and_framing_the_capstone_project/`) so it can be maintained
alongside the data it describes. Files are copied verbatim; each one names its source file and
line range at the top.

| File | Origin | What it covers |
|---|---|---|
| [provenance_and_sources.md](provenance_and_sources.md) | `business_framing_research/05_data_investigation.md` §1 and Sources | Publisher, licence, update frequency, RAO purpose statement, privacy and revision notes, dictionary definitions, table of the six portal files |
| [dataset_facts_verification.md](dataset_facts_verification.md) | v2 `verification/04_source_verification.md` §2C | Each dataset claim in the v2 documents checked against the portal metadata (C1–C14) |
| [publication_lag.md](publication_lag.md) | v2 `verification/03_design_stress_test.md` §2.1 | Publication lag reconstructed from the portal's activity stream; evidence for the D−3 cutoff |
| [scripts/05_fetch_sources_excerpt.py](scripts/05_fetch_sources_excerpt.py) | `05_data_investigation.py`, header to `fetch()` | Download URLs and the browser User-Agent workaround (the portal answers `403 RBAC: access denied` otherwise) |
| [scripts/03_timing_publication_lag.py](scripts/03_timing_publication_lag.py) | v2 `verification/03_timing_publication_lag.py` | Lag reconstruction script |

The two scripts are archival. They are excluded from ruff and basedpyright
(`[tool.ruff] extend-exclude`, `[tool.basedpyright] exclude`), and `03_timing_publication_lag.py`
does not run as-is: it reads CKAN activity pages and a CSV from a session scratch directory that
no longer exists.

## Official documentation snapshot

The framing-stage review quoted the CKAN `package_show` metadata but never saved it, although the
course guide's step 1 requires saving the official documentation used. The saved copy now lives in
[`docs/source/`](../source/), written by:

```bash
uv run python -m ycit440_sim_forecast.metadata snapshot
```

It holds the full package metadata for the interventions and fire-station datasets, the official
type concordance table, a rendered `data_dictionary.md`, and `snapshot.json` with the retrieval
time and source URLs. The files are saved byte-for-byte, so pre-commit's whitespace hooks skip
`docs/source/`. Re-run the command whenever the raw files in `data/raw/` are re-downloaded.
