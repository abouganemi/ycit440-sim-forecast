# EDA presentation decks

Source files of the EDA milestone presentation (Team 4, presented 2026-10-15). Each folder is
the content of one Claude Slides artifact: `deck.json` is the index (title, slide order,
sections, fonts) and `slides/<id>.html` holds one slide with its speaker notes in `<aside>`.

| Folder | Artifact | Contents |
| --- | --- | --- |
| [`v1/`](v1/) | [v1_SIM EDA Presentation](https://claude.ai/artifact/P7phRZndFE7EHBMsXo6RFU) | **The deck that is presented.** The original EDA, matching [`reports/eda_report.md`](../../reports/eda_report.md): the backup slide B1 and slide 9 show the 2024 baseline preview (plain 28-day mean 8.03 against 8.16 for same-weekday 13 or 26 weeks). |
| [`v2/`](v2/) | [SIM EDA Presentation](https://claude.ai/artifact/6vF137Bz7kaxiBjermMT4B) | The same deck with B1 and slide 9 updated to the 2021–2024 rolling-origin baselines (13 weeks 8.48, plain 28-day 8.49, a tie), matching [`reports/eda_report_updated.md`](../../reports/eda_report_updated.md). |

Both artifacts are private; they open only for people they are shared with.

**Charts.** Three slides (`level`, `shocks`, `shared`) show figures uploaded to each artifact;
`<img src="/_blob/<id>">` resolves only on that artifact, so the ids differ between `v1/` and
`v2/`. The same figures are in `reports/figures/`: `07_group_composition.png`,
`07_anomalies.png` and `07_median_polish.png`.

**Editing.** The artifacts are the live copies; these files are a record of them. After changing a
deck on the web, copy the changed slide back here. To republish a slide from here, publish the
file to the artifact's URL at its path under `project/` (for example
`project/slides/cover.html`).
