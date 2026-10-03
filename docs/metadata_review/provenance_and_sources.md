# Provenance, version and sources (framing-stage metadata review)

> Copied verbatim from `ml-capstone-project/2_understanding_and_framing_the_capstone_project/business_framing_research/05_data_investigation.md`, lines 35–53, on 2026-09-28. The Sources section is lines 333–347. Retrieval date in the original: 2026-09-26.

### 1. Provenance and version

- [FACT] **Publisher and licence.** "Interventions des pompiers de Montréal", Ville de Montréal open-data portal, licence **CC BY 4.0**, `update_frequency: daily`, temporal coverage "2005-01-01/". Source: CKAN API `package_show?id=interventions-service-securite-incendie-montreal`, retrieved 2026-09-26 (metadata_modified 2026-09-26T03:04).
- [FACT] **What the data are for.** Quote: « Ces données sont tirées du système de répartition assisté par Poste de travail (RAO) … qui permet la gestion en temps réel, la répartition des véhicules et du suivi opérationnel des interventions. Ces données sont collectées pour produire des rapports requis par le ministère de la Sécurité publique… » (gloss: the data are extracted from the computer-aided dispatch system and collected for reports the Quebec Ministry of Public Security requires, not built for workload analysis).
- [FACT] **Privacy and revision notes.** « La position géographique de l'événement a été localisée à une des intersections du segment de rue » (gloss: each location is moved to an intersection on the street segment). « le type d'un événement peut être modifié si des éléments s'ajoutent au rapport initial » (gloss: an event's type can change if details are added to the initial report).
- [FACT] **Units field.** Official dictionary: `NOMBRE_UNITES` = « Nombre de véhicules déployés pour répondre à l'événement. Une unité peut comprendre de 3 à 5 pompiers. » (gloss: number of vehicles deployed; one unit can hold 3 to 5 firefighters.)

| File (portal resource) | Local name | Last modified (portal) | Rows | Coverage | Columns | Time of day? |
|---|---|---|---|---|---|---|
| Interventions du SIM – 2005 à 2014 | y2005_2014.csv | 2020-03-31 | 908,486 | 2005-01-01 00:03 → 2014-12-31 23:58 | 11 (no MTM8) | yes |
| Interventions du SIM – 2015 à 2022 | y2015_2022.csv | 2024-07-03 | 894,970 | 2015-01-01 → 2022-12-31 | 13 | **no, all rows `YYYY/MM/DD`** |
| Interventions du SIM – 2020 à 2024 | y2020_2024.csv | 2026-01-01 | 539,203 | 2020-01-01 00:01 → **2024-12-30** 23:56 | 13 | yes (7 date-only rows) |
| courant (2 dernières années) | current.csv | 2026-09-26 03:02 | 215,210 | 2025-01-01 00:02 → **2026-09-24 23:59** | 13 | yes (1 date-only row) |
| Tableau concordance type–description | concordance.csv | 2021-09-22 (file dated 2016-11-22) | 170 types | — | 2 | — |
| Casernes de pompiers (package `casernes-pompiers`) | casernes.csv | 2026-09-24 | 68 stations (67 active, 1 closed) | temporal "2019-05-01/", irregular updates | 11 | — |

- [FINDING] **Publication lag.** The latest record is 2026-09-24 23:59:29 and today is 2026-09-26, a lag of **2 days**. The teammate's variable guide says coverage ends Sept 17; that was their download date, and the file keeps growing.
- [FINDING] **De-duplicated series.** Taking 2005–2014, then 2015–2019 from the 2015–2022 file, then 2020–2024, then current gives **2,261,612 incidents**. The only missing calendar day in 2005-01-01 → 2026-09-24 is **2024-12-31**. It is not in the 2020–2024 file (which ends on the 30th) or in the current file (which starts 2025-01-01).

## Sources

- [FACT] Ville de Montréal, "Interventions des pompiers de Montréal", CKAN metadata and dictionary: https://donnees.montreal.ca/api/3/action/package_show?id=interventions-service-securite-incendie-montreal (retrieved 2026-09-26). Dataset page: https://donnees.montreal.ca/dataset/interventions-service-securite-incendie-montreal
- Files (retrieved 2026-09-26), all under https://donnees.montreal.ca/dataset/2fc8a2b9-1556-410e-a118-c46e97e9f19e/resource/:
  - `71e86320-…/download/donneesouvertes-interventions-sim.csv` (current)
  - `4a46d93f-…/download/donneesouvertes-interventions-sim2020.csv` (2020–2024)
  - `005e4eb6-…/download/donneesouvertes-interventions-sim_2015_2022.csv` (2015–2022)
  - `0a778ac0-…/download/donneesouvertes-interventions-sim-2005-2014.csv` (2005–2014)
  - `4f236894-…/download/type-interventions-descriptions20161122.csv` (concordance)

  Full URLs are in `05_data_investigation.py`.
- [FACT] Ville de Montréal, "Casernes de pompiers sur l'île de Montréal": https://donnees.montreal.ca/api/3/action/package_show?id=casernes-pompiers and `casernes.csv` (retrieved 2026-09-26).
- [FACT] Hydro-Québec, "April 2023 ice storm: Hydro-Québec takes action to increase system resilience" (press release): https://news.hydroquebec.com/news/press-releases/all-quebec/april-2023-ice-storm-hydro-quebec-takes-action-to-increase-system-resilience.html. Via web-search summary: the storm hit Montréal on April 5, 2023, affecting over 1.3 million customers. Page not opened in full.
- [FACT, news] CBC News, "Southern Quebec still struggles with remnants of tropical storm Debby": https://www.cbc.ca/news/canada/montreal/rain-tropical-storm-debby-southern-quebec-1.7291210. Via web-search summary: Montréal received a record of about 157 mm on Aug 9, 2024, and firefighters answered hundreds of calls. Page not opened in full.
- Course "Dataset Investigation Guide" and teammate "Firefighter Dataset Variable Guide" (internal).
