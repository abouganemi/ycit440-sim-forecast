# Dataset facts checked against the portal metadata (v2 verification, section 2C)

> Copied verbatim from `ml-capstone-project/2_understanding_and_framing_the_capstone_project/team_contributions/team_accepted_proposed_business_framing/v2/verification/04_source_verification.md`, lines 192–212, on 2026-09-28. Verifier 4 checked each dataset claim in the v2 documents against the CKAN metadata; the `pkg.json` it cites was not kept.

### 2C. Dataset reference and dataset facts (DEFINITIVE §5; DIG §1 and §6)

**Source:** the CKAN API at `https://donnees.montreal.ca/api/3/action/package_show?id=interventions-service-securite-incendie-montreal` (saved as `pkg.json`, metadata_modified 2026-09-26). The landing page is titled "Interventions des pompiers de Montréal - Site web des données ouvertes de la Ville de Montréal".

| # | Claim (where) | Verdict | Primary-source text (verbatim) |
|---|---|---|---|
| C1 | Title "Interventions des pompiers de Montréal"; publisher Ville de Montréal / SIM (DIG; DEFINITIVE ref) | ✓ | `title`: "Interventions des pompiers de Montréal"; `organization`: "Ville de Montréal"; `author`: "Service de sécurité incendie de Montréal" |
| C2 | Source system RAO, "a central subsystem of the intervention-management system that supports real-time intervention management, vehicle dispatch and operational monitoring" (DIG §1, final section) | ✓ | "Ces données sont tirées du système de répartition assisté par Poste de travail (RAO), un sous-système central du système de gestion des interventions qui permet la gestion en temps réel, la répartition des véhicules et du suivi opérationnel des interventions." Also `methodologie`: "Extraction à partir du RAO des données opérationnelles concernant les interventions effectuées par le SIM." Note: the portal expands RAO as "répartition assisté par Poste de travail", whereas SIM's report says "répartition assistée par ordinateur (RAO)" / "Computer-Aided Dispatch (CAD)". The team's "computer-assisted dispatch system" matches SIM's own usage. |
| C3 | Official coverage from January 1, 2005 (DIG) | ✓ | `temporal`: "2005-01-01/"; notes: "…depuis 2005" |
| C4 | Update frequency daily (DIG) | ✓ | `update_frequency`: "daily" |
| C5 | Licence CC BY 4.0 (DIG) | ✓ | `license_title`: "Creative Commons Attribution 4.0 International" |
| C6 | File names "Interventions du SIM – 2020 à 2024" and "– courant (2 dernières années)" (DIG) | ✓ | The resource names match exactly. |
| C7 | Coordinates "displaced to the nearest intersection" (DIG, privacy section) | ✗ (minor) | "La position géographique de l'événement a été localisée à une des intersections du segment de rue où s'est passée l'intervention." The portal says *one of* the segment's intersections, not the nearest. It also states more broadly: "Les données ont été obfusquées et modifiées pour garantir le respect à la vie privée". |
| C8 | Incident type may be modified when information is added (DIG) | ✓ | "le type d'un événement peut être modifié si des éléments s'ajoutent au rapport initial" |
| C9 | INCIDENT_NBR identifies an intervention within a calendar year (DIG) | ✓ | "Identification de l'événement par année." |
| C10 | CASERNE and DIVISION are "responsible for the territory" (DIG) | ✓ | CASERNE: "Numéro de la caserne responsable du territoire où est survenu l'événement"; DIVISION: "Numéro de la division du SIM responsable du territoire où est survenu l'événement" |
| C11 | DESCRIPTION_GROUPE has six documented categories (DIG) | ✓ | "Regroupement des types d'interventions en 6 catégories : Incendies de bâtiments, Autres incendies, Sans incendie, Alarmes-incendie, Premiers répondants, Fausses alertes/annulations." |
| C12 | NOMBRE_UNITES = vehicles deployed; not firefighters (DIG) | ✓ | "Nombre de véhicules déployés pour répondre à l'événement. Une unité peut comprendre de 3 à 5 pompiers." The 3–5 firefighters per unit figure is useful and could be quoted. |
| C13 | "The data … published" contain only type and location (DIG, "not measured" section) | ✓ (not cited) | "Les données diffusées sont limitées aux types d'intervention et leur localisation." This is the strongest official support for the volume ≠ workload boundary, and it is not quoted. |
| C14 | Dataset reference entry: "Ville de Montréal. (2026). Interventions des pompiers de Montréal. Montréal Open Data Portal. URL" | ⚠ | Author, year, title and URL are correct. Missing: the "[Data set]" descriptor and a retrieval date (the dataset changes daily). The portal's own name is "Site web des données ouvertes de la Ville de Montréal"; "Montréal Open Data Portal" is an acceptable translation. The entry is **not cited anywhere in the text**. |
