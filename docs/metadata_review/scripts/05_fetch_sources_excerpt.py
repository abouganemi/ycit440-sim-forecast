# Excerpt (lines 1-75) of ml-capstone-project/2_understanding_and_framing_the_capstone_project/
# business_framing_research/05_data_investigation.py, copied verbatim on 2026-09-28.
# Archival: superseded by `python -m ycit440_sim_forecast.metadata snapshot`.

"""SIM interventions open data: hands-on investigation for Project B workload framing.

Reproduce:
    uv run --with pandas --with numpy python 05_data_investigation.py [DATA_DIR] > 05_data_investigation_output.txt

DATA_DIR defaults to $SIM_DATA_DIR or ./sim_data. Missing files are downloaded from
donnees.montreal.ca (the portal returns 403 "RBAC: access denied" to non-browser
user agents, so a browser User-Agent header is sent).

Sections follow the course Dataset Investigation Guide and the numbered questions
in the team brief (1 provenance ... 9 signal check). Every number quoted in
05_data_investigation.md is printed by this script.
"""

import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SIM_DATA_DIR", "sim_data")
TODAY = pd.Timestamp("2026-09-26")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

RES = "https://donnees.montreal.ca/dataset/2fc8a2b9-1556-410e-a118-c46e97e9f19e/resource/"
SOURCES = {
    "current.csv": RES + "71e86320-e35c-4b4c-878a-e52124294355/download/donneesouvertes-interventions-sim.csv",
    "y2020_2024.csv": RES + "4a46d93f-9fd9-4cce-8952-424918edeafe/download/donneesouvertes-interventions-sim2020.csv",
    "y2015_2022.csv": RES + "005e4eb6-0377-45bf-a911-7077fd3b5ed0/download/donneesouvertes-interventions-sim_2015_2022.csv",
    "y2005_2014.csv": RES + "0a778ac0-9b5a-42cb-8557-167f7f9b8feb/download/donneesouvertes-interventions-sim-2005-2014.csv",
    "concordance.csv": RES + "4f236894-3fdc-4b04-8d11-80f451ffd70d/download/type-interventions-descriptions20161122.csv",
    "casernes.csv": "https://donnees.montreal.ca/dataset/c69e78c6-e454-4bd9-9778-e4b0eaf8105b/resource/"
    "5b9c0e1d-3f75-4e98-b53d-6e979c18cc98/download/casernes.csv",
    "pkg_interventions.json": "https://donnees.montreal.ca/api/3/action/package_show?id=interventions-service-securite-incendie-montreal",
    "pkg_casernes.json": "https://donnees.montreal.ca/api/3/action/package_show?id=casernes-pompiers",
}
INCIDENT_FILES = ["y2005_2014", "y2015_2022", "y2020_2024", "current"]

# 2005-2014 uses long French group labels; later files use short codes.
# Mapping is by label meaning, checked against shared INCIDENT_TYPE_DESC values in section 3.
GROUP_HARMONIZE = {
    "Premier répondant": "1-REPOND",
    "Sans incendie": "SANS FEU",
    "Alarmes-incendies": "Alarmes-incendies",
    "Autres incendies": "AUTREFEU",
    "Incendies de bâtiments": "INCENDIE",
    "Fausses alertes/annulations": "FAU-ALER",
    "": "(blank)",
}

# ASSUMPTION: SIM's real shift boundaries are unknown; 07-17 / 17-07 is an illustrative block.
DAY_START, NIGHT_START = 7, 17
EVAL_START, EVAL_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def fetch(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        os.makedirs(DATA_DIR, exist_ok=True)
        req = urllib.request.Request(SOURCES[name], headers={"User-Agent": UA})
        with urllib.request.urlopen(req) as r, open(path, "wb") as out:
            out.write(r.read())
    return path
