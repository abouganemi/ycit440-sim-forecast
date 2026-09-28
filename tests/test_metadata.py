import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ycit440_sim_forecast import metadata

runner = CliRunner()

RETRIEVED_AT = datetime(2026, 9, 28, 14, 3, 7, tzinfo=UTC)
CONCORDANCE_URL = (
    "https://donnees.montreal.ca/dataset/x/resource/y/download/"
    "type-interventions-descriptions20161122.csv"
)
# Latin-1 bytes and CRLF: must be saved untouched.
CONCORDANCE_BYTES = "TYPE;DESCRIPTION\r\n1;Incendie de bâtiment \r\n".encode("latin-1")

Fetcher = Callable[..., bytes]


def _interventions() -> dict[str, Any]:
    return {
        "title": "Interventions du SIM",
        "author": "Service de sécurité incendie de Montréal",
        "author_email": "sim@example.org",
        "license_title": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "update_frequency": "weekly",
        "temporal": "2005/..",
        "metadata_modified": "2026-09-01T10:00:00",
        "notes": "Liste des interventions.  \r\nDeuxième ligne",
        "methodologie": (
            "### Confidentialité  \r\nAdresses arrondies.\r\n\r\n"
            "## Dictionnaire\r\n| Champ | Description |  \r\n"
            "| --- | --- |\r\n| INCIDENT_TYPE_DESC | voir fichier joint |\r\n"
        ),
        "organization": {"title": "Ville de Montréal"},
        "resources": [
            {
                "name": "Données 2005-2014",
                "format": "CSV",
                "last_modified": "2026-09-01T10:00:00",
                "size": 1234567,
                "url": "https://example.org/a.csv",
            },
            {
                "name": "Tableau concordance : Type incident - Description",
                "format": "CSV",
                "last_modified": None,
                "size": None,
                "url": CONCORDANCE_URL,
            },
        ],
    }


def _casernes() -> dict[str, Any]:
    return {
        "title": "Casernes de pompiers",
        "author": "",
        "notes": None,
        "resources": [],
    }


@pytest.fixture
def packages() -> dict[str, dict[str, Any]]:
    return {
        "interventions-service-securite-incendie-montreal": _interventions(),
        "casernes-pompiers": _casernes(),
    }


def _fake_fetch(
    packages: dict[str, dict[str, Any]], *, success: bool = True
) -> Fetcher:
    def fake(url: str, timeout: float = 60) -> bytes:
        del timeout
        if url == CONCORDANCE_URL:
            return CONCORDANCE_BYTES
        name = url.removeprefix(metadata.API_URL)
        body = {"success": success, "result": packages[name]}
        return json.dumps(body).encode()

    return fake


@pytest.fixture
def portal(
    packages: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> dict[str, dict[str, Any]]:
    monkeypatch.setattr(metadata, "fetch", _fake_fetch(packages))
    return packages


def test_write_snapshot_files(
    portal: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    written = metadata.write_snapshot(tmp_path, RETRIEVED_AT)

    assert [p.name for p in written] == [
        "interventions-service-securite-incendie-montreal.json",
        "casernes-pompiers.json",
        "type-interventions-descriptions20161122.csv",
        "data_dictionary.md",
        "snapshot.json",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in written)


def test_package_json_format(portal: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    metadata.write_snapshot(tmp_path, RETRIEVED_AT)
    name = "interventions-service-securite-incendie-montreal"
    text = (tmp_path / f"{name}.json").read_text(encoding="utf-8")

    assert text == (
        json.dumps(portal[name], indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    assert "Montréal" in text  # not \u-escaped
    assert json.loads(text) == portal[name]


def test_concordance_bytes_verbatim(
    portal: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    metadata.write_snapshot(tmp_path, RETRIEVED_AT)
    path = tmp_path / "type-interventions-descriptions20161122.csv"
    assert path.read_bytes() == CONCORDANCE_BYTES


def test_snapshot_json(portal: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    metadata.write_snapshot(tmp_path, RETRIEVED_AT)
    snapshot = json.loads((tmp_path / "snapshot.json").read_text())

    assert snapshot == {
        "retrieved_at": "2026-09-28T14:03:07Z",
        "sources": {
            "casernes-pompiers.json": metadata.API_URL + "casernes-pompiers",
            "interventions-service-securite-incendie-montreal.json": (
                metadata.API_URL + "interventions-service-securite-incendie-montreal"
            ),
            "type-interventions-descriptions20161122.csv": CONCORDANCE_URL,
        },
    }


def test_default_retrieved_at_is_whole_seconds(
    portal: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    metadata.write_snapshot(tmp_path)
    stamp = json.loads((tmp_path / "snapshot.json").read_text())["retrieved_at"]
    assert datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").tzinfo is None


def test_missing_concordance(
    packages: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interventions = packages["interventions-service-securite-incendie-montreal"]
    interventions["resources"] = interventions["resources"][:1]
    monkeypatch.setattr(metadata, "fetch", _fake_fetch(packages))

    with pytest.raises(ValueError, match="interventions-service-securite-incendie"):
        metadata.write_snapshot(tmp_path / "out", RETRIEVED_AT)
    assert not (tmp_path / "out").exists()  # nothing half-written


def test_concordance_url_without_file_name(
    packages: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    interventions = packages["interventions-service-securite-incendie-montreal"]
    interventions["resources"][1]["url"] = "https://example.org/"
    monkeypatch.setattr(metadata, "fetch", _fake_fetch(packages))

    with pytest.raises(ValueError, match="file name"):
        metadata.write_snapshot(tmp_path, RETRIEVED_AT)


def test_fetch_package_unsuccessful(
    packages: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(metadata, "fetch", _fake_fetch(packages, success=False))
    with pytest.raises(ValueError, match="did not succeed"):
        metadata.fetch_package("casernes-pompiers")


def test_fetch_package_without_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "fetch", lambda _url: b'{"success": true}')
    with pytest.raises(ValueError, match="no result"):
        metadata.fetch_package("casernes-pompiers")


def test_dictionary_normalised(packages: dict[str, dict[str, Any]]) -> None:
    text = metadata.render_dictionary(packages, RETRIEVED_AT)
    lines = text.split("\n")

    assert lines[0] == "# Official dataset documentation (portal snapshot)"
    assert "python -m ycit440_sim_forecast.metadata snapshot" in text
    assert "2026-09-28T14:03:07Z" in text
    assert "\r" not in text
    assert all(line == line.rstrip() for line in lines)
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert "##### Confidentialité" in lines
    assert "#### Dictionnaire" in lines
    assert "### Confidentialité" not in lines
    assert "| INCIDENT_TYPE_DESC | voir fichier joint |" in lines
    assert "## Interventions du SIM" in lines
    assert "## Casernes de pompiers" in lines


def test_dictionary_tables(packages: dict[str, dict[str, Any]]) -> None:
    text = metadata.render_dictionary(packages, RETRIEVED_AT)

    assert "| Publisher | Service de sécurité incendie de Montréal |" in text
    assert "| Organisation | Ville de Montréal |" in text
    assert (
        "| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |"
        in text
    )
    assert f"| Dataset page | {metadata.DATASET_URL}casernes-pompiers |" in text
    assert f"| API | {metadata.API_URL}casernes-pompiers |" in text
    assert "| 1,234,567 |" in text
    assert "| n/a |" in text
    assert f"[link]({CONCORDANCE_URL})" in text


def test_not_provided(packages: dict[str, dict[str, Any]]) -> None:
    text = metadata.render_dictionary(
        {"casernes-pompiers": packages["casernes-pompiers"]}, RETRIEVED_AT
    )
    section = text.split("## Casernes de pompiers", 1)[1]

    assert "| Publisher | _not provided_ |" in section
    assert "| Organisation | _not provided_ |" in section
    assert "| Licence | _not provided_ |" in section
    assert "### Description\n\n_not provided_\n" in section
    assert "### Methodology and data dictionary\n\n_not provided_\n" in section
    assert "### Resources\n\n_not provided_\n" in section


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "n/a"), (True, "n/a"), (1000, "1,000"), ("2048", "2,048"), ("?", "?")],
)
def test_size(value: object, expected: str) -> None:
    assert metadata._size(value) == expected


def test_cell_escapes_pipes_and_newlines() -> None:
    assert metadata._cell("a|b\nc") == "a\\|b c"


def test_heading_demotion_caps_at_six() -> None:
    assert metadata.normalise_markdown("##### Deep\n#tag") == "###### Deep\n#tag"


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_fetch_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> Any:
        seen["request"] = request
        seen["timeout"] = timeout
        return _FakeResponse(b"payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert metadata.fetch("https://example.org/x", timeout=5) == b"payload"
    request: urllib.request.Request = seen["request"]
    assert request.full_url == "https://example.org/x"
    assert request.get_header("User-agent") == metadata.USER_AGENT
    assert seen["timeout"] == 5


@pytest.mark.parametrize(
    "url", ["http://donnees.montreal.ca/x", "file:///etc/passwd", "ftp://a/b"]
)
def test_fetch_rejects_non_https(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(ValueError, match="non-https"):
        metadata.fetch(url)


def test_cli_happy_path(portal: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    result = runner.invoke(metadata.app, ["snapshot", "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert result.output.count("wrote ") == 5
    assert (tmp_path / "data_dictionary.md").is_file()


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("boom"),
        ValueError("bad"),
        json.JSONDecodeError("bad json", "", 0),
        TimeoutError("slow"),
    ],
)
def test_cli_error_exit(
    error: Exception, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(*_args: object, **_kwargs: object) -> bytes:
        raise error

    monkeypatch.setattr(metadata, "fetch", fail)
    result = runner.invoke(metadata.app, ["snapshot", "--out", str(tmp_path)])

    assert result.exit_code == 1
    assert "snapshot failed" in result.output
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
