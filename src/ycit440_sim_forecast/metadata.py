"""Snapshot the official portal documentation of the datasets used.

The Montréal open-data portal (CKAN) publishes each dataset's metadata,
methodology and data dictionary. This command saves that documentation, plus
the official type/description concordance table, into a committed folder so the
project keeps a record of the source it relied on. Usage::

    uv run python -m ycit440_sim_forecast.metadata snapshot [--out docs/source]
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import typer

API_URL = "https://donnees.montreal.ca/api/3/action/package_show?id="
DATASET_URL = "https://donnees.montreal.ca/dataset/"
PACKAGES: tuple[str, ...] = (
    "interventions-service-securite-incendie-montreal",
    "casernes-pompiers",
)
CONCORDANCE_PREFIX = "Tableau concordance"
DEFAULT_OUT = Path("docs/source")
DICTIONARY_NAME = "data_dictionary.md"
SNAPSHOT_NAME = "snapshot.json"
NOT_PROVIDED = "_not provided_"

# The portal answers "403 RBAC: access denied" to non-browser user agents.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"

_HEADING = re.compile(r"^(#{1,6})(?=\s|$)", flags=re.MULTILINE)

app = typer.Typer(help=__doc__, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Snapshot official dataset documentation."""


def fetch(url: str, timeout: float = 60) -> bytes:
    """Download a URL with a browser user agent.

    Args:
        url: HTTPS URL to download.
        timeout: Socket timeout in seconds.

    Returns:
        The response body.

    Raises:
        ValueError: If the URL is not HTTPS.
    """
    if urllib.parse.urlparse(url).scheme != "https":
        msg = f"refusing non-https URL: {url}"
        raise ValueError(msg)
    # Scheme is checked above, so file:// and other schemes cannot reach urlopen.
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch_package(name: str) -> dict[str, Any]:
    """Fetch a package's metadata from the CKAN API.

    Args:
        name: CKAN package name.

    Returns:
        The ``result`` object of ``package_show``.

    Raises:
        ValueError: If the API does not report success or the payload is malformed.
    """
    payload = json.loads(fetch(API_URL + name))
    if not isinstance(payload, dict) or payload.get("success") is not True:
        msg = f"package_show did not succeed for {name!r}"
        raise ValueError(msg)
    result = payload.get("result")
    if not isinstance(result, dict):
        msg = f"package_show returned no result object for {name!r}"
        raise ValueError(msg)
    return result


def format_timestamp(moment: datetime) -> str:
    """Format a datetime as UTC ISO 8601 with seconds precision and ``Z``.

    Args:
        moment: Timezone-aware datetime.

    Returns:
        E.g. ``2026-09-28T14:03:00Z``.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_markdown(text: object, demote: int = 2) -> str:
    """Clean published markdown without changing its wording.

    Normalises CRLF and CR line endings to LF, strips trailing whitespace on each line
    and demotes headings so they nest under the section that embeds them.

    Args:
        text: Markdown as published (``None`` or empty renders as not provided).
        demote: Number of levels to add to each heading (capped at level 6).

    Returns:
        The cleaned markdown, without leading or trailing blank lines.
    """
    if not isinstance(text, str) or not text.strip():
        return NOT_PROVIDED
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = "\n".join(line.rstrip() for line in lines).strip("\n")
    return _HEADING.sub(lambda m: "#" * min(len(m.group(1)) + demote, 6), cleaned)


def _cell(value: object) -> str:
    """Render a value for a single markdown table cell."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return NOT_PROVIDED
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def _link(label: object, url: object) -> str:
    """Render a markdown link, falling back to whichever part is present."""
    if isinstance(url, str) and url.strip():
        text = _cell(label) if label else url.strip()
        return f"[{text}]({url.strip()})"
    return _cell(label)


def _size(value: object) -> str:
    """Render a byte count with thousands separators."""
    if value is None or isinstance(value, bool):
        return "n/a"
    try:
        return f"{int(str(value)):,}"
    except ValueError:
        return _cell(value)


def _render_package(name: str, package: Mapping[str, Any]) -> list[str]:
    """Render one package section of the data dictionary."""
    organisation = package.get("organization") or {}
    rows = [
        ("Publisher", _cell(package.get("author"))),
        ("Organisation", _cell(organisation.get("title"))),
        ("Licence", _link(package.get("license_title"), package.get("license_url"))),
        ("Update frequency", _cell(package.get("update_frequency"))),
        ("Temporal coverage", _cell(package.get("temporal"))),
        ("Metadata last modified", _cell(package.get("metadata_modified"))),
        ("Dataset page", DATASET_URL + name),
        ("API", API_URL + name),
    ]
    lines = [
        f"## {_cell(package.get('title') or name)}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        *(f"| {key} | {value} |" for key, value in rows),
        "",
        "### Description",
        "",
        normalise_markdown(package.get("notes")),
        "",
        "### Methodology and data dictionary",
        "",
        normalise_markdown(package.get("methodologie")),
        "",
        "### Resources",
        "",
    ]
    resources = package.get("resources") or []
    if resources:
        lines += [
            "| Name | Format | Last modified | Size (bytes) | Link |",
            "| --- | --- | --- | --- | --- |",
            *(
                f"| {_cell(res.get('name'))} | {_cell(res.get('format'))} "
                f"| {_cell(res.get('last_modified'))} | {_size(res.get('size'))} "
                f"| {_link('link', res.get('url'))} |"
                for res in resources
            ),
        ]
    else:
        lines.append(NOT_PROVIDED)
    lines.append("")
    return lines


def render_dictionary(
    packages: Mapping[str, dict[str, Any]], retrieved_at: datetime
) -> str:
    """Render the human-readable documentation of all packages.

    Args:
        packages: Package name to ``package_show`` result.
        retrieved_at: When the metadata was fetched.

    Returns:
        Markdown ending with exactly one newline and no trailing whitespace.
    """
    lines = [
        "# Official dataset documentation (portal snapshot)",
        "",
        "Generated by `python -m ycit440_sim_forecast.metadata snapshot`; "
        "do not edit by hand.",
        "",
        f"Retrieved at: {format_timestamp(retrieved_at)}",
        "",
    ]
    for name, package in packages.items():
        lines += _render_package(name, package)
    text = "\n".join(line.rstrip() for line in "\n".join(lines).split("\n"))
    return text.rstrip("\n") + "\n"


def find_concordance(package: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the concordance resource of a package, if any.

    Args:
        package: ``package_show`` result.

    Returns:
        The first resource whose name starts with ``CONCORDANCE_PREFIX``.
    """
    return next(
        (
            res
            for res in package.get("resources") or []
            if str(res.get("name") or "").startswith(CONCORDANCE_PREFIX)
        ),
        None,
    )


def _dump_json(value: object) -> str:
    """Serialise JSON the same way for every snapshot file."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_snapshot(out: Path, retrieved_at: datetime | None = None) -> list[Path]:
    """Fetch the official documentation and write it under ``out``.

    Everything is fetched before anything is written, so a network or API
    failure never leaves a half-updated snapshot.

    Args:
        out: Destination directory (created if needed).
        retrieved_at: Retrieval time; defaults to now (UTC, whole seconds).

    Returns:
        Paths written, in write order.

    Raises:
        ValueError: If the interventions package has no concordance resource.
    """
    moment = retrieved_at or datetime.now(UTC).replace(microsecond=0)
    packages = {name: fetch_package(name) for name in PACKAGES}

    interventions = PACKAGES[0]
    concordance = find_concordance(packages[interventions])
    if concordance is None or not concordance.get("url"):
        msg = f"no {CONCORDANCE_PREFIX!r} resource with a URL in {interventions!r}"
        raise ValueError(msg)
    concordance_url = str(concordance["url"])
    concordance_name = PurePosixPath(urllib.parse.urlparse(concordance_url).path).name
    if not concordance_name:
        msg = f"cannot derive a file name from {concordance_url!r}"
        raise ValueError(msg)
    concordance_bytes = fetch(concordance_url)

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sources: dict[str, str] = {}
    for name, package in packages.items():
        path = out / f"{name}.json"
        path.write_text(_dump_json(package), encoding="utf-8")
        written.append(path)
        sources[path.name] = API_URL + name

    path = out / concordance_name
    path.write_bytes(concordance_bytes)
    written.append(path)
    sources[path.name] = concordance_url

    path = out / DICTIONARY_NAME
    path.write_text(render_dictionary(packages, moment), encoding="utf-8")
    written.append(path)

    path = out / SNAPSHOT_NAME
    snapshot = {"retrieved_at": format_timestamp(moment), "sources": sources}
    path.write_text(_dump_json(snapshot), encoding="utf-8")
    written.append(path)
    return written


@app.command()
def snapshot(
    out: Annotated[Path, typer.Option(help="Destination directory.")] = DEFAULT_OUT,
) -> None:
    """Save the portal metadata, concordance table and data dictionary to OUT."""
    try:
        written = write_snapshot(out)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        # json.JSONDecodeError is a ValueError subclass; HTTPError a URLError one.
        typer.echo(f"snapshot failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    for path in written:
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
