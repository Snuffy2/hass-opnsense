"""Tests for the aiopnsense dependency update workflow."""

from importlib import util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

WORKFLOW_PATH = Path(".github/workflows/update_aiopnsense.yml")
SCRIPT_PATH = Path(".github/scripts/update_aiopnsense_pins.py")
RELEASE_NOTES_SCRIPT_PATH = Path(".github/scripts/build_aiopnsense_release_notes.py")
CLEANUP_SCRIPT_PATH = Path(".github/scripts/cleanup_aiopnsense_update_branches.py")


class FakeHTTPResponse:
    """Fake HTTP response returned by workflow helper request tests."""

    def __init__(
        self,
        *,
        status: int,
        body: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize a fake HTTP response.

        Args:
            status: HTTP status code.
            body: Response body text.
            headers: Optional response headers.
        """
        self.status = status
        self._body = body.encode()
        self.headers = headers or {}

    def read(self) -> bytes:
        """Return the encoded response body."""
        return self._body


class FakeHTTPSConnection:
    """Fake HTTPS connection that records requests and returns one response."""

    def __init__(self, response: FakeHTTPResponse) -> None:
        """Initialize the connection with a single response.

        Args:
            response: Response returned by ``getresponse``.
        """
        self.response = response
        self.requests: list[dict[str, object]] = []
        self.closed = False

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Record the outgoing request."""
        self.requests.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "headers": headers,
            }
        )

    def getresponse(self) -> FakeHTTPResponse:
        """Return the configured response."""
        return self.response

    def close(self) -> None:
        """Record that the connection was closed."""
        self.closed = True


class FakeHTTPSConnectionFactory:
    """Factory that supplies fake HTTPS connections to patched modules."""

    def __init__(self, responses: list[FakeHTTPResponse]) -> None:
        """Initialize the factory with ordered responses.

        Args:
            responses: Responses returned by successive connections.
        """
        self.responses = responses
        self.connections: list[FakeHTTPSConnection] = []

    def __call__(self, *_: object, **__: object) -> FakeHTTPSConnection:
        """Return a fake connection for the next response."""
        connection = FakeHTTPSConnection(self.responses.pop(0))
        self.connections.append(connection)
        return connection


class FakeCleanupClient:
    """Fake GitHub cleanup client that records mutating calls."""

    def __init__(
        self,
        *,
        open_pulls: list[dict[str, object]],
        closed_pulls: list[dict[str, object]],
        fail_on_close: bool = False,
    ) -> None:
        """Initialize fake pull request state.

        Args:
            open_pulls: Pull requests to return for open PR lookups.
            closed_pulls: Pull requests to return for closed PR lookups.
            fail_on_close: Whether closing a PR should fail the test.
        """
        self.open_pulls = open_pulls
        self.closed_pulls = closed_pulls
        self.fail_on_close = fail_on_close
        self.closed_prs: list[int] = []
        self.deleted_refs: list[str] = []

    def list_pulls(self, *, state: str) -> list[dict[str, object]]:
        """Return fake pull requests by state."""
        return self.open_pulls if state == "open" else self.closed_pulls

    def close_pull(self, pull_number: int) -> None:
        """Record or reject a closed pull request."""
        if self.fail_on_close:
            raise AssertionError(f"Unexpected close for PR {pull_number}")
        self.closed_prs.append(pull_number)

    def delete_ref(self, ref: str) -> None:
        """Record a deleted git ref."""
        self.deleted_refs.append(ref)


def _workflow_pull(
    *,
    number: int,
    ref: str = "chore/update-aiopnsense-manifest",
    label: str = "aiopnsense-auto-update",
    merged_at: str | None = None,
) -> dict[str, object]:
    """Return a fake workflow pull request object.

    Args:
        number: Pull request number.
        ref: Pull request head ref.
        label: Pull request label name.
        merged_at: Optional merge timestamp for closed PRs.

    Returns:
        Fake pull request object shaped like the GitHub REST API response.
    """
    return {
        "number": number,
        "merged_at": merged_at,
        "head": {"ref": ref, "repo": {"full_name": "o/r"}},
        "labels": [{"name": label}],
    }


def _write_pin_files(
    tmp_path: Path,
    *,
    manifest_version: str,
    pyproject_version: str | None = None,
    pyproject_text: str | None = None,
) -> tuple[Path, Path]:
    """Write temporary manifest and pyproject files with aiopnsense pins."""
    manifest_path = tmp_path / "manifest.json"
    pyproject_path = tmp_path / "pyproject.toml"
    manifest_path.write_text(
        json.dumps(
            {
                "requirements": [
                    "xmltodict==0.14.2",
                    f"aiopnsense=={manifest_version}",
                ],
            },
        ),
    )

    if pyproject_text is None:
        pyproject_text = f"""[dependency-groups]
ha = [
    "aiopnsense=={pyproject_version or manifest_version}",
]
"""
    pyproject_path.write_text(pyproject_text)
    return manifest_path, pyproject_path


@pytest.fixture
def updater_script() -> ModuleType:
    """Load the aiopnsense pin updater script as a test module."""
    return _load_script("update_aiopnsense_pins", SCRIPT_PATH)


@pytest.fixture
def release_notes_script() -> ModuleType:
    """Load the aiopnsense release-note builder script as a test module."""
    return _load_script("build_aiopnsense_release_notes", RELEASE_NOTES_SCRIPT_PATH)


@pytest.fixture
def cleanup_script() -> ModuleType:
    """Load the aiopnsense cleanup script as a test module."""
    return _load_script("cleanup_aiopnsense_update_branches", CLEANUP_SCRIPT_PATH)


def _load_script(module_name: str, script_path: Path) -> ModuleType:
    """Load a checked-in workflow helper script as a test module."""
    spec = util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("needle", "reason"),
    [
        ("actions/setup-python@v6", "pins a Python runtime"),
        ("python-version: '3.14'", "uses a tomllib-capable Python"),
        ("Automated update of aiopnsense dependency pins.", "describes generated PRs"),
        ("custom_components/opnsense/manifest.json", "updates the integration manifest"),
        ("pyproject.toml", "updates the local dependency pin"),
        ("pyproject_current", "reports pyproject drift in PR metadata"),
        (str(SCRIPT_PATH), "runs the checked-in updater helper"),
        (str(RELEASE_NOTES_SCRIPT_PATH), "runs the checked-in release-note helper"),
        (str(CLEANUP_SCRIPT_PATH), "runs the checked-in cleanup helper"),
        ("--body-path aiopnsense-update-pr-body.md", "uses a PR body file"),
        ("--delete-merged-branches", "cleans merged workflow-owned branches"),
        ("LATEST_VERSION: ${{ steps.versions.outputs.latest }}", "exports latest pin"),
        ('--latest-version "$LATEST_VERSION"', "avoids shell template injection"),
        ("REPOSITORY: ${{ github.repository }}", "exports repository before shell use"),
        ('--repository "$REPOSITORY"', "avoids inline repository template expansion"),
    ],
)
def test_workflow_contains_expected_update_logic(needle: str, reason: str) -> None:
    """Workflow should include the expected aiopnsense updater logic."""
    del reason

    assert needle in _read_update_workflow_surface()


def test_workflow_avoids_inline_repository_template_expansion() -> None:
    """Workflow should not expand the repository context inside shell scripts."""
    assert '--repository "${{ github.repository }}"' not in WORKFLOW_PATH.read_text()


def _read_update_workflow_surface() -> str:
    """Read workflow and helper surfaces checked by workflow assertions."""
    return (
        f"{WORKFLOW_PATH.read_text()}\n"
        f"{RELEASE_NOTES_SCRIPT_PATH.read_text()}\n"
        f"{CLEANUP_SCRIPT_PATH.read_text()}"
    )


@pytest.mark.parametrize(
    (
        "manifest_version",
        "pyproject_version",
        "latest_version",
        "expected_update_needed",
        "expected_target",
    ),
    [
        ("1.0.8", "1.0.8", "1.0.9", True, "1.0.9"),
        ("1.0.0", "1.0.0", "1.0.1rc1", False, "1.0.0"),
        ("1.0.1rc1", "1.0.1rc1", "1.0.1", True, "1.0.1"),
        ("1.0.10", "1.0.9", "1.0.9", True, "1.0.10"),
    ],
)
def test_updater_script_pin_update_scenarios(
    tmp_path: Path,
    updater_script: ModuleType,
    manifest_version: str,
    pyproject_version: str,
    latest_version: str,
    expected_update_needed: bool,
    expected_target: str,
) -> None:
    """Updater script should handle stable, prerelease, and drift pin scenarios."""
    manifest_path, pyproject_path = _write_pin_files(
        tmp_path,
        manifest_version=manifest_version,
        pyproject_version=pyproject_version,
    )

    result = updater_script.update_pins(
        manifest_path=manifest_path,
        pyproject_path=pyproject_path,
        latest_version=latest_version,
    )

    assert result.current == manifest_version
    assert result.pyproject_current == pyproject_version
    assert result.latest == expected_target
    assert result.update_needed is expected_update_needed
    assert f"aiopnsense=={expected_target}" in manifest_path.read_text()
    assert f'    "aiopnsense=={expected_target}",' in pyproject_path.read_text()


def test_updater_script_updates_pyproject_pin_without_trailing_comma(
    tmp_path: Path, updater_script: ModuleType
) -> None:
    """Updater script should preserve valid TOML dependency-list formatting."""
    manifest_path, pyproject_path = _write_pin_files(
        tmp_path,
        manifest_version="1.0.8",
        pyproject_text="""[dependency-groups]
ha = [
    "homeassistant",
    "aiopnsense==1.0.8"
]
""",
    )

    result = updater_script.update_pins(
        manifest_path=manifest_path,
        pyproject_path=pyproject_path,
        latest_version="1.0.9",
    )

    assert result.update_needed is True
    assert '    "aiopnsense==1.0.9"\n' in pyproject_path.read_text()


@pytest.mark.parametrize(
    ("payload", "expected_latest"),
    [
        (
            {
                "info": {"version": "1.1.0rc1"},
                "releases": {
                    "1.0.8": [{"filename": "aiopnsense-1.0.8.tar.gz"}],
                    "1.0.9": [{"filename": "aiopnsense-1.0.9.tar.gz"}],
                    "1.1.0rc1": [{"filename": "aiopnsense-1.1.0rc1.tar.gz"}],
                },
            },
            "1.0.9",
        ),
        (
            {
                "releases": {
                    "1.0.8": [{"filename": "aiopnsense-1.0.8.tar.gz"}],
                    "1.0.9": [],
                    "1.0.10": [{"filename": "aiopnsense-1.0.10.tar.gz", "yanked": True}],
                },
            },
            "1.0.8",
        ),
    ],
)
def test_updater_script_selects_latest_stable_from_pypi_payload(
    updater_script: ModuleType,
    payload: dict[str, object],
    expected_latest: str,
) -> None:
    """Updater script should select the latest installable stable PyPI release."""
    latest = updater_script._select_latest_stable_version(
        payload,
    )

    assert latest == expected_latest


def test_updater_script_rejects_duplicate_pyproject_pins(
    tmp_path: Path,
    updater_script: ModuleType,
) -> None:
    """Updater script should fail clearly when pyproject has ambiguous pins."""
    manifest_path, pyproject_path = _write_pin_files(
        tmp_path,
        manifest_version="1.0.8",
        pyproject_text="""[dependency-groups]
ha = [
    "aiopnsense==1.0.8",
    "aiopnsense==1.0.9",
]
""",
    )

    with pytest.raises(ValueError, match="Expected exactly one pinned aiopnsense dependency"):
        updater_script.update_pins(
            manifest_path=manifest_path,
            pyproject_path=pyproject_path,
            latest_version="1.0.10",
        )


def test_release_note_script_builds_sanitized_pr_body(
    tmp_path: Path,
    release_notes_script: ModuleType,
) -> None:
    """Release-note script should build a mention-safe PR body file."""
    releases = [
        {
            "tag_name": "1.0.8",
            "name": "Ignored current",
            "body": "old",
            "html_url": "https://example.test/1.0.8",
        },
        {
            "tag_name": "1.0.9",
            "name": "Fixes @someone",
            "body": "fixes #123 and thanks [@helper](https://github.com/helper)",
            "html_url": "https://example.test/1.0.9",
        },
        {
            "tag_name": "1.1.0rc1",
            "name": "Ignored prerelease",
            "body": "future",
            "html_url": "https://example.test/1.1.0rc1",
        },
    ]
    body_path = tmp_path / "body.md"

    release_notes_script.write_pr_body(
        body_path=body_path,
        releases=releases,
        current_version="1.0.8",
        pyproject_current_version="1.0.8",
        latest_version="1.0.9",
    )

    body = body_path.read_text()
    assert "Automated update of aiopnsense dependency pins." in body
    assert "Updated pinned version: `aiopnsense==1.0.9`" in body
    assert "### Fixes @<!-- -->someone (1.0.9)" in body
    assert "fixes \\#123 and thanks helper" in body
    assert "Ignored current" not in body
    assert "Ignored prerelease" not in body


def test_release_note_script_handles_url_errors(
    tmp_path: Path,
    release_notes_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release-note script should report network failures without a traceback."""

    def raise_url_error(**_: object) -> list[dict[str, object]]:
        """Raise a urlopen-style network failure."""
        raise release_notes_script.URLError("DNS failure")

    monkeypatch.setattr(release_notes_script, "fetch_releases", raise_url_error)

    result = release_notes_script.main(
        [
            "--current-version",
            "1.0.8",
            "--pyproject-current-version",
            "1.0.8",
            "--latest-version",
            "1.0.9",
            "--release-owner",
            "Snuffy2",
            "--release-repo",
            "aiopnsense",
            "--body-path",
            str(tmp_path / "body.md"),
        ],
    )

    assert result == 1


def test_release_note_fetch_releases_paginates_with_link_header(
    release_notes_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release-note helper should follow GitHub Link pagination."""
    factory = FakeHTTPSConnectionFactory(
        [
            FakeHTTPResponse(
                status=200,
                body='[{"tag_name": "1.0.8"}]',
                headers={"Link": '<https://api.github.com/repos/o/r/releases?page=2>; rel="next"'},
            ),
            FakeHTTPResponse(status=200, body='[{"tag_name": "1.0.9"}]'),
        ]
    )
    monkeypatch.setattr(release_notes_script.http.client, "HTTPSConnection", factory)

    releases = release_notes_script.fetch_releases(owner="o", repo="r")

    assert [release["tag_name"] for release in releases] == ["1.0.8", "1.0.9"]
    assert [connection.requests[0]["path"] for connection in factory.connections] == [
        "/repos/o/r/releases?per_page=100",
        "/repos/o/r/releases?page=2",
    ]
    assert all(connection.closed for connection in factory.connections)


def test_release_note_request_json_raises_for_non_success_status(
    release_notes_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release-note helper should expose failed HTTP responses."""
    factory = FakeHTTPSConnectionFactory(
        [FakeHTTPResponse(status=500, body='{"message": "server failed"}')]
    )
    monkeypatch.setattr(release_notes_script.http.client, "HTTPSConnection", factory)

    with pytest.raises(release_notes_script.URLError, match="HTTP 500"):
        release_notes_script._request_json(
            url="https://api.github.com/repos/o/r/releases",
            headers={},
        )


def test_updater_request_json_rejects_non_object_payload(
    updater_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Updater helper should reject unexpected JSON payload shapes."""
    factory = FakeHTTPSConnectionFactory([FakeHTTPResponse(status=200, body="[]")])
    monkeypatch.setattr(updater_script.http.client, "HTTPSConnection", factory)

    with pytest.raises(TypeError, match="Expected a JSON object"):
        updater_script._request_json("https://pypi.org/pypi/aiopnsense/json")


def test_updater_request_json_raises_for_non_success_status(
    updater_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Updater helper should fail clearly for non-success HTTP responses."""
    factory = FakeHTTPSConnectionFactory([FakeHTTPResponse(status=503, body="unavailable")])
    monkeypatch.setattr(updater_script.http.client, "HTTPSConnection", factory)

    with pytest.raises(ValueError, match="HTTP 503"):
        updater_script._request_json("https://pypi.org/pypi/aiopnsense/json")


def test_cleanup_request_json_handles_no_content_response(
    cleanup_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup helper should treat no-content responses as empty payloads."""
    factory = FakeHTTPSConnectionFactory([FakeHTTPResponse(status=204, body="")])
    monkeypatch.setattr(cleanup_script.http.client, "HTTPSConnection", factory)

    payload, link_header = cleanup_script._request_json(
        method="DELETE",
        url="https://api.github.com/repos/o/r/git/refs/heads/b",
        headers={},
    )

    assert payload == {}
    assert link_header is None


def test_cleanup_request_json_raises_api_error_for_non_success_status(
    cleanup_script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup helper should preserve failed GitHub response details."""
    factory = FakeHTTPSConnectionFactory(
        [FakeHTTPResponse(status=422, body='{"message": "Reference does not exist"}')]
    )
    monkeypatch.setattr(cleanup_script.http.client, "HTTPSConnection", factory)

    with pytest.raises(cleanup_script.GithubAPIError) as exc_info:
        cleanup_script._request_json(
            method="DELETE",
            url="https://api.github.com/repos/o/r/git/refs/heads/b",
            headers={},
        )

    assert exc_info.value.status == 422
    assert "Reference does not exist" in exc_info.value.body


def test_cleanup_script_closes_stale_prs_and_deletes_workflow_branches(
    cleanup_script: ModuleType,
) -> None:
    """Cleanup script should close stale PRs and remove workflow-created branches."""
    client = FakeCleanupClient(
        open_pulls=[
            _workflow_pull(number=10),
            _workflow_pull(number=9, ref="chore/update-aiopnsense-old"),
            _workflow_pull(number=11, ref="feature/manual"),
        ],
        closed_pulls=[
            _workflow_pull(number=8, merged_at="2026-05-28T00:00:00Z"),
            _workflow_pull(number=7, ref="chore/update-aiopnsense-old"),
        ],
    )

    result = cleanup_script.cleanup_update_branches(
        client=client,
        repository="o/r",
        branch="chore/update-aiopnsense-manifest",
        branch_prefix="chore/update-aiopnsense",
        label_name="aiopnsense-auto-update",
        keep_pr_number=None,
        close_stale_prs=True,
        delete_stale_branch=True,
        delete_merged_branches=True,
    )

    assert client.closed_prs == [10, 9]
    assert client.deleted_refs == [
        "heads/chore/update-aiopnsense-manifest",
        "heads/chore/update-aiopnsense-old",
    ]
    assert result.closed_prs == [10, 9]
    assert result.deleted_branches == [
        "chore/update-aiopnsense-manifest",
        "chore/update-aiopnsense-old",
    ]


def test_cleanup_script_keeps_active_update_branch(cleanup_script: ModuleType) -> None:
    """Cleanup script should not delete the branch for the kept update PR."""
    client = FakeCleanupClient(
        open_pulls=[_workflow_pull(number=12)],
        closed_pulls=[
            _workflow_pull(number=8, merged_at="2026-05-28T00:00:00Z"),
            _workflow_pull(
                number=6,
                ref="chore/update-aiopnsense-old",
                merged_at="2026-05-20T00:00:00Z",
            ),
        ],
        fail_on_close=True,
    )

    result = cleanup_script.cleanup_update_branches(
        client=client,
        repository="o/r",
        branch="chore/update-aiopnsense-manifest",
        branch_prefix="chore/update-aiopnsense",
        label_name="aiopnsense-auto-update",
        keep_pr_number=12,
        close_stale_prs=True,
        delete_stale_branch=False,
        delete_merged_branches=True,
    )

    assert client.deleted_refs == ["heads/chore/update-aiopnsense-old"]
    assert result.closed_prs == []
    assert result.deleted_branches == ["chore/update-aiopnsense-old"]
