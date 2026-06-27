"""Clean up stale aiopnsense update pull requests and branches."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import logging
import os
import sys
from typing import Protocol

import requests

GITHUB_API_URL = "https://api.github.com"
LOGGER = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of cleaning workflow-owned pull requests and branches.

    Attributes:
        closed_prs: Pull request numbers closed as stale.
        deleted_branches: Branch names deleted from the repository.
    """

    closed_prs: list[int] = field(default_factory=list)
    deleted_branches: list[str] = field(default_factory=list)


class GithubCleanupClient(Protocol):
    """Protocol for GitHub operations needed by the cleanup routine."""

    def list_pulls(self, *, state: str) -> list[dict[str, object]]:
        """List pull requests by state."""

    def close_pull(self, pull_number: int) -> None:
        """Close a pull request by number."""

    def delete_ref(self, ref: str) -> None:
        """Delete a git ref by name."""


class GithubClient:
    """Small GitHub REST client for the workflow cleanup task."""

    def __init__(self, *, repository: str, token: str) -> None:
        """Initialize the client.

        Args:
            repository: Repository in ``owner/name`` format.
            token: GitHub token for API calls.
        """
        self.repository = repository
        self.token = token

    def list_pulls(self, *, state: str) -> list[dict[str, object]]:
        """List pull requests by state.

        Args:
            state: Pull request state to request.

        Returns:
            Pull request objects from GitHub.
        """
        pulls: list[dict[str, object]] = []
        url: str | None = (
            f"{GITHUB_API_URL}/repos/{self.repository}/pulls?state={state}&per_page=100"
        )
        while url is not None:
            payload, link_header = self._request("GET", url)
            if not isinstance(payload, list):
                raise TypeError(f"Expected pull request list from {url}")
            pulls.extend(pull for pull in payload if isinstance(pull, dict))
            url = _next_link(link_header)
        return pulls

    def close_pull(self, pull_number: int) -> None:
        """Close a pull request.

        Args:
            pull_number: Pull request number to close.
        """
        url = f"{GITHUB_API_URL}/repos/{self.repository}/pulls/{pull_number}"
        self._request("PATCH", url, payload={"state": "closed"})

    def delete_ref(self, ref: str) -> None:
        """Delete a git ref if it exists.

        Args:
            ref: Ref path such as ``heads/branch-name``.
        """
        url = f"{GITHUB_API_URL}/repos/{self.repository}/git/refs/{ref}"
        try:
            self._request("DELETE", url)
        except GithubAPIError as err:
            if err.status == 404 or (err.status == 422 and _is_missing_ref_error(err.body)):
                LOGGER.info("Ref %s was already deleted.", ref)
                return
            raise

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> tuple[object, str | None]:
        """Send a GitHub REST request.

        Args:
            method: HTTP method.
            url: Request URL.
            payload: Optional JSON payload.

        Returns:
            Decoded payload and Link header.
        """
        return _request_json(
            method=method,
            url=url,
            headers=_github_headers(self.token),
            payload=payload,
        )


class GithubAPIError(Exception):
    """Error raised for non-successful GitHub API responses."""

    def __init__(self, status: int, body: str) -> None:
        """Initialize a GitHub API error.

        Args:
            status: HTTP status code.
            body: Response body.
        """
        super().__init__(f"GitHub API request failed with HTTP {status}")
        self.status = status
        self.body = body


def _request_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object] | None = None,
) -> tuple[object, str | None]:
    """Send a JSON request to the GitHub API over HTTPS.

    Args:
        method: HTTP method.
        url: Target HTTPS URL.
        headers: Request headers.
        payload: Optional JSON payload.

    Returns:
        Decoded JSON payload and Link header.

    Raises:
        GithubAPIError: If the response is not successful.
    """
    if not url.startswith("https://"):
        raise GithubAPIError(0, f"Unsupported URL: {url}")

    data = json.dumps(payload).encode() if payload is not None else None

    try:
        response = requests.request(method, url, data=data, headers=dict(headers), timeout=30)
    except requests.RequestException as err:
        raise GithubAPIError(0, str(err)) from err
    if response.status_code < 200 or response.status_code >= 300:
        raise GithubAPIError(response.status_code, response.text)
    if response.status_code == 204 or not response.content:
        return {}, response.headers.get("Link")
    return response.json(), response.headers.get("Link")


def cleanup_update_branches(
    *,
    client: GithubCleanupClient,
    repository: str,
    branch: str,
    branch_prefix: str,
    label_name: str,
    keep_pr_number: int | None,
    close_stale_prs: bool,
    delete_stale_branch: bool,
    delete_merged_branches: bool,
) -> CleanupResult:
    """Clean workflow-owned stale pull requests and branches.

    Args:
        client: GitHub client with list, close, and delete methods.
        repository: Repository in ``owner/name`` format.
        branch: Current workflow update branch.
        branch_prefix: Prefix for workflow-owned update branches.
        label_name: Label identifying workflow-created PRs.
        keep_pr_number: Optional PR number to preserve.
        close_stale_prs: Whether to close open stale update PRs.
        delete_stale_branch: Whether to delete the current stale branch.
        delete_merged_branches: Whether to delete branches from merged update PRs.

    Returns:
        Summary of cleanup actions.
    """
    result = CleanupResult()
    protected_branches: set[str] = set()
    branches_to_delete: set[str] = set()

    open_pulls = client.list_pulls(state="open")
    for pull in open_pulls:
        if not _is_workflow_pull(
            pull,
            repository=repository,
            branch=branch,
            branch_prefix=branch_prefix,
            label_name=label_name,
        ):
            continue

        pull_number = _pull_number(pull)
        head_ref = _head_ref(pull)
        if keep_pr_number is not None and pull_number == keep_pr_number:
            protected_branches.add(head_ref)
            continue

        if close_stale_prs:
            client.close_pull(pull_number)
            result.closed_prs.append(pull_number)
            branches_to_delete.add(head_ref)

    if delete_stale_branch and branch not in protected_branches:
        branches_to_delete.add(branch)

    if delete_merged_branches:
        closed_pulls = client.list_pulls(state="closed")
        for pull in closed_pulls:
            if pull.get("merged_at") is not None and _is_workflow_pull(
                pull,
                repository=repository,
                branch=branch,
                branch_prefix=branch_prefix,
                label_name=label_name,
            ):
                branches_to_delete.add(_head_ref(pull))

    branches_to_delete -= protected_branches
    for branch_name in sorted(branches_to_delete):
        client.delete_ref(f"heads/{branch_name}")
        result.deleted_branches.append(branch_name)

    return result


def _is_workflow_pull(
    pull: Mapping[str, object],
    *,
    repository: str,
    branch: str,
    branch_prefix: str,
    label_name: str,
) -> bool:
    """Return whether a pull request belongs to this workflow.

    Args:
        pull: Pull request object.
        repository: Repository in ``owner/name`` format.
        branch: Current workflow update branch.
        branch_prefix: Prefix for workflow-owned update branches.
        label_name: Label identifying workflow-created PRs.

    Returns:
        True when the PR head branch is owned by this workflow.
    """
    labels = pull.get("labels", [])
    if not isinstance(labels, list) or not any(
        isinstance(label, dict) and label.get("name") == label_name for label in labels
    ):
        return False

    head = pull.get("head", {})
    if not isinstance(head, dict):
        return False
    head_ref = head.get("ref")
    if not isinstance(head_ref, str):
        return False
    if head_ref != branch and not head_ref.startswith(branch_prefix):
        return False

    head_repo = head.get("repo", {})
    if not isinstance(head_repo, dict):
        return False
    return head_repo.get("full_name") == repository


def _head_ref(pull: Mapping[str, object]) -> str:
    """Return a pull request head ref.

    Args:
        pull: Pull request object.

    Returns:
        Pull request head ref.
    """
    head = pull["head"]
    if not isinstance(head, dict) or not isinstance(head.get("ref"), str):
        raise TypeError("Pull request is missing a head ref")
    return head["ref"]


def _pull_number(pull: Mapping[str, object]) -> int:
    """Return a pull request number.

    Args:
        pull: Pull request object.

    Returns:
        Pull request number.
    """
    number = pull["number"]
    if isinstance(number, int):
        return number
    if isinstance(number, str):
        return int(number)
    raise TypeError("Pull request is missing a numeric number")


def _github_headers(token: str) -> dict[str, str]:
    """Build GitHub API request headers.

    Args:
        token: GitHub token.

    Returns:
        Request headers.
    """
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "hass-opnsense-update-aiopnsense-cleanup",
    }


def _is_missing_ref_error(body: str) -> bool:
    """Return whether a GitHub 422 error reports a missing ref.

    Args:
        body: Response body from the GitHub API.

    Returns:
        True when the error body says the reference does not exist.
    """
    if not body:
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return "Reference does not exist" in body
    if not isinstance(payload, dict):
        return False
    message = payload.get("message")
    return isinstance(message, str) and "Reference does not exist" in message


def _next_link(link_header: str | None) -> str | None:
    """Return the next pagination URL from a GitHub Link header.

    Args:
        link_header: Raw Link header value.

    Returns:
        Next URL when present.
    """
    if not link_header:
        return None
    for link in link_header.split(","):
        url_part, *params = link.split(";")
        if any(param.strip() == 'rel="next"' for param in params):
            return url_part.strip()[1:-1]
    return None


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments excluding the executable name.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--branch-prefix", required=True)
    parser.add_argument("--label-name", required=True)
    parser.add_argument("--keep-pr-number", type=int)
    parser.add_argument("--close-stale-prs", action="store_true")
    parser.add_argument("--delete-stale-branch", action="store_true")
    parser.add_argument("--delete-merged-branches", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run workflow branch cleanup.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        LOGGER.error("GITHUB_TOKEN is required for cleanup.")
        return 1

    client = GithubClient(repository=args.repository, token=token)
    result = cleanup_update_branches(
        client=client,
        repository=args.repository,
        branch=args.branch,
        branch_prefix=args.branch_prefix,
        label_name=args.label_name,
        keep_pr_number=args.keep_pr_number,
        close_stale_prs=args.close_stale_prs,
        delete_stale_branch=args.delete_stale_branch,
        delete_merged_branches=args.delete_merged_branches,
    )
    LOGGER.info("Closed stale PRs: %s", result.closed_prs or "none")
    LOGGER.info("Deleted branches: %s", result.deleted_branches or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
