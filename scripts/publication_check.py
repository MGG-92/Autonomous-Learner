"""A conservative local preflight, not a guarantee that content is secret-free."""
import re
import subprocess
import sys
from pathlib import Path


PATTERNS = {
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "OpenAI credential": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    "Windows personal path": re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\s\"']+", re.I),
    "Unix personal path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
}
FORBIDDEN = {".pt", ".pth", ".sqlite", ".sqlite3", ".db", ".pem", ".key", ".pfx", ".zip"}


def public_email(email):
    # GitHub creates a synthetic merge commit for PR CI with its service identity.
    # Human author/committer addresses must still use GitHub's no-reply domain.
    return email.lower() == "noreply@github.com" or email.lower().endswith("@users.noreply.github.com")


def git(*args):
    return subprocess.check_output(["git", *args])


def check_content(name, content):
    failures = []
    if len(content) > 1_000_000:
        failures.append(f"{name}: unexpectedly large tracked file")
    decoded = content.decode("utf-8", errors="replace")
    for label, pattern in PATTERNS.items():
        if pattern.search(decoded):
            failures.append(f"{name}: possible {label} (value suppressed)")
    return failures


def main():
    errors = []
    paths = git("ls-files", "-z").decode().split("\0")
    for name in filter(None, paths):
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN or any(p in {".venv", "runs", "attachments", ".aws"} for p in path.parts) or path.name.startswith(".env"):
            errors.append(f"{name}: private/runtime artifact tracked")
        if path.is_file():
            errors.extend(check_content(name, path.read_bytes()))
    # Only current branch ancestry: remote-tracking refs may temporarily point at
    # the private bootstrap commit during the initial metadata cleanup.
    revisions = git("rev-list", "HEAD").decode().split()
    checked = set()
    for revision in revisions:
        for line in git("ls-tree", "-r", revision).decode().splitlines():
            metadata, name = line.split("\t", 1)
            object_id = metadata.split()[2]
            if object_id not in checked:
                errors.extend(check_content(f"history:{name}", git("cat-file", "blob", object_id)))
                checked.add(object_id)
        emails = git("show", "-s", "--format=%ae%n%ce", revision).decode().splitlines()
        if any(not public_email(email) for email in emails):
            errors.append(f"commit {revision[:8]}: author/committer email is not GitHub no-reply")
    if errors:
        print("Publication preflight failed:\n" + "\n".join(sorted(set(errors))))
        return 1
    print(f"Publication preflight passed: {len(list(filter(None, paths)))} tracked files, {len(revisions)} commits.")
    print("Review staged diffs manually too; pattern scanning cannot prove absence of secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
