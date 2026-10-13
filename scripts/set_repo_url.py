"""Fill in the GitHub repository URL everywhere it is referenced, before the first push.

    python scripts/set_repo_url.py https://github.com/<username>/GazeGrounded-UX

Updates the README links, the website's GitHub buttons and clone command, and CITATION.cff.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "https://github.com/OWNER/GazeGrounded-UX"


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", sys.argv[1]):
        print(__doc__)
        return 1
    url = sys.argv[1].rstrip("/")
    owner, repo = url.split("/")[-2:]
    pages = f"https://{owner.lower()}.github.io/{repo}/"

    readme = ROOT / "README.md"
    text = readme.read_text().replace(PLACEHOLDER, url)
    text = text.replace("https://OWNER.github.io/GazeGrounded-UX/", pages)
    readme.write_text(text)

    app = ROOT / "docs" / "assets" / "app.js"
    app.write_text(
        re.sub(r'const REPO_URL = "[^"]*";', f'const REPO_URL = "{url}";', app.read_text())
    )

    cff = ROOT / "CITATION.cff"
    lines = [ln for ln in cff.read_text().splitlines() if not ln.startswith("repository-code:")]
    lines.append(f'repository-code: "{url}"')
    cff.write_text("\n".join(lines) + "\n")
    print(f"Repository: {url}\nWebsite:    {pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
