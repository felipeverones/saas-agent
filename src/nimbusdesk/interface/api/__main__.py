"""Run the API: `uv run python -m nimbusdesk.interface.api` (or `make run`)."""

import uvicorn

from nimbusdesk.interface.api.app import create_app

app = create_app()


def main() -> None:
    uvicorn.run(
        "nimbusdesk.interface.api.__main__:app", host="0.0.0.0", port=8000, log_level="info"
    )


if __name__ == "__main__":
    main()
