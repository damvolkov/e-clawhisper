"""Entry point — routes to CLI."""

from __future__ import annotations

from e_heed.cli.main import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
