"""Точка входа: ``python -m hh_radar.mcp_server``."""

from __future__ import annotations

import logging
import sys

from hh_radar.mcp_server.server import server


def main() -> None:
    """Настраивает логирование в stderr и поднимает stdio-сервер.

    stdio-транспорт MCP использует stdout как единственный канал протокола:
    любой print или логгер, пишущий в stdout, перемешивается с сообщениями
    JSON-RPC и ломает клиента. Поэтому весь логгинг здесь явно направлен
    в stderr, а не оставлен на дефолт.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server.run("stdio")


if __name__ == "__main__":
    main()
