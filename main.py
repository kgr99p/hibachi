"""Hibachi BTC Grid Trading Bot - Entry Point"""

import argparse
import asyncio
import signal
import sys
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from bot import GridBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hibachi_grid")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def main():
    parser = argparse.ArgumentParser(description="Hibachi BTC Grid Trading Bot")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Simulation mode (no real orders)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load .env from script directory
    script_dir = Path(__file__).parent
    env_path = script_dir / ".env"
    if not env_path.exists():
        logger.error(".env 파일이 없습니다. .env.example을 참고하여 .env를 생성하세요.")
        sys.exit(1)
    load_dotenv(env_path)

    config_path = script_dir / args.config
    config = load_config(str(config_path))
    logger.info(f"Config loaded: {config_path}")

    bot = GridBot(config=config, dry_run=args.dry_run)

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            signal.signal(sig, lambda s, f: _signal_handler())

    try:
        await bot.initialize()
        await bot.run(shutdown_event)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await bot.shutdown()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
