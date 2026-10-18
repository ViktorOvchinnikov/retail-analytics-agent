from __future__ import annotations

import argparse
import sys
import uuid

from .agent import RetailAgent
from .analytics import BigQueryRunner
from .config import Settings
from .observability import configure_logging
from .safety import profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retail analytics agent")
    parser.add_argument(
        "--user",
        default="manager_1",
        choices=("manager_1", "manager_2", "ceo"),
    )
    parser.add_argument("--conversation", default=None)
    parser.add_argument(
        "--list-brands",
        action="store_true",
        help="List brands from BigQuery, then exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    configure_logging(settings.log_path)

    if args.list_brands:
        try:
            for brand in BigQueryRunner(settings).list_brands():
                print(brand)
        except Exception as error:
            print(f"Could not list brands: {error}", file=sys.stderr)
            raise SystemExit(2) from error
        return

    try:
        agent = RetailAgent(settings)
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    conversation = args.conversation or str(uuid.uuid4())
    profile = profiles(settings)[args.user]
    scope = "all brands" if profile.role == "ceo" else ", ".join(profile.allowed_brands)
    print(f"Retail Analytics Agent — {args.user} ({scope})")
    print(f"Conversation: {conversation}")
    print("Type /help for commands, or quit to exit.\n")

    try:
        while True:
            try:
                message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message.casefold() in {"quit", "exit", "/quit"}:
                break
            if message.casefold() == "/help":
                print(HELP_TEXT)
                continue
            try:
                answer = agent.chat(
                    message, user_id=args.user, conversation_id=conversation
                )
            except Exception as error:
                print(f"agent> Request failed safely: {error}")
                continue
            print(f"agent> {answer}\n")
    finally:
        agent.close()


HELP_TEXT = """Ask for revenue, order count, or AOV grouped by brand, product,
state, or month. Basic product and period comparisons are supported.

Report commands:
  /save
  /reports
  delete all reports from this conversation
  delete reports mentioning TEXT
  /confirm ACTION_ID TOKEN
  /cancel ACTION_ID
"""


if __name__ == "__main__":
    main()
