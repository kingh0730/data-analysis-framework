from typing import Any
from django.core.management.base import BaseCommand, CommandParser, CommandError
from base.models import Month


class Command(BaseCommand):
    help = "Download one month with all govs and all indexes"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-m", "--month", type=int, help="month", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        month = options["month"]
        download_one_month(month)


def download_one_month(month: int) -> None:
    print(f"Todo: {month}")
