from typing import Any
from django.core.management.base import BaseCommand, CommandParser, CommandError
from base.models import Month


class Command(BaseCommand):
    help = "Populate months"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "-s", "--start", type=int, help="start month", required=True
        )
        parser.add_argument("-e", "--end", type=int, help="end month", required=True)

    def handle(self, *args: Any, **options: Any) -> None:
        start = options["start"]
        end = options["end"]
        populate_months(start, end)


def populate_months(start: int, end: int) -> None:
    for month_int in range(start, end + 1):
        exists = Month.objects.filter(month_int=month_int).exists()
        if not exists:
            month = Month.objects.create_month(month_int=month_int)
            print(f"created: {month}")
        else:
            print(f"exists: {month_int}")
