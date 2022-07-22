import os
import sys
from pathlib import Path

import django

PROJECT_ROOT_PATH = Path(os.path.abspath(__file__)).parents[1]
DJANGO_ROOT_PATH = PROJECT_ROOT_PATH


def django_setup() -> None:
    sys.path.append(DJANGO_ROOT_PATH.as_posix())

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")

    django.setup()


django_setup()
