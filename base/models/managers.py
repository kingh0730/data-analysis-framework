import time
from typing import Any


from django.db import models
from django.core.files.base import File
import pandas as pd

from server.settings import MEDIA_ROOT


from utils import month_int_to_str


class MonthManager(models.Manager):
    def create_month(self, month_int: int) -> models.Model:
        month = self.create(month_int=month_int, month_str=month_int_to_str(month_int))
        return month


class DataFrameFileManager(models.Manager):
    def create_data_frame_file(
        self,
        data_frame: pd.DataFrame,
        # predecessor_data_frame_file: models.Model,
        job: models.Model,
        file_name: str = None,
        **kwargs: Any,
    ) -> models.Model:
        def curr_time_str() -> str:
            return (
                time.asctime(time.localtime(time.time()))
                .replace(" ", "_")
                .replace(":", "-")
            )

        def calc_relative_file_path() -> str:
            return (
                f"{job.id}---{job.__class__.__name__}___{curr_time_str()}.csv"
                if not file_name
                else file_name
            )

        try:
            relative_file_path = calc_relative_file_path()
            data_frame.to_csv(MEDIA_ROOT / relative_file_path, mode="x")
        except FileExistsError:
            time.sleep(1)
            relative_file_path = calc_relative_file_path()
            data_frame.to_csv(MEDIA_ROOT / relative_file_path, mode="x")

        data_frame_file = self.create(**kwargs)
        data_frame_file.cached_file.name = relative_file_path
        data_frame_file.save()
        return data_frame_file

        # with open(file_path, encoding="utf-8") as file_io:

        #     # history = predecessor_data_frame_file.history + [
        #     #     [
        #     #         {
        #     #             "id": predecessor_data_frame_file.id,
        #     #             "file": str(predecessor_data_frame_file.cached_file),
        #     #         },
        #     #         {"id": job.id, "job": job.__class__.__name__},
        #     #     ]
        #     # ]

        #     data_frame_file = self.create(cached_file=File(file_io), **kwargs)
        #     return data_frame_file


class JobManager(models.Manager):
    def create(
        self,
        *args: Any,
        should_output_in_memory: bool = True,
        should_output_in_file: bool = True,
        in_memory_inputs: dict[str, models.Model] = None,
        **kwargs: Any,
    ) -> models.Model:
        """
        Will create attributes called:
        - my_manager_created
        - should_output_in_memory
        - in_memory_inputs
            - The job should validate the input format and transform it if necessary.
        - in_memory_outputs

        Todo: Implement in-memory for all jobs:
        - inputs
        - inputs validation/transformation
        - outputs
        """
        result = super().create(*args, **kwargs)
        result.my_manager_created = True
        result.should_output_in_memory = should_output_in_memory
        result.should_output_in_file = should_output_in_file
        result.in_memory_inputs = in_memory_inputs if in_memory_inputs else {}
        result.in_memory_outputs = {}
        result.save()
        return result


class TimeSeriesSimilaritiesManager(JobManager):
    def create_time_series_similarities(
        self, months: list[models.Model], **kwargs: Any
    ) -> models.Model:
        if not months:
            raise ValueError("Months cannot be empty!")
        similarity_matrix = self.create()
        similarity_matrix.input_months.set(months)
        return similarity_matrix


class DownloadOneGovAcrossTimeManager(JobManager):
    def create_download_one_gov_across_time(
        self, months: list[models.Model], gov_id: int, **kwargs: Any
    ) -> models.Model:
        if not months:
            raise ValueError("Months cannot be empty!")
        one_gov_across_time = super().create(gov_id=gov_id)
        one_gov_across_time.months.set(months)
        return one_gov_across_time

    # * Not sure if this is a bad idea.
    create = create_download_one_gov_across_time  # type: ignore


class DownloadOneGovMacroDataAcrossTimeManager(JobManager):
    def create_download_one_gov_macro_data_across_time(
        self, months: list[models.Model], gov_id: int, **kwargs: Any
    ) -> models.Model:
        if not months:
            raise ValueError("Months cannot be empty!")
        one_gov_macro_data_across_time = self.create(gov_id=gov_id)
        one_gov_macro_data_across_time.months.set(months)
        return one_gov_macro_data_across_time
