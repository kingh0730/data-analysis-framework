import enum
from itertools import chain
from typing import Any, Optional, Sequence
from django.db import models
from django.dispatch import receiver
from django.db.models.signals import post_save, m2m_changed
from django.core.exceptions import FieldError
import pandas as pd

from base.jobs.lower_jobs import on_transaction_commit
from base.models import (
    EIGHTY_TWO_INDUSTRY_CHOICES,
    SEVEN_AREAS_CHOICES,
    Job,
    Month,
)
from techniques.technique_exec import (
    MINING_TECHNIQUES,
    SAMPLE_SELECTION_TECHNIQUES,
    MiningTechnique,
    SampleSelectionTechnique,
    exec_technique,
)
from techniques.technique_itp import (
    MINING_TECHNIQUES_ITP,
    SAMPLE_SELECTION_TECHNIQUES_ITP,
    aggregate_interpretations,
)
from utils import (
    INCOME_PER_CAPITA_INDEX,
    POP_MONTH_INDEX,
    GovLevel,
    calc_district_income_level,
    calc_district_population_level,
    calc_district_seven_areas_level,
    get_gov_level_ids,
)
from utils.provincial_capitals import PROVINCIAL_CAPITALS_CHILDREN


# To samples classes


def sample_selection_districts_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    if gov_id not in get_gov_level_ids(GovLevel.DISTRICT_LEVEL):
        raise ValueError("Gov is not on the district level.")

    return {}


def sample_selection_districts_population_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    if gov_id not in get_gov_level_ids(GovLevel.DISTRICT_LEVEL):
        raise ValueError("Gov is not on the district level.")

    all_districts = exec_technique(
        SAMPLE_SELECTION_TECHNIQUES[SampleSelectionTechnique.DISTRICTS],
        month_=month,
    )
    pop_level = calc_district_population_level(
        pd.read_csv(all_districts.output.cached_file.path, index_col=0)[
            str(POP_MONTH_INDEX)
        ][gov_id]
    )
    samples_classes = {
        "pop_level": pop_level,
    }
    return samples_classes


def sample_selection_districts_income_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    if gov_id not in get_gov_level_ids(GovLevel.DISTRICT_LEVEL):
        raise ValueError("Gov is not on the district level.")

    all_districts = exec_technique(
        SAMPLE_SELECTION_TECHNIQUES[SampleSelectionTechnique.DISTRICTS],
        month_=month,
    )
    income_level = calc_district_income_level(
        pd.read_csv(all_districts.output.cached_file.path, index_col=0)[
            str(INCOME_PER_CAPITA_INDEX)
        ][gov_id]
    )
    samples_classes = {
        "income_level": income_level,
    }
    return samples_classes


def sample_selection_districts_seven_areas_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    if gov_id not in get_gov_level_ids(GovLevel.DISTRICT_LEVEL):
        raise ValueError("Gov is not on the district level.")

    seven_areas_level = calc_district_seven_areas_level(gov_id)

    class DummySevenAreasLevel:
        name = seven_areas_level
        value = [
            choice[0]
            for choice in SEVEN_AREAS_CHOICES
            if choice[1] == seven_areas_level
        ][0]

    samples_classes = {
        "seven_areas_level": DummySevenAreasLevel(),
    }
    return samples_classes


def sample_selection_districts_provincial_capitals_children_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    provincial_capitals_children_ids = [
        child["gov_id"] for child in PROVINCIAL_CAPITALS_CHILDREN
    ]
    if gov_id not in provincial_capitals_children_ids:
        raise ValueError("Gov is not a provincial capital's child.")

    return {}


def sample_selection_districts_by_industry_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    if gov_id not in get_gov_level_ids(GovLevel.DISTRICT_LEVEL):
        raise ValueError("Gov is not on the district level.")

    daas_id = kwargs["daas_id"]
    industry_id, industry_name = [
        choice for choice in EIGHTY_TWO_INDUSTRY_CHOICES if choice[0] == daas_id
    ][0]

    # Check if gov_id is among the highest in the given industry
    districts_high_by_industry = exec_technique(
        SAMPLE_SELECTION_TECHNIQUES[SampleSelectionTechnique.DISTRICTS_BY_INDUSTRY],
        month_=month,
        industry_=industry_id,
    )
    if (
        gov_id
        not in pd.read_csv(
            districts_high_by_industry.output.cached_file.path, index_col=0
        )["gov_id"].values
    ):
        raise ValueError("Gov is not high in the given industry.")

    class DummyByIndustryLevel:
        name = industry_name
        value = industry_id

    samples_classes = {
        "industry": DummyByIndustryLevel(),
    }
    return samples_classes


def sample_selection_time_series_to_classes(
    gov_id: int, month: Month, **kwargs: Any
) -> dict:
    month = month.month_int
    PAST_N_MONTHS = 24
    months = [
        Month.objects.get(
            month_int=month_int,
        )
        for month_int in range(month - PAST_N_MONTHS + 1, month + 1)
    ]

    class DummyGovId:
        name = None
        value = gov_id

    class DummyMonths:
        name = f"{months[0].month_str}到{months[-1].month_str}"
        value = months

    return {
        "gov_id": DummyGovId(),
        "months": DummyMonths(),
    }


# Sample selection
SAMPLE_SELECTION_TO_CLASSES = {
    SampleSelectionTechnique.DISTRICTS: sample_selection_districts_to_classes,
    SampleSelectionTechnique.DISTRICTS_POPULATION: sample_selection_districts_population_to_classes,
    SampleSelectionTechnique.DISTRICTS_INCOME: sample_selection_districts_income_to_classes,
    SampleSelectionTechnique.DISTRICTS_SEVEN_AREAS: sample_selection_districts_seven_areas_to_classes,
    SampleSelectionTechnique.DISTRICTS_PROVINCIAL_CAPITALS_CHILDREN: sample_selection_districts_provincial_capitals_children_to_classes,
    SampleSelectionTechnique.DISTRICTS_BY_INDUSTRY: sample_selection_districts_by_industry_to_classes,
    SampleSelectionTechnique.TIME_SERIES: sample_selection_time_series_to_classes,
}


# Create your models here.


class InputToOutputDocument(models.Model):
    index_id = models.PositiveSmallIntegerField()
    gov_id = models.PositiveSmallIntegerField()
    month = models.ForeignKey(Month, on_delete=models.PROTECT)

    # Todo: May not be the most elegant
    sample_selection_kwargs = models.JSONField(blank=True, default=dict)

    output = models.JSONField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    progress = models.PositiveSmallIntegerField(default=0, blank=True)


@receiver(post_save, sender=InputToOutputDocument)
@on_transaction_commit
def create_input_to_output_document_job(
    sender: Any, instance: InputToOutputDocument, created: bool, **kwargs: Any
) -> None:
    if created:
        month = instance.month
        index_id = instance.index_id
        gov_id = instance.gov_id
        sample_selection_kwargs = instance.sample_selection_kwargs

        # Techniques
        SAMPLE_SELECTION = SampleSelectionTechnique.DISTRICTS_BY_INDUSTRY
        MININGS = [
            MiningTechnique.PEARSON_CORR,
        ]

        # Determine classes
        samples_classes = SAMPLE_SELECTION_TO_CLASSES[SAMPLE_SELECTION](
            gov_id, month, **sample_selection_kwargs
        )
        samples_classes_values = {f"{k}_": v.value for k, v in samples_classes.items()}

        # Execute technique
        samples_job = exec_technique(
            SAMPLE_SELECTION_TECHNIQUES[SAMPLE_SELECTION],
            month_=month,
            **samples_classes_values,
        )
        mining_jobs = [
            exec_technique(
                MINING_TECHNIQUES[mining],
                samples_dff_=samples_job.output,
            )
            for mining in MININGS
        ]

        # Get what I want
        samples_itp = SAMPLE_SELECTION_TECHNIQUES_ITP[SAMPLE_SELECTION](
            index_id,
            gov_id,
            pd.read_csv(samples_job.output.cached_file.path, index_col=0),
            **samples_classes,
        )
        relations_itp = list(
            chain.from_iterable(
                [
                    MINING_TECHNIQUES_ITP[mining](
                        index_id,
                        gov_id,
                        pd.read_csv(job.output.cached_file.path, index_col=0),
                    )
                    for mining, job in zip(MININGS, mining_jobs)
                ]
            )
        )

        json_obj = aggregate_interpretations(
            samples_itp,
            relations_itp,
            # Todo: This will not be the only case where there is a unique gov.
            SAMPLE_SELECTION != SampleSelectionTechnique.TIME_SERIES,
        )

        # Save
        instance.output = json_obj

        instance.progress = 100
        instance.save()
