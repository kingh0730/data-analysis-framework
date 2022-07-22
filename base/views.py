import json
from typing import Any, Iterable, Optional
from django.core import serializers
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

from .models import (
    ALL_INPUTS_RELATED_NAMES,
    ALL_OUTPUTS_RELATED_NAMES,
    DATA_FRAME_FILE_TYPES,
    RELATED_NAMES,
    DataFrameFile,
    Job,
)

# Create your views here.


def home(request: HttpRequest) -> HttpResponse:
    context = {"title": "Home"}
    return render(request, "base/home.html", context)


def streamlit(request: HttpRequest) -> HttpResponse:
    # context = {"title": "Streamlit"}
    return render(request, "base/streamlit.html")


# Tests


def test_all_related_names(request: HttpRequest) -> JsonResponse:
    return JsonResponse(
        {"inputs": ALL_INPUTS_RELATED_NAMES, "outputs": ALL_OUTPUTS_RELATED_NAMES}
    )


def test_job(request: HttpRequest, job_id: int) -> JsonResponse:
    job = Job.objects.get(id=job_id)
    serialized = serializers.serialize("json", [job])
    struct = json.loads(serialized)
    return JsonResponse(struct[0])


def test_data_frame_file(request: HttpRequest, dff_id: int) -> JsonResponse:
    dff = DataFrameFile.objects.get(id=dff_id)
    serialized = serializers.serialize("json", [dff])
    struct = json.loads(serialized)
    return JsonResponse(struct[0])


def test_specific_job(request: HttpRequest, job_id: int) -> JsonResponse:
    job = Job.objects.get(id=job_id)
    specific_job = job.get_specific_job()

    # JSON

    serialized = serializers.serialize("json", [specific_job])
    struct = json.loads(serialized)
    return JsonResponse(struct[0])


def test_specific_data_frame_file(request: HttpRequest, dff_id: int) -> JsonResponse:
    dff = DataFrameFile.objects.get(id=dff_id)
    specific_dff = dff.get_specific_data_frame_file()

    # JSON

    serialized = serializers.serialize("json", [specific_dff])
    struct = json.loads(serialized)
    return JsonResponse(struct[0])


def test_get_upper_job_of_data_frame_file(
    request: HttpRequest, dff_id: int
) -> JsonResponse:
    dff = DataFrameFile.objects.get(id=dff_id)
    upper_job = dff.get_upper_job()

    # JSON

    serialized = serializers.serialize("json", [upper_job])
    struct = json.loads(serialized)
    return JsonResponse(struct[0])


def test_get_all_inputs_of_job(request: HttpRequest, job_id: int) -> JsonResponse:
    job = Job.objects.get(id=job_id)
    all_inputs = job.get_all_inputs()

    # # Does not work because of many to many relationships

    # if all(all_inputs.values()):
    #     serialized = serializers.serialize("json", all_inputs.values())
    #     struct = json.loads(serialized)
    #     zipped = dict(zip(all_inputs, struct))
    #     return JsonResponse(zipped)

    def ser(item_or_items: DataFrameFile | Iterable[DataFrameFile] | None) -> Any:
        if item_or_items is None:
            return None
        if isinstance(item_or_items, DataFrameFile):
            serialized = serializers.serialize("json", [item_or_items])
            struct = json.loads(serialized)
            return struct[0]

        serialized = serializers.serialize("json", item_or_items)
        struct = json.loads(serialized)
        return struct

    serialized_inputs = [ser(item) for item in all_inputs.values()]
    return JsonResponse(dict(zip(all_inputs, serialized_inputs)))


def test_get_all_history_of_data_frame_file(
    request: HttpRequest, dff_id: int
) -> JsonResponse:
    dff = DataFrameFile.objects.get(id=dff_id)
    specific_dff = dff.get_specific_data_frame_file()
    res = get_all_history_of_specific_data_frame_file(specific_dff)

    return JsonResponse(res)


def test_get_all_history_of_job(request: HttpRequest, job_id: int) -> JsonResponse:
    job = Job.objects.get(id=job_id)
    specific_job = job.get_specific_job()
    res = _get_all_history_of_specific_job(specific_job)

    return JsonResponse(res)


# Helper functions


def get_all_history_of_specific_data_frame_file(
    specific_dff: Optional[DataFrameFile],
) -> Any:
    if specific_dff is None:
        return None

    # JSON
    serialized = serializers.serialize("json", [specific_dff])
    struct = json.loads(serialized)[0]

    # Upper job
    upper_job = specific_dff.get_upper_job()

    struct["upper_job"] = _get_all_history_of_specific_job(upper_job)

    return struct


def _get_all_history_of_specific_job(specific_job: Optional[Job]) -> Any:
    if specific_job is None:
        return None

    # JSON
    serialized = serializers.serialize("json", [specific_job])
    struct = json.loads(serialized)[0]

    # Inputs
    all_inputs = specific_job.get_all_inputs()

    struct["inputs_dict"] = {
        k: _get_all_history_of_specific_data_frame_file_or_iterable(item_or_items)
        for k, item_or_items in all_inputs.items()
    }

    return struct


def _get_all_history_of_specific_data_frame_file_or_iterable(
    item_or_items: DataFrameFile | Iterable[DataFrameFile] | None,
) -> Any:
    if item_or_items is None:
        return None

    if isinstance(item_or_items, DataFrameFile):
        return get_all_history_of_specific_data_frame_file(item_or_items)

    return [get_all_history_of_specific_data_frame_file(item) for item in item_or_items]
