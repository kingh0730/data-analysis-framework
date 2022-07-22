from django.urls import path


from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("streamlit/", views.streamlit, name="streamlit"),
    # Tests
    path(
        "test/all_related_names",
        views.test_all_related_names,
        name="test_all_related_names",
    ),
    path(
        "test/data_frame_files/<int:dff_id>",
        views.test_data_frame_file,
        name="test_data_frame_file",
    ),
    path(
        "test/jobs/<int:job_id>",
        views.test_job,
        name="test_job",
    ),
    path(
        "test/specific_data_frame_file/<int:dff_id>",
        views.test_specific_data_frame_file,
        name="test_specific_data_frame_file",
    ),
    path(
        "test/specific_job/<int:job_id>",
        views.test_specific_job,
        name="test_specific_job",
    ),
    path(
        "test/get_upper_job_of_data_frame_file/<int:dff_id>",
        views.test_get_upper_job_of_data_frame_file,
        name="test_get_upper_job_of_data_frame_file",
    ),
    path(
        "test/get_all_inputs_of_job/<int:job_id>",
        views.test_get_all_inputs_of_job,
        name="test_get_all_inputs_of_job",
    ),
    path(
        "test/get_all_history_of_data_frame_file/<int:dff_id>",
        views.test_get_all_history_of_data_frame_file,
        name="test_get_all_history_of_data_frame_file",
    ),
    path(
        "test/get_all_history_of_job/<int:job_id>",
        views.test_get_all_history_of_job,
        name="test_get_all_history_of_job",
    ),
]
