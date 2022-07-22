from django.urls import path


from . import views


urlpatterns = [
    path("test/all_techniques", views.test_all_techniques, name="test_all_techniques"),
]
