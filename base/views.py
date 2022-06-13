from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

# Create your views here.


def home(request: HttpRequest) -> HttpResponse:
    context = {"title": "Home"}
    return render(request, "base/home.html", context)
