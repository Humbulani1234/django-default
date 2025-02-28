from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home_page"),
    path("about/", views.about, name="about"),
    path("github-django-pd/", views.github_django_pd, name="github_django_pd"),
    path(
        "github-django-streamlit/",
        views.github_django_streamlit,
        name="github_django_streamlit",
    ),
    path("real-analysis/", views.real_analysis, name="real_analysis"),
    path("streamlit/", views.streamlit, name="streamlit"),
]
