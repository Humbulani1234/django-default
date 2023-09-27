

from django.urls import path
from . import views

urlpatterns = [

    # General

    path('',views.home, name='home_page'),
    path('about/',views.about, name='about'),

    # Models-Logistic

    path('features/',views.inputs, name='features-log'),
    path('github-django-pd/',views.github_django_pd, name='github_django_pd'),

    # Peformance

    path('roc/',views.roc, name='roc-log'),
    path('confusion/',views.confusion_logistic, name='confusion-log'),

    # Diagnostics

    path('residuals/',views.residuals, name='residuals'),
    path('student/',views.student, name='student_residuals'),
    path('normal/',views.normal_plot, name='normal_plot'),
    path('partial/',views.partial, name='partial_plot'),
    path('cooks/',views.cooks, name='cooks_plot'),

    # Clustering risk

    path('elbow/',views.elbow_plot, name='elbow-plot'),
    path('clustering-pd/',views.probability_cluster, name='probability_risk_cluster'),
]