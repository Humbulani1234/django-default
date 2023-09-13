
from django.contrib import admin
from django.urls import path, include

from . import views


urlpatterns = [

    # General

    path('',views.home, name='home_page'),
    path('about',views.about, name='about'),

    # Models-Logistic

    path('features/',views.inputs, name='features-log'),

    # Peformance

    path('roc',views.roc, name='roc-log'),
    path('confusion',views.confusion_logistic, name='confusion-log'),

    # Diagnostics

    path('residuals',views.residuals, name='residuals'),
    path('student',views.student, name='student_residuals'),
    path('normal',views.normal_plot, name='normal_plot'),
    path('partial',views.partial, name='partial_plot'),
    path('cooks',views.cooks, name='cooks_plot'),
]