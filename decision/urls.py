from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    # Models-Decision
    path("decisionfeatures/", views.tree, name="features-dec"),
    # Peformance
    path("decisiontree/", views.decision_tree, name="tree-dec"),
    path("confusiondecision/", views.confusion_decision, name="confusion-dec"),
    path("crossvalidate/", views.cross_validate, name="cross_validate-dec"),
    # Comparison
    path("confusioncmp/", views.confusion_cmp, name="conf-cmp"),
    path("overfittingcmp/", views.overfitting_cmp, name="overfitting-cmp"),
    path("perfanlyticscmp/", views.perf_analytics_cmp, name="perf-cmp"),
]
