
from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [

    # Models-Decision

    path('decisionfeatures/',views.tree, name='features-dec'),

    # Peformance

    path('decisiontree/',views.decision_tree, name='tree-dec'),
    path('confusiondecision/',views.confusion_decision, name='confusion-dec'),
    path('crossvalidate/',views.cross_validate, name='cross_validate-dec'),
]