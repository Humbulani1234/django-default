from django import forms
from django.db.models.fields import BLANK_CHOICE_DASH

from .models import LogFeatures


class Inputs(forms.ModelForm):
    class Meta:
        model = LogFeatures
        fields = [
            "NAME_CUSTOMER",
            "CHILDREN",
            "PERS_H",
            "AGE",
            "TMADD",
            "TMJOB1",
            "TEL",
            "NMBLOAN",
            "FINLOAN",
            "INCOME",
            "EC_CARD",
            "INC",
            "INC1",
            "BUREAU",
            "LOCATION",
            "LOANS",
            "REGN",
            "DIV",
            "CASH",
            "TITLE",
            "STATUS",
            "PRODUCT",
            "RESID",
            "NAT",
            "PROF",
            "CAR",
            "CARDS",
        ]
