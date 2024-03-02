import pandas as pd
import numpy as np
import sys
import pickle
import types
import io
import base64
import statsmodels.api as sm

from django.shortcuts import render, redirect
from django.contrib import messages
from pathlib import Path
from django.template import loader
from django.http import HttpResponse
from django.http import JsonResponse
from django.core.cache import cache
from django.db import transaction
from django.contrib.auth.decorators import login_required

from .models import Item


def home(request):
    query = request.GET.get("q")
    print(query)
    if query is not None:
        results = Item.objects.filter(menu__icontains=query)[:10] if query else []
        suggestions = [
            {"menu": result.menu, "url": result.get_absolute_url()}
            for result in results
        ]
        print(suggestions)
        return JsonResponse({"results": suggestions})
    else:
        return render(request, "search/home_page.html")


def about(request):
    return render(request, "search/about_page.html")


def github_django_streamlit(request):
    external_url = "https://github.com/Humbulani1234/Streamlit/"

    return redirect(external_url)


def real_analysis(request):
    external_url = "https://gitfront.io/r/Humbulani/21772ff35fd550f95e800f6a616cd08ba6b9183b/Real-Analysis-and-Measure-theory/"

    return redirect(external_url)


def streamlit(request):
    external_url = "https://classdeployedpy-bjw4uzjdm5ezhumufi2t2e.streamlit.app/"

    return redirect(external_url)


@login_required
def github_django_pd(request):
    external_url = "https://github.com/Humbulani1234/Django_Anyway/"

    return redirect(external_url)
