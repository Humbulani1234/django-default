
from django.contrib import admin

from .models import DecFeatures
from .models import DecProbability

admin.site.register(DecFeatures)
admin.site.register(DecProbability)
