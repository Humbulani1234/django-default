

from django.db import models
from django.urls import reverse

class Item(models.Model, object):

    menu = models.CharField(max_length=20)

    def __str__(self):        
        return f"{self.__class__.__name__} for search selections"

    def get_absolute_url(self):
    
        if self.menu == "logistic":
            return reverse('features-log')
        if self.menu == "decision":
            return reverse('features-dec')
        if self.menu == "streamlit":
            return reverse('streamlit')
        if self.menu == "github-pd":
            return reverse('github_django_pd')
        if self.menu == "github-streamlit":
            return reverse('github_django_streamlit')
        if self.menu == "streamlit":
            return reverse('streamlit')
        if self.menu == "real-analysis":
            return reverse('real_analysis')

