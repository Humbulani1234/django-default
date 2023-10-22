
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

sys.path.append('/home/humbulani/django/django_ref/refactored_pd')

import data
from .forms import Inputs
from .models import LogFeatures
from .models import Probability

#-------------------------------------------------------------------Defined variables----------------------------------------------------

def image_generator(f):

    buffer = io.BytesIO()
    f.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return image_base64

#------------------------------------------------------------------ Performance measures-------------------------------------------------

def roc(request):

    f = data.c.roc_curve_analytics(data.m.x_test_glm, data.m.y_test_glm)[1]
    image_base64 = image_generator(f)                 

    return render (request, 'logistic/peformance/log_peformance_roc.html', {'image_base64':image_base64})

def confusion_logistic(request):

    f = data.m.confusion_matrix_plot(data.m.x_test_glm, data.m.y_test_glm)
    image_base64 = image_generator(f)

    return render (request, 'logistic/peformance/log_peformance_confusion.html', {'image_base64':image_base64})

#-------------------------------------------------------------------Model Diagnostics-----------------------------------------------------

def normal_plot(request):

    f = data.k.plot_normality_quantile(data.m.x_test_glm)
    image_base64 = image_generator(f)

    return render (request, 'logistic/diagnostics/normal_plot.html', {'image_base64':image_base64})

def residuals(request):

    f = data.b.plot_quantile_residuals(data.m.x_test_glm)
    image_base64 = image_generator(f)

    return render (request, 'logistic/diagnostics/residuals.html', {'image_base64':image_base64})

def partial(request):

    f = data.h.partial_plots_quantile(data.ind_var, data.m.x_test_glm)
    image_base64 = image_generator(f)

    return render (request, 'logistic/diagnostics/partial.html', {'image_base64':image_base64})

def student(request):

    f = data.i.plot_lev_stud_quantile(data.m.x_test_glm)
    image_base64 = image_generator(f)

    return render (request, 'logistic/diagnostics/student_residuals.html', {'image_base64':image_base64})

def cooks(request):

    f = data.j.plot_cooks_dis_quantile(data.m.x_test_glm)

    cache_key = 'cooks_plot'
    cached_result = cache.get(cache_key)

    if cached_result is not None:

        render (request, 'logistic/diagnostics/cooks.html', {'image_base64':cached_result})

    image_base64 = image_generator(f)

    cache.set(cache_key, image_base64, 3600)

    return render (request, 'logistic/diagnostics/cooks.html', {'image_base64':image_base64})

# -------------------------------------------------------------------General Views-----------------------------------------------------

# testing 

from django.db import models
from django import forms
from django.apps import AppConfig
# l = []
# print(sys.path)
# from django_ref import settings
from logistic import views
from django_ref import urls
# print(type(l))
from django.urls import path, include, resolve
# print(dir(Probability))
from django.db import models
# print(help(Probability))
# help(request.session)
from django.contrib.auth.models import User
# print(__builtins__)
# print(locals())
print(User.objects.__dict__)

# help(User.objects)
# print(type(Probability.objects))

import dis
# print(dis.dis(Probability.objects.all))
# print((Probability.probability))

# print(Probability.__init__.__code__)

# print(dir())
# print(locals())
# print(globals()["Probability"])
# print(__builtins__)

def home(request):

    from django.contrib.auth.models import User
    from django.contrib.auth.forms import UserCreationForm

    print(type(User))

    # print(dir(request.session))
    # x = Probability.objects.all()
    # print(type(x))
    # print(x[0].default)
    # help(request.session)
    # print(locals())


    # help(path)

    # print(dir(render(request, 'logistic/general/home_page.html')))
    # print(dir(settings))
    # print(request.session._get_session_key())
    print(request.user)
    # print(type(user))
    # print(request.COOKIES)

    return render(request, 'logistic/general/home_page.html')
    
def about(request):

    return render(request, 'logistic/general/about_page.html')


@login_required
def github_django_pd(request):

    external_url = "https://github.com/Humbulani1234/Django_Anyway/"
    return redirect(external_url)

def inputs(request):

    answer = ""
    if request.method == 'POST':
        form = Inputs(request.POST)
        print(form.__dict__['data'])
        # p = {{ form|crispy }}
        # print(p)
        # print(dir(request))
        if form.is_valid():
            with transaction.atomic():
                instance = form.save()
                print(instance.pk)
                saved_pk = instance.pk

            """ Float features """
            
            NAME = form.cleaned_data.get("NAME")
            AGE = form.cleaned_data.get("AGE")
            CHILDREN = form.cleaned_data.get("CHILDREN")
            PERS_H = form.cleaned_data.get("PERS_H")
            TMADD = form.cleaned_data.get("TMADD")
            TMJOB1 = form.cleaned_data.get("TMJOB1")
            TEL = form.cleaned_data.get("TEL")
            NMBLOAN = form.cleaned_data.get("NMBLOAN")
            FINLOAN = form.cleaned_data.get("FINLOAN")
            INCOME = form.cleaned_data.get("INCOME")
            EC_CARD = form.cleaned_data.get("EC_CARD")
            INC = form.cleaned_data.get("INC")
            INC1 = form.cleaned_data.get("INC1")
            BUREAU = form.cleaned_data.get("BUREAU")
            LOCATION = form.cleaned_data.get("LOCATION")
            LOANS = form.cleaned_data.get("LOANS")
            REGN = form.cleaned_data.get("REGN")
            DIV = form.cleaned_data.get("DIV")
            CASH = form.cleaned_data.get("CASH")
                        
            """ Categorical features """
            
            TITLE = form.cleaned_data.get("TITLE")
            H = 0

            if TITLE == 'H':
                H=1    
            else:
                H=0
            
            STATUS = form.cleaned_data.get("STATUS")
            V, U, G, E, T = 0,0,0,0,0    

            if STATUS == 'V':
                V=1
            elif STATUS == 'U':
                U=1
            elif STATUS == 'G':
                G=1
            elif STATUS == 'E':
                E=1
            elif STATUS=='T':
                T=1
            else:
                V, U, G, E, T = 0,0,0,0,0  

            PRODUCT = form.cleaned_data.get("PRODUCT") 
            Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0    

            if PRODUCT=='Furniture_Carpet':
                Furniture_Carpet=1
            elif PRODUCT=='Dept_Store_Mail':
                Dept_Store_Mail=1
            elif PRODUCT=='Leisure':
                Leisure=1
            elif PRODUCT=='Cars':
                Cars=1
            elif PRODUCT=='OT':
                OT=1
            else:
                Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0   

            RESID = form.cleaned_data.get("RESID")
            Lease = 0    

            if RESID=='Lease':
                Lease=1    
            else:
                Lease=0

            NAT = form.cleaned_data.get("NAT")
            German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0    

            if NAT=='German':
                German=1
            elif NAT=='Turkish':
                Turkish=1        
            elif NAT=='RS':
                RS=1
            elif NAT=='Greek':
                Greek=1
            elif NAT=='Italian':
                Italian=1
            elif NAT=='Other_European':
                Other_European=1
            elif NAT=='Spanish_Portugue':
                Spanish_Portugue=1
            else:
                German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0 

            PROF = form.cleaned_data.get("PROF")  
            Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0    

            if PROF=='Others':
                Others=1
            elif PROF=='Civil_Service_M':
                Civil_Service_M=1
            elif PROF=='Self_employed_pe':
                Self_employed_pe=1
            elif PROF=='Food_Building_Ca':
                Food_Building_Ca=1
            elif PROF=='Chemical_Industr':
                Chemical_Industr=1
            elif PROF=='Pensioner':
                Pensioner=1
            elif PROF=='Sea_Vojage_Gast':
                Sea_Vojage_Gast=1
            elif PROF=='Military_Service':
                Military_Service=1
            else:
                Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
                ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0 

            CAR = form.cleaned_data.get("CAR")   
            Car,Car_and_Motor_bi= 0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Car,Car_and_Motor_bi= 0,0 

            CARDS = form.cleaned_data.get("CARDS") 
            print(CARDS)
            Cheque_card, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0
 
            if CARDS=='Cheque_card':
                no_credit_cards=1
            elif CARDS=='Mastercard_Euroc':
                Mastercard_Euroc=1
            elif CARDS == 'VISA_mybank':
                VISA_mybank=1
            elif CARDS=='VISA_Others':
                VISA_Others=1
            elif CARDS=='Other_credit_car':
                Other_credit_car=1
            elif CARDS=='American_Express':
                American_Express=1
            else:
                Cheque_card, Mastercard_Euroc, VISA_mybank,VISA_Others\
                ,Other_credit_car, American_Express = 0,0,0,0,0,0  

            inputs1 = [H, E, G, T, U, V, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Lease, German, Greek, 
            Italian, Other_European, RS, Spanish_Portugue, Turkish, Chemical_Industr, Civil_Service_M, 
            Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, Car, 
            Car_and_Motor_bi, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others, VISA_mybank]            
            inputs2 = [ 1, CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, 
                        LOCATION, LOANS, REGN, DIV, CASH ]    
            list_ = inputs2 + inputs1
            inputs = np.array(list_).reshape(1,-1)
            answer = np.array(data.m.glm_sample_prob_pred(data.sample, inputs.reshape(1,-1)))
            answer = "{: .10f}".format(answer[0])
            try:
                with transaction.atomic():
                    log_features_object = LogFeatures.objects.get(pk=saved_pk)
                    probability_instance = Probability(CUSTOMER_ID=log_features_object)
                    # print(type(probability_instance.probability))
                    probability_instance.probability = answer # <OnTrue> if <Condition> else <OnFalse>
                    probability_instance.default = 'default'
                    probability_instance.save()
            except LogFeatures.DoesNotExist:
                print('Model does not exixt')
            x = JsonResponse({"probability": answer})
            print(x.content)
            return JsonResponse({"probability": answer})
    else:
        form = Inputs()

    return render(request, 'logistic/model/log_features.html', {'form':form, 'answer':answer})

