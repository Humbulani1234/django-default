
import pandas as pd
import numpy as np
import pickle
import sys
import io
import base64
import logging

sys.path.append('/home/humbulani/django-pd/django_ref/refactored_pd')

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.db import transaction
from django.core.cache import cache
from django.contrib.auth.decorators import login_required

from .models import DecFeatures, DecProbability
from .forms import Inputs

from class_decision_tree import DecisionTree
from class_missing_values import ImputationCat
from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
import data

#-------------------------------------------------------------------Defined variables----------------------------------------------------

def image_generator(f):

    buffer = io.BytesIO()
    f.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64

#-------------------------------------------------------------------Perfomance----------------------------------------------

def confusion_decision(request):

    f = data.d.dt_pruned_conf_matrix(data.ccpalpha, data.threshold_1, data.threshold_2,
                                    data.d.x_test_dt, data.d.y_test_dt)
    cache_key = 'confusion_plot'
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        render (request, 'decision/peformance/confusion_decision.html', {'image_base64':cached_result})
    image_base64 = image_generator(f)
    cache.set(cache_key, image_base64, 3600)
    image_base64 = image_generator(f)                 
    return render (request, 'decision/peformance/confusion_decision.html', {'image_base64':image_base64})

def decision_tree(request):

    f = data.d.dt_pruned_tree(data.ccpalpha, data.threshold_1, data.threshold_2)
    cache_key = 'decision_plot'
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        render (request, 'decision/peformance/decision_tree.html', {'image_base64':cached_result})
    image_base64 = image_generator(f)
    cache.set(cache_key, image_base64, 3600)
    image_base64 = image_generator(f)  
    return render (request, 'decision/peformance/decision_tree.html', {'image_base64':image_base64}) 

def cross_validate(request):

    f = data.d.cross_validate_alphas(data.ccpalpha)[1]
    cache_key = 'cross_validate_plot'
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        render (request, 'decision/peformance/cross_validate.html', {'image_base64':cached_result})
    image_base64 = image_generator(f)
    cache.set(cache_key, image_base64, 3600)
    image_base64 = image_generator(f)   
    return render (request, 'decision/peformance/cross_validate.html', {'image_base64':image_base64})

#-------------------------------------------------------------------Comparison----------------------------------------------

def confusion_cmp(request):

    f = data.o.cmp_confusion_matrix_plot(data.ccpalpha, data.threshold_1, data.threshold_2, data.threshold)
    cache_key = 'confusion_cmp_plot'
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        render (request, 'decision/comparison/confusion_cmp.html', {'image_base64':cached_result})
    image_base64 = image_generator(f)
    cache.set(cache_key, image_base64, 3600)
    image_base64 = image_generator(f)                 
    return render (request, 'decision/comparison/confusion_cmp.html', {'image_base64':image_base64})

def overfitting_cmp(request):

    f = data.o.cmp_overfitting(data.ccpalpha, data.threshold_1, data.threshold_2, *np.arange(0.1, 0.9, 0.05))
    cache_key = 'overfitting_plot'
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        render (request, 'decision/comparison/overfitting.html', {'image_base64':cached_result})
    image_base64 = image_generator(f)
    cache.set(cache_key, image_base64, 3600)
    image_base64 = image_generator(f)  
    return render (request, 'decision/comparison/overfitting.html', {'image_base64':image_base64}) 

def perf_analytics_cmp(request):

    f = data.o.cmp_performance_metrics(data.ccpalpha, data.threshold_1, data.threshold_2, data.threshold)
    cache_key = 'perf_analytics_plot'
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        render (request, 'decision/comparison/perf_analytics.html', {'image_base64':cached_result})
    image_base64 = image_generator(f)
    cache.set(cache_key, image_base64, 3600)
    image_base64 = image_generator(f)   
    return render (request, 'decision/comparison/perf_analytics.html', {'image_base64':image_base64})

#------------------------------------------------------------Calculation------------------------------------------

# @login_required
def tree(request):

    answer = ""
    if request.method == 'POST':
        form = Inputs(request.POST)
        if form.is_valid():
            with transaction.atomic():
                instance = form.save()
                saved_pk = instance.pk

            # Float features
            
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
                        
            # Categorical features
            
            TITLE = form.cleaned_data.get("TITLE")
            R,H = 0,0

            if TITLE == 'H':
                H=1
            else:
                R=0
           
            STATUS = form.cleaned_data.get("STATUS")
            W,V, U, G, E, T = 0,0,0,0,0,0    

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
                W = 0 

            PRODUCT = form.cleaned_data.get("PRODUCT") 
            Radio_TV_Hifi, Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0,0    

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
                Radio_TV_Hifi = 0   

            RESID = form.cleaned_data.get("RESID")
            Owner,Lease = 0,0    

            if RESID=='Lease':
                Lease=1    
            else:
                Owner=0

            NAT = form.cleaned_data.get("NAT")
            Yugoslav,German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0,0    

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
                Yugoslav = 1 

            PROF = form.cleaned_data.get("PROF")  
            State_Steel_Ind,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0,0    

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
                State_Steel_Ind = 1 

            CAR = form.cleaned_data.get("CAR")   
            Without_Vehicle,Car,Car_and_Motor_bi= 0,0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Without_Vehicle= 1    

            Cheque_card,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0,0  
            CARDS = form.cleaned_data.get("CARDS")  

            if CARDS=='no_credit_cards':
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
                Cheque_card = 1  

            inputs1 = [H, R, E, G, T, U, V, W, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Radio_TV_Hifi, Lease, Owner,  
                      German, Greek, Italian, Other_European, RS, Spanish_Portugue, Turkish, Yugoslav, Chemical_Industr,
                      Civil_Service_M, Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe,
                      State_Steel_Ind, Car, Car_and_Motor_bi, Without_Vehicle, American_Express, Cheque_card, Mastercard_Euroc,
                      Other_credit_car, VISA_Others, VISA_mybank, no_credit_cards]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS,
                       REGN, DIV, CASH]    

            list_ = inputs2 + inputs1
            inputs = np.array([list_]).reshape(1,-1)           
            answer = data.d.dt_sample_pruned_prob(data.ccpalpha, data.threshold_1, data.threshold_2,
                                                  data.sample, inputs)
            try:
                with transaction.atomic():
                    dec_features_object = DecFeatures.objects.get(pk=saved_pk)
                    probability_instance = DecProbability(CUSTOMER_ID=dec_features_object)
                    probability_instance.probability = answer
                    probability_instance.default = 'default' if answer > 0.47 else 'nodefault'
                    probability_instance.save()
            except DecFeatures.DoesNotExist:
                print('Model doesnt not exixt')
            return JsonResponse({"probability": answer})
    else:
        form = Inputs()
    return render(request, 'decision/model/decision.html', {'form':form, 'answer':answer})