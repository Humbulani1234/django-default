
from django.test import TestCase
from django.urls import reverse

import json

class ViewsTests(TestCase, object):

    def test_logistic_input_prob_range(self):

        data1 = {"H":1, E, G, T, U, V, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Lease, German, Greek, 
        Italian, Other_European, RS, Spanish_Portugue, Turkish, Chemical_Industr, Civil_Service_M, 
        Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, Car, 
        Car_and_Motor_bi, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others, VISA_mybank,
        1, CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, 
        LOCATION, LOANS, REGN, DIV, CASH }   

        url = reverse("features")
        response = self.client.get(url)
        # response = self.client.post(url, data1)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content)
        prob = content["probability"]
        self.assertTrue(0 <= prob <= 1)

    def test_decision_input_prob_range(self):

        data1 = {"H":1, E, G, T, U, V, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Lease, German, Greek, 
        Italian, Other_European, RS, Spanish_Portugue, Turkish, Chemical_Industr, Civil_Service_M, 
        Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, Car, 
        Car_and_Motor_bi, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others, VISA_mybank,
        1, CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, 
        LOCATION, LOANS, REGN, DIV, CASH }   

        url = reverse("features")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content)
        prob = content["probability"]
        self.assertTrue(0 <= prob <= 1)

    def test_search_output(self):

        data1 = {"H":1, E, G, T, U, V, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Lease, German, Greek, 
        Italian, Other_European, RS, Spanish_Portugue, Turkish, Chemical_Industr, Civil_Service_M, 
        Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, Car, 
        Car_and_Motor_bi, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others, VISA_mybank,
        1, CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, 
        LOCATION, LOANS, REGN, DIV, CASH }   

        url = reverse("features")
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        content = json.loads(response.content)
        prob = content["probability"]
        self.assertTrue(0 <= prob <= 1)