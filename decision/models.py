from django.db import models
from django.db.models.fields import BLANK_CHOICE_DASH
from django.utils import timezone


class DecFeatures(models.Model):

    """Floating features"""

    NAME_CUSTOMER = models.CharField(max_length=100)
    AGE = models.FloatField()
    CHILDREN = models.FloatField()
    PERS_H = models.FloatField()
    TMADD = models.FloatField()
    TMJOB1 = models.FloatField()
    TEL = models.FloatField()
    NMBLOAN = models.FloatField()
    FINLOAN = models.FloatField()
    INCOME = models.FloatField()
    EC_CARD = models.FloatField()
    INC = models.FloatField()
    INC1 = models.FloatField()
    BUREAU = models.FloatField()
    LOCATION = models.FloatField()
    LOANS = models.FloatField()
    REGN = models.FloatField()
    DIV = models.FloatField()
    CASH = models.FloatField()

    """ Categorical features """

    GENRE_CHOICES_1 = BLANK_CHOICE_DASH + [("H", "H"), ("R", "R")]
    GENRE_CHOICES_2 = BLANK_CHOICE_DASH + [
        ("V", "V"),
        ("U", "U"),
        ("G", "G"),
        ("E", "E"),
        ("T", "T"),
        ("W", "W"),
    ]
    GENRE_CHOICES_3 = BLANK_CHOICE_DASH + [
        ("Radio_TV_Hifi", "Radio_TV_Hifi"),
        ("Furniture_Carpet", "Furniture_Carpet"),
        ("Dept_Store_Mail", "Dept_Store_Mail"),
        ("Leisure", "Leisure"),
        ("Cars", "Cars"),
        ("OT", "OT"),
    ]
    GENRE_CHOICES_4 = BLANK_CHOICE_DASH + [("Lease", "Lease"), ("Owner", "Owner")]
    GENRE_CHOICES_5 = BLANK_CHOICE_DASH + [
        ("German", "German"),
        ("Turkish", "Turkish"),
        ("RS", "RS"),
        ("Greek", "Greek"),
        ("Yugoslav", "Yugoslav"),
        ("Italian", "Italian"),
        ("Other_European", "Other_European"),
        ("Spanish_Portugue", "Spanish_Portugue"),
    ]
    GENRE_CHOICES_6 = BLANK_CHOICE_DASH + [
        ("Others", "Others"),
        ("Civil_Service_M", "Civil_Service_M"),
        ("Self_employed_pe", "Self_employed_pe"),
        ("Food_Building_Ca", "Food_Building_Ca"),
        ("Chemical_Industr", "Chemical_Industr"),
        ("Pensioner", "Pensioner"),
        ("Sea_Vojage_Gast", "Sea_Vojage_Gast"),
        ("State_Steel_Ind,", "State_Steel_Ind,"),
        ("Military_Service", "Military_Service"),
    ]
    GENRE_CHOICES_7 = BLANK_CHOICE_DASH + [
        ("Car", "Car"),
        ("Without_Vehicle", "Without_Vehicle"),
        ("Car_and_Motor_bi", "Car_and_Motor_bi"),
    ]
    GENRE_CHOICES_8 = BLANK_CHOICE_DASH + [
        ("Cheque_card", "Cheque_card"),
        ("no_credit_cards", "no_credit_cards"),
        ("Mastercard_Euroc", "Mastercard_Euroc"),
        ("VISA_mybank", "VISA_mybank"),
        ("VISA_Others", "VISA_Others"),
        ("Other_credit_car", "Other_credit_car"),
        ("American_Express", "American_Express"),
    ]

    TITLE = models.CharField(max_length=20, choices=GENRE_CHOICES_1)
    STATUS = models.CharField(max_length=20, choices=GENRE_CHOICES_2)
    PRODUCT = models.CharField(max_length=20, choices=GENRE_CHOICES_3)
    RESID = models.CharField(max_length=20, choices=GENRE_CHOICES_4)
    NAT = models.CharField(max_length=20, choices=GENRE_CHOICES_5)
    PROF = models.CharField(max_length=20, choices=GENRE_CHOICES_6)
    CAR = models.CharField(max_length=20, choices=GENRE_CHOICES_7)
    CARDS = models.CharField(max_length=20, choices=GENRE_CHOICES_8)
    CUSTOMER_ID = models.AutoField(primary_key=True)

    def __str__(self):
        return f"{self.__class__.__name__} inputs to calculate the probability"


class DecProbability(models.Model):
    CUSTOMER_ID = models.OneToOneField(
        DecFeatures,
        on_delete=models.CASCADE,
        related_name="dec_probability",
        to_field="CUSTOMER_ID",
    )
    probability = models.CharField(max_length=20, null=True)
    default = models.CharField(max_length=20, null=True)
    PROBABILITY_ID = models.AutoField(primary_key=True)
    DATE = models.DateTimeField(default=timezone.now)
    PROBABILITY_ID = models.AutoField(primary_key=True)

    def __str__(self):
        return f"{self.__class__.__name__} of a customer defaulting"
