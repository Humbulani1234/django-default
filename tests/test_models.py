
from django.test import TestCase
from .models import LogFeatures

class ModelsTest(TestCase):

    def test_model_logfeatures(self):
        # my_instance = LogFeatures.objects.create(<features>)
        # self.assertEqual(my_instance.<feature>, "feature")