
import unittest
import pandas as pd
import numpy as np
import sys
import re
import logging

sys.path.append('refactored_pd/')

import data
from class_modelperf import ModelPerfomance

diagnostics_logger = logging.getLogger("class_logistic_unittest")
diagnostics_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt="{levelname}:{name}:{message}", style="{"))
diagnostics_logger.addHandler(console_handler)
diagnostics_logger.info("LOGISTIC REGRESSION UNIT TESTS")

class TestDataframe(unittest.TestCase):

    def __str__(self):

        """  This executed when calling print on this object """
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in TestDataframe.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def test_no_missing_values(self):

        """ Here we are testing if our Imputation methods worked form the Imputation class """

        dataframe = data.imputer_cat
        self.assertFalse(dataframe.isnull().values.any(), "Dataframe contains missing values")

    def test_zeros_and_ones_values(self):

        """ Here we are testing if Onehot Encoding worked from OneHotEncoding class """

        dataframe = data.instance_stats
        self.assertTrue((dataframe.onehot_encoding().applymap(lambda x: x in [0,1])).all().all(),
                         "Dataframe contains values other than zero or one")

class TestProbability(unittest.TestCase, object):

    def __str__(self):

        """  This executed when calling print on this object """
        
        pattern = re.compile(r'^_')
        method_names = []
        for name, func in TestProbability.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def test_no_values_less_than_one(self):

        """ Here we are testing if our Logistic regression does not return nonsensical values,
        e.g. values less than zero"""

        values = np.array(data.m.probability_prediction())
        self.assertFalse((values < 0).any(),"Prediction contains missing values less than 0")

    def test_no_values_greater_than_one(self):


        """ Here we are testing if our Logistic regression does not return nonsensical values,
        e.g. values greater than one"""

        values = np.array(data.m.probability_prediction())
        self.assertFalse((values > 1).any(),"Prediction contains missing values less than 0")

if __name__ == "__main__":
    unittest.main()

