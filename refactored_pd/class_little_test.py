"""
     =====================================
     Little Test Algorithm presented below
     =====================================
     =======
     Inputs
     =======
     values = non-missing columns
     dataframe = dataframe with suitable missing column 
"""


class LittleTest:
    def __init__(self):
        pass

    def __str__(self):
        pattern = re.compile(r"^_")
        method_names = []
        for name, func in LittleTest.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)

        return f"This is Class {self.__class__.__name__} with methods {method_names}"

    def _integers_convert(x):
        """Converts a binary-string (list) to a corresponding integer string"""

        strings = [str(integer) for integer in x]
        a_string = "".join(strings)
        an_integer = int(a_string)

        return an_integer

    def _binary_to_Decimal(binary):
        """Calculate a decimal from binary"""

        binary1 = binary
        decimal, i, n = 0, 0, 0
        while binary != 0:
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary // 10
            i += 1

        return decimal

    def _convert_pattern_to_deci(dataframe, values):
        """Converts each missing values pattern to a decimal number"""

        df_float = dataframe.drop(labels=values, axis=1)
        df_pattern = 1 * df_float.isnull()
        l = []
        for index, row in df_pattern.iterrows():
            d = integers_convert(x=row.tolist())
            l.append(binary_to_Decimal(binary=d))

        return l

    def _pattern_value_counts(dataframe, values):
        """Counts how many sample/index points for each pattern"""

        df_float = dataframe.drop(labels=values, axis=1)
        df_pattern = 1 * dataframe.isnull()
        c = df_pattern["l"].value_counts()

        return c.to_dict()

    def _dict_pattern_observed_vars(dataframe, values):
        df_float = dataframe.drop(labels=values, axis=1)
        df_pattern = 1 * df_students_missing_float.isnull()
        l = Convert_Pattern_to_Decimal(dataframe, values)
        df_pattern["l"] = l
        t = []
        for index, row in df_pattern.iterrows():
            attr = []
            for i in range(len(row)):
                if row[i] == 0:
                    attr.append(row.index.values[i])
                else:
                    pass
            t.append(attr)
        dictionary = dict(zip(l, t))
        for value in dictionary.values():
            if "l" in value:
                value.remove("l")

        return dictionary

    def observed_vars_imputed_mean_cov(dataframe, values):
        """Imputed means and Covariance.

        Parameters
        ==========
        X_test: Dataframe
                Dataframe for testing model perfomance

        ==========
        Returns

        predict_binary: Series
                        Predicted series of 0 and 1"""

        df_float = dataframe.drop(labels=values, axis=1)
        df_array = df_float.to_numpy()
        df_impute = impy.em(df_array)
        y = df_impute
        df_float_impute = pd.DataFrame(y, columns=df_float.columns)
        dictionary = Dictionary_Pattern_Observed_Variables(dataframe, values)
        mean_observed_imputed = []
        for value in dictionary.values():
            mean_value = df_float_impute[value].mean()
            mean_observed_imputed.append(mean_value)
        covariance_observed_imputed = []
        for value in dictionary.values():
            covariance_value = df_float_impute[value].cov()
            covariance_observed_imputed.append(covariance_value)

        return mean_observed_imputed, covariance_observed_imputed

    def observed_vars_mean_cov(dataframe, values):
        """Observed values Mean and Covariance calculations"""

        df_float = dataframe.drop(labels=values, axis=1)
        df_pattern = 1 * df_float.isnull()
        l = Convert_Pattern_to_Decimal(dataframe, values)
        df_pattern["l"] = l
        a = df_pattern.groupby(df_pattern["l"])
        dictionary = Dictionary_Pattern_Observed_Variables(dataframe, values)
        list_ = []
        for pattern in dictionary:
            b = a.get_group(pattern)
            index = b.index
            b_values = df_float.iloc[index].values
            b = b.drop(labels="l", axis=1)
            df_b = pd.DataFrame(b_values, columns=b.columns)
            df_b.dropna(axis=1, inplace=True)
            list_.append(df_b)
        mean_observed = []
        covariance_observed = []
        for element in list_:
            mean_observed.append(element.mean())
            covariance_observed.append(element.cov())

        return mean_observed, covariance_observed

    def little_test(dataframe, values):
        try:
            df_float = dataframe.drop(labels=values, axis=1)
            df_pattern = 1 * df_float.isnull()
            x = Convert_Pattern_to_Decimal(dataframe, values)
            dictionary = Dictionary_Pattern_Observed_Variables(dataframe, values)
            df_pattern["x"] = x
            h = df_pattern["x"].value_counts()
            k = h.index.tolist()
            Little_Test = []
            for i in range(len(k)):
                first_component = (
                    np.array(Observed_Variables_Mean_Cov(dataframe, values)[0][i])
                    - np.array(
                        Observed_Variables_imputed_mean_cov(dataframe, values)[0][i]
                    )
                )[:, np.newaxis].T
                second_component = np.linalg.inv(
                    np.array(Observed_Variables_Mean_Cov(dataframe, values)[1][i])
                )
                third_component = np.array(
                    Observed_Variables_Mean_Cov(dataframe, values)[0][i]
                ) - np.array(
                    Observed_Variables_imputed_mean_cov(dataframe, values)[0][i]
                )
                Little_Test.append(
                    k[i]
                    * (
                        np.dot(
                            np.dot(first_component, second_component), third_component
                        )
                    )
                )

            return sum(Little_Test)

        except:
            print("An exception occurred, Singular matrix")
