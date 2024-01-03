"""
    ===================================
    MODEL ALTERNATIVES - Decision Tree
    ===================================

    We investigate Decision Trees as a model alternative to GLM - Binomial

    ==========
    Base Tree
    ==========

    ============================
    Fit a base tree and Prune it
    ============================
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from typing import Type

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning_pd
from class_missing_values import ImputationCat

decision_logger = logging.getLogger("class_decision")
decision_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(fmt="{levelname}:{name}:{message}", style="{")
)
decision_logger.addHandler(console_handler)
decision_logger.info("MODEL DECISION TREE IS INCLUDED")

# ---------------------------------------------------Class DecisionTree-------------------------------------------


class DecisionTree(OneHotEncoding, object):

    """Fit a base and assess it using complexity pruning to determine a properly fit tree -
    free of overfitting. This object is created in such a way that it has to behave accordingly to the supplied
    dataframes during initialisation process, meaning the methods (mostly) will not require dataframes user inputs when
    called on the object"""

    def __init__(
        self,
        custom_rcParams,
        df_nomiss_cat,
        type_,
        df_loan_float,
        target,
        grid_search: Type[GridSearchCV],
        randomstate,
        onehot,
        threshold=None,
    ):
        OneHotEncoding.__init__(
            self, custom_rcParams, df_nomiss_cat, type_, randomstate, onehot
        )
        self.randomstate_dt = self.random_state_one
        self.threshold_dt = threshold
        self.df_loan_float_dt = df_loan_float
        self.target_dt = target
        self.x_train_dt = super(DecisionTree, self).train_val_test(
            self.df_loan_float_dt, self.target_dt
        )[0]
        self.y_train_dt = super(DecisionTree, self).train_val_test(
            self.df_loan_float_dt, self.target_dt
        )[1]
        self.x_val_dt = super(DecisionTree, self).train_val_test(
            self.df_loan_float_dt, self.target_dt
        )[2]
        self.y_val_dt = super(DecisionTree, self).train_val_test(
            self.df_loan_float_dt, self.target_dt
        )[3]
        self.x_test_dt = super(DecisionTree, self).train_val_test(
            self.df_loan_float_dt, self.target_dt
        )[4]
        self.y_test_dt = super(DecisionTree, self).train_val_test(
            self.df_loan_float_dt, self.target_dt
        )[5]

    def __str__(self):
        pattern = re.compile(r"^_")
        method_names = []
        for name, func in DecisionTree.__dict__.items():
            if not pattern.match(name) and callable(func):
                method_names.append(name)
        return f"This is class: {self.__class__.__name__}, and this is the public interface: {method_names}"

    def dt_grid_search(self):
        """The following uses GridSearchCV to find the best l2 penalty regularization parameter
        GridSearchCV is being implemented as arm to the LogisticRegression class"""

        param_grid = {
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        print(best_estimator)
        return best_estimator

    def dt_classification_fit(self, ccpalpha):
        """DT Classification fit - Fit a base tree without pruning, likely overfit"""

        clf_dt = DecisionTreeClassifier(
            random_state=self.randomstate_dt, ccp_alpha=ccpalpha
        )
        clf_dt = clf_dt.fit(self.x_train_dt, self.y_train_dt)
        return clf_dt

    def _dt_predict_prob_positive(self, x_test, ccpalpha):
        """Logistic Regression model prediction values - returns probabilities for the positive class"""

        pos_prob = self.dt_classification_fit(ccpalpha).predict_proba(x_test)[:, 1]
        return pos_prob

    def dt_overfitting(self, x_train, y_train, x_test, y_test, ccpalpha, *thresholds):
        """Calculation of recall, precision, accuracy and F1 score based on user supplied thresholds"""

        train_precisions = []
        train_recalls = []
        test_precisions = []
        test_recalls = []
        for threshold in thresholds:
            y_prob_train = self._dt_predict_prob_positive(x_train, ccpalpha)
            y_pred_train = (y_prob_train > threshold).astype(int)

            y_prob_test = self._dt_predict_prob_positive(x_test, ccpalpha)
            y_pred_test = (y_prob_test > threshold).astype(int)

            train_precision = precision_score(y_train, y_pred_train)
            train_recall = recall_score(y_train, y_pred_train)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)

            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)

        plt.close("all")
        self.fig, (self.axs1, self.axs2) = plt.subplots(1, 2)
        self.axs1.plot(thresholds, train_precisions, label="Training Precision")
        self.axs1.plot(thresholds, test_precisions, label="Testing Precision")
        self.axs1.set_label("Threshold")
        self.axs1.set_label("Precision")
        self.axs1.legend()
        self.axs2.plot(thresholds, train_recalls, label="Training Recalls")
        self.axs2.plot(thresholds, test_recalls, label="Testing Recalls")
        self.axs2.set_label("Threshold")
        self.axs2.set_label("Precision")
        self.axs2.legend()
        return self.fig, train_precisions, train_recalls

    def dt_feature_importance(self, ccpalpha):
        """Determination of feature importance"""

        feature_importances = self.dt_classification_fit(ccpalpha).feature_importances_
        feature_names = self.x_train_dt.columns
        feature_importance_dict = dict(zip(feature_names, feature_importances))
        sorted_feature_importance = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        self.fig, self.axs = plt.subplots(1, 1)
        self.axs.barh(*zip(*sorted_feature_importance))
        self.axs.set_label("Feature Importance")
        self.axs.set_label("Feature Name")
        self.axs.set_title("Feature Importance in Decision Tree")
        return self.fig

    def dt_sample_class_pred(self, sample, input_sample, ccpalpha):
        """Classification prediction - It classifies a customer based on a threshold: whether they are
        postive or negative"""

        pred_prob = self.self._dt_predict_prob_positive(input_sample, ccpalpha)
        if self.threshold_dt is None:
            raise ValueError("Provide the threshold value in the class constructor")
        else:
            if pred_prob[sample] >= self.threshold_dt:  # assign threshold manually
                return 1
            else:
                return 0

    def dt_sample_prob_pred(self, sample, input_sample, ccpalpha):
        """Probability prediction - It returns the probability that is greater than the threshold:
        The postive class or else it returns the negative class probability"""

        pred_prob = (
            self.dt_classification_fit(ccpalpha).predict_proba(input_sample).round(10)
        )
        if self.threshold_dt is None:
            raise ValueError("Provide the threshold value in the class constructor")
        if pred_prob[sample][1] >= self.threshold_dt:
            return pred_prob[sample][1]
        else:
            return pred_prob[sample][0]

    def dt_optimal_threshold(self, x_test, y_test, ccpalpha):
        fpr, tpr, thresholds = roc_curve(
            y_test, self._dt_predict_prob_positive(x_test, ccpalpha)
        )
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thres = thresholds[optimal_idx]
        return optimal_thres

    def dt_binary_prediction(self, x_test, ccpalpha, threshold=None):
        """Binary prediction function threshold or user supplied threshold"""

        if threshold is None:
            print("running with the default 0.5 threshold")
            pred_bin = self.dt_classification_fit(ccpalpha).predict(x_test)
            return pred_bin
        else:
            predict_binary = self._dt_predict_prob_positive(x_test, ccpalpha)
            for i in range(x_test.shape[0]):
                if predict_binary[i] >= threshold:
                    predict_binary[i] = 1
                else:
                    predict_binary[i] = 0
            return predict_binary

    def dt_perf_analytics(self, x_test, y_test, ccpalpha, threshold=None):
        """Roc curve analytics and plot - Rocs points represent confusion matrices at varying
        thresholds"""

        def generate(data, axs, title_):
            sns.set_theme(style="ticks", color_codes=True)
            sns.barplot(x="Metric", y="Value", data=data, ax=axs)
            axs.spines["top"].set_visible(False)
            axs.spines["right"].set_visible(False)
            axs.set_title(title_)
            for p in axs.patches:
                axs.annotate(
                    f"{p.get_height().round(1)}",
                    (
                        p.get_x().round(2) + p.get_width().round(2) / 2.0,
                        p.get_height().round(2),
                    ),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

        if threshold:
            plt.close("all")
            self.fig, self.axs = plt.subplots(1, 1)
            y_bin = self.dt_binary_prediction(x_test, ccpalpha, threshold)
            accuracy = accuracy_score(y_test, y_bin)
            # scores = cross_val_score(clf_dt, self.x_train_dt, self.y_train_dt, cv=5)
            f1 = f1_score(y_test, y_bin)
            auc = roc_auc_score(y_test, y_bin)
            data = pd.DataFrame(
                {
                    "Metric": ["threshold", "accuracy", "f1", "auc"],
                    "Value": [threshold, accuracy, f1, auc],
                }
            )
            generate(data, self.axs, "DT")
            return self.fig, threshold, accuracy, f1, auc
        else:
            print("running with the default 0.5 threshold")
            plt.close("all")
            self.fig, self.axs = plt.subplots(1, 1)
            y_bin = self.dt_binary_prediction(x_test, ccpalpha)
            threshold = 0.5
            accuracy = accuracy_score(y_test, y_bin)
            f1 = f1_score(y_test, y_bin)
            auc = roc_auc_score(y_test, y_bin)
            data = pd.DataFrame(
                {
                    "Metric": ["threshold", "accuracy", "f1", "auc"],
                    "Value": [threshold, accuracy, f1, auc],
                }
            )
            generate(data, self.axs, "DT")
            return self.fig, threshold, accuracy, f1, auc

    def dt_confusion_matrix_plot(self, x_test, y_test, ccpalpha, threshold=None):
        """Base tree Confusion matrix"""

        def conf_plot():
            conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")
            conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
            conf_matrix_plot.ax_.set_xlabel("Predicted Label", fontsize=14)
            conf_matrix_plot.ax_.set_ylabel("True Label", fontsize=14)

        plt.close("all")
        self.fig, self.axs = plt.subplots(1, 1)
        pred_bin = pd.Series(self.dt_binary_prediction(x_test, ccpalpha, threshold))
        conf_matrix = confusion_matrix(y_test, pred_bin)
        conf_matrix_plot = ConfusionMatrixDisplay(
            conf_matrix, display_labels=["No Default", "Yes Default"]
        )
        conf_plot()
        return self.fig, conf_matrix

    def plot_dt(self, ccpalpha):
        """Tree plot visualisation"""

        self.fig, self.axs = plt.subplots(1, 1)
        clf_dt = self.dt_classification_fit(ccpalpha)
        plot_tree(
            clf_dt,
            filled=True,
            rounded=True,
            feature_names=self.x_train_dt.columns.tolist(),
            ax=self.axs,
        )
        return self.fig

    def pruning(self, ccpalpha):
        """Extracting alphas for pruning"""

        clf_dt = self.dt_classification_fit(ccpalpha)
        path = clf_dt.cost_complexity_pruning_path(self.x_train_dt, self.y_train_dt)
        ccp_alphas = path.ccp_alphas
        ccp_alphas = ccp_alphas[:-1]
        return ccp_alphas

    def cross_validate_alphas(self, ccpalpha):
        """Cross validation for best alpha cross valiadtion fixed at 5 samples"""

        self.fig, self.axs = plt.subplots(1, 1)
        alpha_loop_values = []
        ccp_alphas = self.pruning(ccpalpha)
        for ccp_alpha in ccp_alphas:
            clf_dt = DecisionTreeClassifier(
                random_state=self.randomstate_dt, ccp_alpha=ccp_alpha
            )
            scores = cross_val_score(clf_dt, self.x_train_dt, self.y_train_dt, cv=5)
            alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
        alpha_results = pd.DataFrame(
            alpha_loop_values, columns=["alpha", "mean_accuracy", "std"]
        )
        alpha_results.plot(
            ax=self.axs,
            x="alpha",
            y="mean_accuracy",
            yerr="std",
            marker="o",
            linestyle="--",
        )
        self.axs.spines["top"].set_visible(False)
        self.axs.spines["right"].set_visible(False)
        return alpha_results, self.fig

    def ideal_alpha(self, ccpalpha, threshold_1, threshold_2):
        """Extraction of ideal alpha threshold1 and threshold2 should be supplied after (visual inspection)
        of the alpha results figure to interpolate the best alpha"""

        alpha_results = self.cross_validate_alphas(ccpalpha)[0]
        ideal_ccp_alpha = alpha_results[
            (alpha_results["alpha"] > threshold_1)
            & (alpha_results["alpha"] < threshold_2)
        ]["alpha"]
        ideal_ccp_alpha = ideal_ccp_alpha.values.tolist()
        return ideal_ccp_alpha[0]

    """ Here we perform similar analysis but on a pruned tree: confusion matrices, metrics, classification,
    feature importance and plotting the tree """

    def dt_pruned_alpha(self, ccpalpha, threshold_1, threshold_2):
        """Ideal alpha value for pruning the tree"""

        ideal_ccp_alpha = self.ideal_alpha(ccpalpha, threshold_1, threshold_2)
        return ideal_ccp_alpha

    def dt_pruned_fit(self, ccpalpha, threshold_1, threshold_2):
        """Pruned tree fitting"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_clf_dt = self.ideal_alpha(ideal_ccp_alpha)
        return pruned_clf_dt

    def dt_sample_pruned_class(
        self, ccpalpha, threshold_1, threshold_2, sample, input_sample
    ):
        """Confusion matrix plot"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_classifier = self.dt_sample_class_pred(
            sample, input_sample, ideal_ccp_alpha
        )
        return pruned_classifier

    def dt_sample_pruned_prob(
        self, ccpalpha, threshold_1, threshold_2, sample, input_sample
    ):
        """Prediction and perfomance analytics"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_predict_dt = self.dt_sample_prob_pred(
            sample, input_sample, ideal_ccp_alpha
        )
        return pruned_predict_dt

    def dt_pruned_conf_matrix(
        self, ccpalpha, threshold_1, threshold_2, x_test, y_test, threshold=None
    ):
        """Confusion matrix plot"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_confusion_matrix = self.dt_confusion_matrix_plot(
            x_test, y_test, ideal_ccp_alpha, threshold
        )
        return pruned_confusion_matrix

    def dt_pruned_perf_analytics(
        self, ccpalpha, threshold_1, threshold_2, x_test, y_test, threshold=None
    ):
        """Perfomance metrics plot"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_performance_metrics = self.dt_perf_analytics(
            x_test, y_test, ideal_ccp_alpha, threshold
        )
        return pruned_performance_metrics

    def dt_pruned_feature_imp(self, ccpalpha, threshold_1, threshold_2):
        """Confusion matrix plot"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_feature_importance = self.dt_feature_importance(ideal_ccp_alpha)
        return pruned_feature_importance

    def dt_pruned_overfitting(
        self,
        ccpalpha,
        threshold_1,
        threshold_2,
        x_train,
        y_train,
        x_test,
        y_test,
        *thresholds,
    ):
        """Confusion matrix plot"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_overfitting = self.dt_overfitting(
            x_train, y_train, x_test, y_test, ideal_ccp_alpha, *thresholds
        )
        return pruned_overfitting

    def dt_pruned_tree(self, ccpalpha, threshold_1, threshold_2):
        """Plot final tree"""

        ideal_ccp_alpha = self.dt_pruned_alpha(ccpalpha, threshold_1, threshold_2)
        pruned_plot_tree = self.plot_dt(ideal_ccp_alpha)
        return pruned_plot_tree
