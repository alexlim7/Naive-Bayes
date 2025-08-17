"""
This module contains the model implementation.
"""

import numpy as np


class NaiveBayes(object):
    """Bernoulli Naive Bayes model
    
    Parameters
    ----------
    n_classes : int
        The number of classes.

    Attributes
    ----------
    n_classes: int
        The number of classes.
    attr_dist: np.ndarray
        2D (n_classes x n_attributes) NumPy array of the attribute distributions
    label_priors: np.nparray
        1D NumPy array of the priors distribution
    """

    def __init__(self, n_classes: int) -> None:
        """
        Constructor for NaiveBayes model with n_classes.
        """
        self.n_classes = n_classes
        self.attr_dist = None
        self.label_priors = None

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Trains the model using maximum likelihood estimation.

        Parameters
        ----------
        X_train: np.ndarray
            a 2D (n_examples x n_attributes) numpy array
        y_train: np.ndarray
            a 1D (n_examples) numpy array

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """
        class_counts = np.bincount(y_train, minlength=self.n_classes)
        self.label_priors = (class_counts + 1) / (X_train.shape[0] + self.n_classes)
        self.attr_dist = np.zeros((self.n_classes, X_train.shape[1]))
        for c in range(self.n_classes):
            X_c = X_train[y_train == c]
            self.attr_dist[c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)
        
        return self.attr_dist, self.label_priors

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Outputs a predicted label for each input in inputs.

        Parameters
        ----------
        inputs: np.ndarray
            a 2D NumPy array containing inputs

        Returns
        -------
        np.ndarray
            a 1D numpy array of predictions
        """
        n_samples = inputs.shape[0]
        log_priors = np.log(self.label_priors)
        log_likelihoods = np.log(self.attr_dist)
        log_inv_likelihoods = np.log(1 - self.attr_dist)
        
        log_probs = np.zeros((n_samples, self.n_classes))
        for c in range(self.n_classes):
            log_probs[:, c] = log_priors[c] + np.sum(
                inputs * log_likelihoods[c] + (1 - inputs) * log_inv_likelihoods[c], axis=1
            )
        
        return np.argmax(log_probs, axis=1)

    def accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Outputs the accuracy of the trained model on a given dataset (data).

        Parameters
        ----------
        X_test: np.ndarray
            a 2D numpy array of examples
        y_test: np.ndarray
            a 1D numpy array of labels

        Returns
        -------
        float
            a float number indicating accuracy (between 0 and 1)
        """
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

    def print_fairness(
        self, X_test: np.ndarray, y_test: np.ndarray, x_sens: np.ndarray
    ) -> np.ndarray:
        """
        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 0 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit.

        Parameters
        ----------
        X_test: np.ndarray
            a 2D numpy array of examples
        y_test: np.ndarray
            a 1D numpy array of labels
        x_sens: np.ndarray
            a numpy array of sensitive attribute values

        Returns
        -------
        np.ndarray
            a 1D numpy array of predictions
        """
        predictions = self.predict(X_test)

        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged classes are
        # grouped together as values of 1 and all privileged classes are given
        # the class 0.
        #
        # Given a dataset with sensitive attribute S (e.g., race, sex, religion,
        # etc.), where S=1 indicates that the person is in a protected group,
        # and binary class to be predicted Y (e.g., "will hire"), we will
        # say that a model has disparate impact if:
        #      P[Y^ = 1 | S = 1] / P[Y^ = 1 | S != 1] <= (t = 0.8).
        #
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens == 1)]) / np.mean(
            predictions[np.where(x_sens == 0)]
        )
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group

        pred_priv = predictions[np.where(x_sens == 0)]
        pred_unpr = predictions[np.where(x_sens == 1)]
        y_priv = y_test[np.where(x_sens == 0)]
        y_unpr = y_test[np.where(x_sens == 1)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1)) / np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1)) / np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0)) / (
            len(y_priv) - np.sum(y_priv)
        )
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0)) / (
            len(y_unpr) - np.sum(y_unpr)
        )

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr
        unpr_fpr = 1 - unpr_tnr

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (
            np.mean(predictions[np.where(x_sens == 0)])
            - np.mean(predictions[np.where(x_sens == 1)])
        )

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(
            predictions[np.where(x_sens == 0)] == y_test[np.where(x_sens == 0)]
        )
        unpriv_accuracy = np.mean(
            predictions[np.where(x_sens == 1)] == y_test[np.where(x_sens == 1)]
        )

        return predictions
