import pytest
import numpy as np
import joblib
import pandas as pd
from permutation import selection_permutaion
import pickle
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix,log_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold

# import data
X_train_smote_stand = joblib.load("D:/BOOTCAMP/project/Project Pribadi/ml churn/X_train_smote_stand.csv")
y_train_smote = joblib.load("D:/BOOTCAMP/project/Project Pribadi/ml churn/y_train_smote.csv")
# import model
with open("D:/BOOTCAMP/project/Project Pribadi/ml churn/model_svm.pkl", 'rb') as file:
    model_svm = pickle.load(file)

def test_selection_features():
    select_columns = selection_permutaion(X_train_smote_stand, y_train_smote, model_svm, 14)

    # Get the first and last column names of the selected features
    first_column = select_columns[0]
    last_column = select_columns[-1]

    # check start and end columns
    assert first_column == 'gender_Male' # check start columns
    assert last_column == 'credit_card' # check final columns

selected_feature_names = X_train_smote_stand[['gender_Male', 'country_France', 'Age_young', 'gender_Female',
    'country_Spain', 'country_Germany', 'Age_mature', 'active_member','tenure', 'balance','credit_score',]]

# hyper parameter svm Tree model
def test_svm_model():
            # create decision model
            model = SVC(C = 90, gamma='auto')
            # fit the model on training data
            model.fit(selected_feature_names, y_train_smote)
            # make predictions on training data
            y_pred = model.predict(selected_feature_names)
            # evaluate model
            tn, fp, fn, tp = confusion_matrix(y_train_smote, y_pred).ravel()
            roc_auc = roc_auc_score(y_train_smote, y_pred)
            results_dict = {'accuracy': accuracy_score(y_train_smote, y_pred),
                            'recall': recall_score(y_train_smote, y_pred),
                            'precision': precision_score(y_train_smote, y_pred),
                            'f1_score': f1_score(y_train_smote, y_pred),
                            'roc_auc' :roc_auc,
                            'log_loss' :log_loss(y_train_smote, y_pred),
                            'fn': fn,
                            'fp': fp,
                            'tn': tn,
                            'tp': tp}
            
            # testing metrics

            # Test accuracy within a range
            expected_accuracy_min = 0.8324871001031992
            expected_accuracy_max = 0.8724871001031992
            assert expected_accuracy_min <= results_dict['accuracy'] <= expected_accuracy_max

            # Test recall within a range
            expected_recall_min = 0.7702755417956656
            expected_recall_max = 0.8272755417956656
            assert expected_recall_min <= results_dict['recall'] <= expected_recall_max

            # Test precision within a range
            expected_precision_min = 0.8824825986078886
            expected_precision_max = 0.9174825986078886
            assert expected_precision_min <= results_dict['precision'] <= expected_precision_max

            # Test F1 score within a range
            expected_f1_score_min = 0.8224511196067721,
            expected_f1_score_max = 0.8644511196067721
            assert expected_f1_score_min <= results_dict['f1_score'] <= expected_f1_score_max

            # Test ROC AUC within a range
            expected_roc_auc_min = 0.804871001031992,
            expected_roc_auc_max = 0.8724871001031992,
            assert expected_roc_auc_min <= results_dict['roc_auc'] <= expected_roc_auc_max

            # Test log loss within a range
            expected_log_loss_max = 6.049560204921958
            assert results_dict['log_loss'] <= expected_log_loss_max

def test_cross_validation():
       # CV svm
        skf = StratifiedKFold(n_splits=5)
        cv_scores_svm = cross_val_score(model_svm, selected_feature_names, y_train_smote, cv=skf)
        result_cv = cv_scores_svm.mean()

        # pytest
        expected_min_score = 0.7918556701030928
        expected_max_score = 0.8218556701030928
        # Test the range of the cross-validation scores
        assert expected_min_score <= result_cv <= expected_max_score
