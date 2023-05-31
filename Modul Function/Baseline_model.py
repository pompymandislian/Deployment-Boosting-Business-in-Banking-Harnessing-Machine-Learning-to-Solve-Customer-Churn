from sklearn.dummy import DummyRegressor
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Baseline:
    """
    Class for make baseline classification and regression, this is benckmark
    if we make model
    """
    def __init__(self):
        pass
    
    def print_binary_evaluation(self, X_train, X_test, y_train, y_true, strategy):
        """
        This function used for make baseline regression,show accuracy, recall, precision,
        f1-score, roc-auc, fn,fp,tn,tp
        """
        # inisialitaion model
        dummy_clf = DummyClassifier(strategy=strategy)
        dummy_clf.fit(X_train, y_train)
        # predict data
        y_pred = dummy_clf.predict(X_test)
        y_prob = dummy_clf.predict_proba(X_test)[:, 1]
        # make matric
        roc_auc = roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results_dict = {'accuracy': accuracy_score(y_true, y_pred),
                        'recall': recall_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred),
                        'f1_score': f1_score(y_true, y_pred),
                        'roc_auc': roc_auc,
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp' : tp}
        return results_dict

    def baseline_regressor(self, X_train, y_train, X_test, y_test):
        """
        This function used for make baseline regression,
        show from this function is rmse, mae, r2, and mae for evaluation
        """
        # Inisialisasi model baseline
        regressor_cummy = DummyRegressor(strategy='mean')
        regressor_cummy.fit(X_train, y_train)
        # predict data
        y_pred = regressor_cummy.predict(X_test)
        y_prob = regressor_cummy.predict_proba(X_test)[:, 1]
        model_baseline.fit(X_train, y_train)
        y_pred = model_baseline.predict(X_test)

        # R-squared
        r2 = r2_score(y_test, y_pred)
        # RMSE
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        # MSE
        mse = mean_squared_error(y_test, y_pred)
        # MAE
        mae = mean_absolute_error(y_test, y_pred)    
        results_dict = {'R-squared' : r2, 'RMSE' : rmse, 'MSE' : mse, 'MAE' : mae}

        return results_dict 