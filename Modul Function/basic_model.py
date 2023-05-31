from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, log_loss

class simple_model_classification:
    """
    This class used for model classification without hyperparameter tuning,
    this class just simple model.
    """
    def __init__(self):
        pass
   
    def Knn_model(self, X_train, y_train, n_neighbors):
        # create KNN model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        # fit the model on training data
        model.fit(X_train, y_train)
        # make predictions on training data
        y_pred = model.predict(X_train)
        # evaluate model
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        roc_auc = roc_auc_score(y_train, y_pred)
        results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                        'recall': recall_score(y_train, y_pred),
                        'precision': precision_score(y_train, y_pred),
                        'f1_score': f1_score(y_train, y_pred),
                        'roc_auc' :roc_auc,
                        'log_loss' :log_loss(y_train, y_pred),
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp': tp}
        return results_dict
  
    def decision_tree(self, X_train, y_train):
        # create decision model
        model = DecisionTreeClassifier()
        # fit the model on training data
        model.fit(X_train, y_train)
        # make predictions on training data
        y_pred = model.predict(X_train)
        # evaluate model
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        roc_auc = roc_auc_score(y_train, y_pred)
        results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                        'recall': recall_score(y_train, y_pred),
                        'precision': precision_score(y_train, y_pred),
                        'f1_score': f1_score(y_train, y_pred),
                        'roc_auc' :roc_auc,
                        'log_loss' :log_loss(y_train, y_pred),
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp': tp}
        return results_dict
 
    def svm(self, X_train, y_train):
            # create decision model
            model = SVC()
            # fit the model on training data
            model.fit(X_train, y_train)
            # make predictions on training data
            y_pred = model.predict(X_train)
            # evaluate model
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
            roc_auc = roc_auc_score(y_train, y_pred)
            results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                            'recall': recall_score(y_train, y_pred),
                            'precision': precision_score(y_train, y_pred),
                            'f1_score': f1_score(y_train, y_pred),
                            'roc_auc' :roc_auc,
                            'log_loss' :log_loss(y_train, y_pred),
                            'fn': fn,
                            'fp': fp,
                            'tn': tn,
                            'tp': tp}
            return results_dict
     
    def bayesian(self, X_train, y_train):
        # create decision model
        model = GaussianNB()
        # fit the model on training data
        model.fit(X_train, y_train)
        # make predictions on training data
        y_pred = model.predict(X_train)
        # evaluate model
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        roc_auc = roc_auc_score(y_train, y_pred)
        results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                        'recall': recall_score(y_train, y_pred),
                        'precision': precision_score(y_train, y_pred),
                        'f1_score': f1_score(y_train, y_pred),
                        'roc_auc' :roc_auc,
                        'log_loss' :log_loss(y_train, y_pred),
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp': tp}
        return results_dict
        
    def RF_model(self, X_train, y_train):
        # create Random Forest model
        model = RandomForestClassifier(random_state=42)
        # fit the model on training data
        model.fit(X_train, y_train)
        # make predictions on training data
        y_pred = model.predict(X_train)
        # evaluate model
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        roc_auc = roc_auc_score(y_train, y_pred)
        results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                        'recall': recall_score(y_train, y_pred),
                        'precision': precision_score(y_train, y_pred),
                        'f1_score': f1_score(y_train, y_pred),
                        'roc_auc' :roc_auc,
                        'log_loss' :log_loss(y_train, y_pred),
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp': tp}
        return results_dict     
    
    def Adabost(self, X_train, y_train):
        # Make base estimator
        base_estimator = DecisionTreeClassifier()
        # create Adabost model
        model = model = AdaBoostClassifier(base_estimator=base_estimator)
        # fit the model on training data
        model.fit(X_train, y_train)
        # make predictions on training data
        y_pred = model.predict(X_train)
        # evaluate model
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        roc_auc = roc_auc_score(y_train, y_pred)
        results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                        'recall': recall_score(y_train, y_pred),
                        'precision': precision_score(y_train, y_pred),
                        'f1_score': f1_score(y_train, y_pred),
                        'roc_auc' :roc_auc,
                        'log_loss' :log_loss(y_train, y_pred),
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp': tp}
        return results_dict     
    
    def Xgbost(self, X_train, y_train):
        # create Adabost model
        model = XGBClassifier()
        # fit the model on training data
        model.fit(X_train, y_train)
        # make predictions on training data
        y_pred = model.predict(X_train)
        # evaluate model
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        roc_auc = roc_auc_score(y_train, y_pred)
        results_dict = {'accuracy': accuracy_score(y_train, y_pred),
                        'recall': recall_score(y_train, y_pred),
                        'precision': precision_score(y_train, y_pred),
                        'f1_score': f1_score(y_train, y_pred),
                        'roc_auc' :roc_auc,
                        'log_loss' :log_loss(y_train, y_pred),
                        'fn': fn,
                        'fp': fp,
                        'tn': tn,
                        'tp': tp}
        return results_dict     
    
    import numpy as np