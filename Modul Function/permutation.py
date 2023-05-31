from sklearn.inspection import permutation_importance
import numpy as np
def selection_permutaion(X_train, y_train, model, select_k):
            """
            This function for selection feature with permutaion method.
            Step :
            - input best model
            - input best X_train and y_train
            - fitting model
            - Permutation data and then sort with argsort from high score until low
            - fill sum of feature
            - return result of feature
            """
            # Fit the data to the model
            model.fit(X_train, y_train)

            # Perform permutation feature importance
            perm_importance = permutation_importance(model, X_train, y_train)

            # Get the feature importances and indices
            feature_importances = perm_importance.importances_mean
            feature_indices = np.argsort(feature_importances)[::-1]

            # Set the number of top features to select
            k = select_k

            # Get the indices of the top k features
            top_k_indices = feature_indices[:k]

            # Get the column names of the top k features
            selected_feature_names = X_train.columns[top_k_indices]

            return selected_feature_names