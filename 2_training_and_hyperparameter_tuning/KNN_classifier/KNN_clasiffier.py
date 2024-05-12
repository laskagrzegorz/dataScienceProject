import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load in the dataset
X_train = pd.read_csv('../../1_data/derived/X_train_normalised.csv').to_numpy()
y_train = pd.read_csv('../../1_data/derived/y_train.csv').to_numpy().ravel()


def knn_cross_validation(data_x, data_y, ks=np.arange(1, 40)[::2], ps=[1, 2], weights=['uniform', 'distance']):
    """
    Perform k-Nearest Neighbors cross-validation for classification.

    Parameters:
        data_x (numpy.ndarray): Input features.
        data_y (numpy.ndarray): Target labels.
        ks (array-like, optional): List of numbers of neighbors to consider. Defaults to np.arange(1, 40)[::2].
        ps (list, optional): List of Minkowski distances parameter values. Defaults to [1, 2].
        weights (list, optional): List of weight functions used in prediction. Defaults to ['uniform', 'distance'].

    Returns:
        dict: Dictionary containing mean cross-validation scores for each combination of parameters.
    """
    # Dictionary to store mean cross-validation scores for each combination
    mean_cv_scores_dict = {}

    # Loop over each combination of weight function and Minkowski distance parameter
    for weight in weights:
        for p in ps:
            mean_cv_scores = []  # List to store mean cross-validation scores for the current combination

            # Loop over each number of neighbors
            for parameter in ks:
                # Initialize KNN classifier model with the current combination of parameters
                KNN = KNeighborsClassifier(n_neighbors=parameter, weights=weight, algorithm="auto", metric="minkowski",
                                           p=p)

                # Perform k-fold Cross-Validation
                kf = KFold(n_splits=2)  # Split data into 2 folds

                # List to store cross-validation scores
                cv_scores = []

                # Iterate over each fold and calculate cross-validation score
                for train_index, val_index in kf.split(data_x):
                    X_train, X_val = data_x[train_index], data_x[val_index]
                    y_train, y_val = data_y[train_index], data_y[val_index]

                    # Fit KNN classifier model
                    KNN.fit(X_train, y_train)

                    # Predict on the validation set
                    y_pred = KNN.predict(X_val)

                    # Calculate accuracy
                    accuracy = accuracy_score(y_val, y_pred)

                    # Append accuracy to the list of cross-validation scores
                    cv_scores.append(accuracy)

                # Calculate mean cross-validation score
                mean_cv_score = np.mean(cv_scores)
                mean_cv_scores.append(mean_cv_score)

                print(parameter)

                # Store the mean cross-validation scores for the current combination in the dictionary
            mean_cv_scores_dict[(weight, p)] = mean_cv_scores

            print(f'Finished: {weight} with distance {p}')

    return mean_cv_scores_dict


mean_cv_scores_dict = knn_cross_validation(X_train, y_train)





