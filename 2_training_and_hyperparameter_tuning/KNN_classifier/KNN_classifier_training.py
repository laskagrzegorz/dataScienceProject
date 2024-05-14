import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

# =======================================================================================
# Hyper-tuning and training of KNN classifier using 5 fold cross-validation for odd k's in
# range 1 to 39 and for different  weights and distance functions
# =======================================================================================

# Load in the training dataset
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
                kf = KFold(n_splits=5)

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


# -------------------------------------------------------------------------------
# Perform cross-validation
# -------------------------------------------------------------------------------

# Fit all models
ks = np.arange(1, 40)[::2]
mean_cv_scores_dict = knn_cross_validation(X_train, y_train, ks)


def plot_mean_cv_scores_list(mean_cv_scores_dicts, titles, ks):
    """
    Plots the mean cross-validation scores for different combinations of KNN parameters.

    Parameters:
    - mean_cv_scores_dicts (list of dictionaries): List of dictionaries containing mean cross-validation scores
                                                    for different parameter combinations.
    - titles (list of str): Titles for each subplot.
    - ks (list): List of values for the number of neighbors (k).

    Returns:
    - max_accuracy_p (int): Value of parameter p corresponding to the highest accuracy.
    - max_accuracy_weight (str): Weighting scheme ('uniform' or 'distance') corresponding to the highest accuracy.
    - max_accuracy_k (int): Value of k corresponding to the highest accuracy.
    """

    # Define values for p and weights
    p_s = [1, 2]
    weights = ['uniform', 'distance']

    # Get the number of plots
    num_plots = len(mean_cv_scores_dicts)

    # Create subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=(14, 6 * num_plots))

    # Iterate over each dictionary of mean_cv_scores and its corresponding title
    for i, (mean_cv_scores_dict, title) in enumerate(zip(mean_cv_scores_dicts, titles)):
        # Initialize variables to keep track of maximum accuracy and its corresponding parameters
        max_accuracy = 0
        max_accuracy_k = None
        max_accuracy_weight = None
        max_accuracy_p = None

        # Iterate over weights and p values
        for weight in weights:
            for p in p_s:
                # Extract mean cross-validation scores for the current combination
                mean_cv_scores = mean_cv_scores_dict[(weight, p)]

                # Find the maximum accuracy and its corresponding k value
                max_mean_cv_score = max(mean_cv_scores)
                max_mean_cv_score_k = ks[np.argmax(mean_cv_scores)]

                # Update maximum accuracy and corresponding parameters if applicable
                if max_mean_cv_score > max_accuracy:
                    max_accuracy = max_mean_cv_score
                    max_accuracy_k = max_mean_cv_score_k
                    max_accuracy_weight = weight
                    max_accuracy_p = p

                # Plot mean CV scores for the current combination
                if weight == "distance":
                    axs[i].plot(ks, mean_cv_scores, label=f'Distance-weighted KNN (p={p})')
                else:
                    axs[i].plot(ks, mean_cv_scores, label=f'Uniform KNN (p={p})')

        # Plot vertical line at the k value corresponding to the highest accuracy
        axs[i].axvline(x=max_accuracy_k, color='black', linestyle='--')

        # Mark the highest accuracy point with a black dot
        axs[i].scatter(max_accuracy_k, max_accuracy, color="black")

        # Add text annotation for highest accuracy and its corresponding parameters
        if max_accuracy_weight == "distance":
            axs[i].text(max_accuracy_k - 0.5, max_accuracy * 0.95,
                        f'Highest Accuracy: {max_accuracy:.5f} \nDistance-weighted KNN (p={max_accuracy_p}), K={max_accuracy_k}',
                        verticalalignment='bottom', horizontalalignment='right', color='black')
        else:
            axs[i].text(max_accuracy_k + 0.1, max_accuracy * 0.95,
                        f'Highest Accuracy: {max_accuracy:.5f} \nUniform KNN (p={max_accuracy_p}), K={max_accuracy_k}',
                        verticalalignment='bottom', horizontalalignment='right', color='black')

        # Set title, labels, legend, and grid for the subplot
        axs[i].set_title(title, fontweight="bold")
        axs[i].set_xlabel('Number of Neighbors (k)')
        axs[i].set_ylabel('Average Accuracy')
        axs[i].legend(loc='upper right', title="Parameter")
        axs[i].grid(True)

    plt.tight_layout()

    # Save the plot as an image file with high resolution (300 dpi)
    plt.savefig("KNN_classifier_hyperparameter_tuning_plot.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    # Return the parameters corresponding to the highest accuracy
    return max_accuracy_p, max_accuracy_weight, max_accuracy_k


# -------------------------------------------------------------------------------
# Make a plot showing results of cross-validation
# -------------------------------------------------------------------------------

# Make the hyper-tuning plot
titles = ['Cross-Validation Accuracy for different KNN Classifiers']
max_accuracy_p, max_accuracy_weight, max_accuracy_k = plot_mean_cv_scores_list([mean_cv_scores_dict], titles, ks)


# -------------------------------------------------------------------------------
# Fit and save best model
# -------------------------------------------------------------------------------

# Fit the best model on whole data
best_model_knn = KNeighborsClassifier(n_neighbors=max_accuracy_k, weights=max_accuracy_weight,
                                      algorithm="auto", metric="minkowski", p=max_accuracy_p)
best_model_knn.fit(X_train, y_train)

# Save the model
with open("best_model_knn.pkl", "wb") as f:
    pickle.dump(best_model_knn, f)







