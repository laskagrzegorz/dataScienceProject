from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# =======================================================================================
# Hyper-tuning and training of Decision Tree classifier using 5 fold cross-validation for
# different complexity parameter alpha on log scale.
# =======================================================================================

# Load in the training dataset
X_train = pd.read_csv('../../../1_data/derived/X_train_normalised.csv').to_numpy()
y_train = pd.read_csv('../../../1_data/derived/y_train.csv').to_numpy().ravel()

def tune_decision_tree_ccp_alpha_classifier(X_train_data, y_train_data, ccp_alpha_values, n_splits=5):
    """
    Tunes the 'ccp_alpha' hyperparameter of Decision Tree Classifier using K-Fold Cross-Validation.

    Parameters:
    - X_train_data (array-like): Training features.
    - y_train_data (array-like): Training labels.
    - ccp_alpha_values (array-like): List of values to try for the 'ccp_alpha' hyperparameter.
    - n_splits (int, optional): Number of folds for K-Fold Cross-Validation. Default is 5.

    Returns:
    - best_ccp_alpha (float): Best value found for the 'ccp_alpha' hyperparameter.
    - best_accuracy (float): Accuracy achieved by the best 'ccp_alpha'.
    - accuracy_values (dict): Dictionary containing average accuracy values for each 'ccp_alpha'.
    """

    accuracy_values = {}  # Dictionary to store average accuracy values for each ccp_alpha

    # Initialize K-Fold cross-validator
    kf = KFold(n_splits=n_splits)

    num_comb = len(ccp_alpha_values)
    i = 0
    # Loop through all values of ccp_alpha
    for ccp_alpha in ccp_alpha_values:
        accuracy_list = []  # List to store accuracy values for current ccp_alpha
        # Loop through each fold of K-Fold CV
        for train_index, val_index in kf.split(X_train_data):
            X_train, X_val = X_train_data[train_index], X_train_data[val_index]
            y_train, y_val = y_train_data[train_index], y_train_data[val_index]
            # Initialize the DecisionTreeClassifier with current ccp_alpha
            tree_classifier = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=50)
            # Fit the model to the training data
            tree_classifier.fit(X_train, y_train)
            # Predict on the validation set
            y_pred = tree_classifier.predict(X_val)
            # Calculate accuracy for this fold
            accuracy = accuracy_score(y_val, y_pred)
            accuracy_list.append(accuracy)
        # Calculate average accuracy across all folds for this ccp_alpha
        avg_accuracy = np.mean(accuracy_list)
        # Store the average accuracy value for this ccp_alpha
        accuracy_values[ccp_alpha] = avg_accuracy
        i += 1
        print(f'Done {i}/{num_comb}')

    # Find the best ccp_alpha based on the highest average accuracy
    best_ccp_alpha = max(accuracy_values, key=accuracy_values.get)
    best_accuracy = accuracy_values[best_ccp_alpha]

    print("Best ccp_alpha:", best_ccp_alpha)
    print("Accuracy (after tuning):", best_accuracy)

    return best_ccp_alpha, best_accuracy, accuracy_values


# -------------------------------------------------------------------------------
# Perform cross-validation
# -------------------------------------------------------------------------------

# Define lists of values for hyperparameters
ccp_alpha_values = np.logspace(np.log10(0.000001), np.log10(0.1), 200)

best_ccp_alpha, best_accuracy, accuracy_values = tune_decision_tree_ccp_alpha_classifier(X_train, y_train, ccp_alpha_values, n_splits=5)


def plot_accuracy_values(accuracy_values, best_ccp_alpha, best_accuracy):
    """
    Plots the average accuracy values for different ccp_alpha values.

    Parameters:
    - accuracy_values (dict): Dictionary containing average accuracy values for each ccp_alpha.
    - best_ccp_alpha (float): Best value found for the 'ccp_alpha' hyperparameter.
    - best_accuracy (float): Accuracy achieved by the best 'ccp_alpha'.
    """

    # Extract ccp_alpha values and corresponding accuracy values
    ccp_alphas = list(accuracy_values.keys())
    accuracies = list(accuracy_values.values())

    plt.figure(figsize=(14, 6))
    # Plot accuracy values
    plt.plot(ccp_alphas, accuracies, linestyle='-')

    # Plot vertical line at the best ccp_alpha
    plt.axvline(x=best_ccp_alpha, color='black', linestyle='--')

    # Annotate best ccp_alpha and accuracy
    plt.text(best_ccp_alpha * 1.1, best_accuracy * 0.90,
             r'Best $\alpha: $' + f'{best_ccp_alpha:.2e}\nAccuracy: {best_accuracy:.5f}',
             verticalalignment='bottom', horizontalalignment='left', color='black')

    # Labels and title
    plt.xlabel(r'Complexity parameter $\alpha$ used for Minimal Cost-Complexity Pruning')
    plt.xscale('log')
    plt.ylabel('Average Accuracy')
    plt.title('Cross-Validation Accuracy for different Decision Tree Classifiers', fontweight='bold')

    # Save the plot as an image file with high resolution (300 dpi)
    plt.savefig("Decision_Tree_classifier_hyperparameter_tuning_plot.png", dpi=300, bbox_inches='tight')

    # Show plot
    plt.show()

# -------------------------------------------------------------------------------
# Make a plot showing results of cross-validation
# -------------------------------------------------------------------------------


plot_accuracy_values(accuracy_values, best_ccp_alpha, best_accuracy)

# -------------------------------------------------------------------------------
# Fit and save best model
# -------------------------------------------------------------------------------

# Fit the best model on whole data
best_tree_classifier = DecisionTreeClassifier(ccp_alpha=best_ccp_alpha, random_state=50)
best_tree_classifier.fit(X_train, y_train)

# Save the model
with open("../../../3_tests/best_model_decision_tree.pkl", "wb") as f:
    pickle.dump(best_tree_classifier, f)


