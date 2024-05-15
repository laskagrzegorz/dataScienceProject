import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =======================================================================================
# Testing performance of best model's of each class (KNN, MLP, Decision Tree) and
# plotting the confusion matrices for each model
# =======================================================================================

# -------------------------------------------------------------------------------
# Load test dataset
# -------------------------------------------------------------------------------

X_test = pd.read_csv('../../1_data/derived/X_test_normalised.csv').to_numpy()
y_test = pd.read_csv('../../1_data/derived/y_test.csv').to_numpy().ravel()

# -------------------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------------------

# Load the Decision Tree model
with open("../training_and_hyperparameter_tuning/Decision_Tree_classifier/best_model_decision_tree.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

# Load the KNN model
with open("../training_and_hyperparameter_tuning/KNN_classifier/best_model_knn.pkl", "rb") as f:
    knn_model = pickle.load(f)

# Load the MLP model
with open("../training_and_hyperparameter_tuning/MLP_classifier/best_model_MLP.pkl", "rb") as f:
    mlp_model = pickle.load(f)

# -------------------------------------------------------------------------------
# Make predictions
# -------------------------------------------------------------------------------

# Make predictions for Decision Tree model
decision_tree_predictions = decision_tree_model.predict(X_test)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)

# Make predictions for KNN model
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Make predictions for MLP model
mlp_predictions = mlp_model.predict(X_test)
# Convert MLP predictions to labels
mlp_labels = np.argmax(mlp_predictions, axis=1)
mlp_accuracy = accuracy_score(y_test, mlp_labels)

# -------------------------------------------------------------------------------
# Save test accuracy scores to csv
# -------------------------------------------------------------------------------

# Define accuracy scores
accuracy_scores = {
    "Model": ["Decision Tree", "KNN", "MLP"],
    "Accuracy": [decision_tree_accuracy, knn_accuracy, mlp_accuracy]
}

# Create a DataFrame from accuracy_scores
accuracy_scores_df = pd.DataFrame(accuracy_scores)

# Round accuracy scores to 4 decimal places
accuracy_scores_df["Accuracy"] = accuracy_scores_df["Accuracy"].round(4)

# Save DataFrame to CSV file
accuracy_scores_df.to_csv("test_accuracy_scores.csv", index=False)

# -------------------------------------------------------------------------------
# Calculate Confusion Matrices
# -------------------------------------------------------------------------------

# Calculate confusion matrices
decision_tree_cm = confusion_matrix(y_test, decision_tree_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)
mlp_cm = confusion_matrix(y_test, mlp_labels)

# Convert confusion matrices to DataFrames for better visualization
decision_tree_cm_df = pd.DataFrame(decision_tree_cm, index=range(7), columns=range(7))
knn_cm_df = pd.DataFrame(knn_cm, index=range(7), columns=range(7))
mlp_cm_df = pd.DataFrame(mlp_cm, index=range(7), columns=range(7))

# Normalize confusion matrices to show percentages
decision_tree_cm_norm = decision_tree_cm.astype('float') / decision_tree_cm.sum(axis=1)[:, np.newaxis]
knn_cm_norm = knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis]
mlp_cm_norm = mlp_cm.astype('float') / mlp_cm.sum(axis=1)[:, np.newaxis]

# -------------------------------------------------------------------------------
# Plot Confusion Matrices
# -------------------------------------------------------------------------------

# Read label_encoder object
with open('../../1_data/derived/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define class names using the classes_ attribute of LabelEncoder
class_names = label_encoder.classes_

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plot Decision Tree Confusion Matrix
sns.heatmap(decision_tree_cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=axs[0, 0])
axs[0, 0].set_title("Decision Tree Classifier", fontweight = "bold", fontsize = 16)
axs[0, 0].set_xticklabels(class_names, rotation=25, ha='right', color='red')
axs[0, 0].set_yticklabels(class_names, rotation=25, ha='right', color='green')

# Plot KNN Confusion Matrix
sns.heatmap(knn_cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=axs[0, 1])
axs[0, 1].set_title("KNN Classifier", fontweight = "bold", fontsize = 16)
axs[0, 1].set_xticklabels(class_names, rotation=25, ha='right', color='red')
axs[0, 1].set_yticklabels(class_names, rotation=25, ha='right', color='green')

# Plot MLP Confusion Matrix
sns.heatmap(mlp_cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=axs[1, 0])
axs[1, 0].set_title("MLP Classifier", fontweight = "bold", fontsize = 16)
axs[1, 0].set_xticklabels(class_names, rotation=25, ha='right', color='red')
axs[1, 0].set_yticklabels(class_names, rotation=25, ha='right', color='green')

# Hide the last subplot
axs[1, 1].axis('off')

plt.tight_layout()

# Save the plot as an image file with high resolution (300 dpi)
plt.savefig("confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.show()
