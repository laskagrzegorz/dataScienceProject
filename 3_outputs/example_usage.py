import pickle
import numpy as np

# =======================================================================================
# Example use of the trained models (important to scale the new data before passing it
# to the model.
# =======================================================================================

# -------------------------------------------------------------------------------
# Load label encoder, scaler and model of choosing
# -------------------------------------------------------------------------------


# Read label_encoder object
with open('../../1_data/derived/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Read scaler object
with open('../1_data/derived/scaler_values.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the Decision Tree model
with open("../3_outputs/training_and_hyperparameter_tuning/Decision_Tree_classifier/best_model_decision_tree.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

# -------------------------------------------------------------------------------
# Example of usage
# -------------------------------------------------------------------------------

# Example input
example_input = np.random.normal(loc=0, scale=1, size=(1, 9))

# Transform input before putting it through the model
standardised_example_input = scaler.fit(example_input)

# Pass it through the model to get the encoded prediction
encoded_predicted_genre = decision_tree_model.predict(standardised_example_input)

# Invert the encoding to get genre classification
predicted_genre = label_encoder.inverse_transform(encoded_predicted_genre)

print(predicted_genre)

