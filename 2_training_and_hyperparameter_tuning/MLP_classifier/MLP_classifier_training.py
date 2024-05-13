import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# Load in the training dataset
X_train = pd.read_csv('../../1_data/derived/X_train_normalised.csv').to_numpy()
y_train = pd.read_csv('../../1_data/derived/y_train.csv').to_numpy().ravel()


def train_mlp_models_classifier(X, y, hidden_layers, neurons, learning_rates, patience=20):
    """
    Trains MLP models for classification.

    Parameters:
    - X (array-like): Input features.
    - y (array-like): Target labels.
    - hidden_layers (list): List of integers specifying the number of hidden layers to try.
    - neurons (list): List of integers specifying the number of neurons to try in each hidden layer.
    - learning_rates (list): List of floats specifying the learning rates to try.
    - patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 10.

    Returns:
    - training_history (dict): Dictionary containing training history for each combination of hyperparameters.
    - best_model (tf.keras.Model): Trained model with the best performance on the validation set.
    - best_config (tuple): Tuple containing the best configuration of hyperparameters (hidden_layers, neurons, learning_rate).
    """

    # Split data into train and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=50)

    # Check the number of unique labels in y_train
    num_unique_labels = len(np.unique(y_train))
    if num_unique_labels == 2:
        output_activation = 'sigmoid'  # Binary classification
        loss_function = 'binary_crossentropy'
    else:
        output_activation = 'softmax'  # Multiclass classification
        loss_function = 'sparse_categorical_crossentropy'

    # Dictionary to store training history for each combination of hidden layer and number of neurons
    training_history = {}

    # Best model variables
    best_model = None
    best_val_accuracy = 0
    best_config = None

    # Train MLP models for each combination of hidden layers, neurons, and learning rates
    for hidden_layer in hidden_layers:
        for neuron in neurons:
            for learning_rate in learning_rates:
                tf.random.set_seed(32)

                # Define the model architecture
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Input(shape=X_train.shape[1:]))
                for _ in range(hidden_layer):
                    model.add(tf.keras.layers.Dense(neuron, activation='relu'))
                model.add(tf.keras.layers.Dense(num_unique_labels, activation=output_activation))

                # Compile the model
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

                # Define early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                                                  restore_best_weights=True)

                # Train the model
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
                                    callbacks=[early_stopping], verbose=1, batch_size=128)

                # Evaluate model on validation set
                _, val_accuracy = model.evaluate(X_val, y_val, verbose=False)

                print(hidden_layer, neuron, np.round(val_accuracy, 5))

                # Store training history
                training_history[(hidden_layer, neuron, learning_rate)] = history.history

                # Check if current model is the best so far
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model = model
                    best_config = (hidden_layer, neuron, learning_rate)

                print("Finished", learning_rate)

    return training_history, best_model, best_config


training_history, best_model, best_config = train_mlp_models_classifier(X_train, y_train, hidden_layers=[1, 2, 3],
                                                                        neurons=[32, 64, 128],
                                                                        learning_rates=[0.01, 0.001, 0.0001, 0.00001])



