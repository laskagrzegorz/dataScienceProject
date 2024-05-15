import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv

# =======================================================================================
# Training of Multilayer Perceptron (MLP) models with different numbers of hidden layers
# and number of neurons in each layer
# =======================================================================================

# Load in the training dataset
X_train = pd.read_csv('../../../1_data/derived/X_train_normalised.csv').to_numpy()
y_train = pd.read_csv('../../../1_data/derived/y_train.csv').to_numpy().ravel()


def train_mlp_models_classifier(X, y, hidden_layers, neurons, learning_rates, patience=20):
    """
    Trains MLP models for classification.

    Parameters:
    - X (array-like): Input features.
    - y (array-like): Target labels.
    - hidden_layers (list): List of integers specifying the number of hidden layers to try.
    - neurons (list): List of integers specifying the number of neurons to try in each hidden layer.
    - learning_rates (list): List of floats specifying the learning rates to try.
    - patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 20.

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
    best_val_loss = float('inf')
    best_config = None

    # Train MLP models for each combination of hidden layers, neurons, and learning rates
    for hidden_layer in hidden_layers:
        for neuron in neurons:
            for learning_rate in learning_rates:
                tf.random.set_seed(44)

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
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                                  restore_best_weights=True)

                # Train the model
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
                                    callbacks=[early_stopping], verbose=0, batch_size=128)

                # Evaluate model on validation set
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=False)

                print(hidden_layer, neuron, np.round(val_loss, 5))

                # Store training history
                training_history[(hidden_layer, neuron, learning_rate)] = history.history

                # Check if current model is the best so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_config = (hidden_layer, neuron, learning_rate)

                print("Finished", learning_rate)

    return training_history, best_model, best_config


# -------------------------------------------------------------------------------
# Perform training and selection
# -------------------------------------------------------------------------------

training_history, best_model, best_config = train_mlp_models_classifier(
    X_train, y_train, hidden_layers=[1, 2, 3], neurons=[32, 64, 128], learning_rates=[0.01, 0.001, 0.0001, 0.00001])

# -------------------------------------------------------------------------------
# Save best results for each combination of # neurons and # hidden layers to csv
# -------------------------------------------------------------------------------


def select_best_learning_rate(training_history):
    """
    Selects the best learning rate for each model based on the last epoch's validation loss and accuracy.

    Parameters:
    - training_history (dict): Dictionary containing training history for each combination of hyperparameters.

    Returns:
    - best_learning_rates (dict): Dictionary containing the best learning rate, validation loss, and accuracy for each model.
    """
    best_learning_rates = {}

    # Iterate through training history
    for config, history in training_history.items():
        # Get the configuration (hidden layer, neuron, learning rate)
        hidden_layer, neuron, learning_rate = config
        # Get the last epoch's validation loss and accuracy
        last_epoch_val_loss = history['val_loss'][-1]
        last_epoch_val_accuracy = history['val_accuracy'][-1]
        # If learning rate already exists for this model, update if the current loss is smaller
        if (hidden_layer, neuron) in best_learning_rates:
            if last_epoch_val_loss < best_learning_rates[(hidden_layer, neuron)]["val_loss"]:
                best_learning_rates[(hidden_layer, neuron)]["learning_rate"] = learning_rate
                best_learning_rates[(hidden_layer, neuron)]["val_loss"] = last_epoch_val_loss
                best_learning_rates[(hidden_layer, neuron)]["val_accuracy"] = last_epoch_val_accuracy
        else:  # If not, create a new entry for this model
            best_learning_rates[(hidden_layer, neuron)] = {
                "learning_rate": learning_rate,
                "val_loss": last_epoch_val_loss,
                "val_accuracy": last_epoch_val_accuracy
            }

    return best_learning_rates


def save_results(best_learning_rates, filename='validation_results.csv'):
    """
    Saves the results to a CSV file.

    Parameters:
    - best_learning_rates (dict): Dictionary containing the best learning rate, validation loss, and accuracy for each model.
    - filename (str, optional): Name of the CSV file to save. Default is 'validation_results.csv'.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['hidden_layer', 'neuron', 'learning_rate', 'val_loss', 'val_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (hidden_layer, neuron), details in best_learning_rates.items():
            val_loss_formatted = '{:.3f}'.format(details['val_loss'])  # Format validation loss to 3 decimal places
            val_accuracy_formatted = '{:.3f}'.format(details['val_accuracy'])
            writer.writerow({
                'hidden_layer': hidden_layer,
                'neuron': neuron,
                'learning_rate': details['learning_rate'],
                'val_loss': val_loss_formatted,
                'val_accuracy': val_accuracy_formatted
            })


best_learning_rates = select_best_learning_rate(training_history)
save_results(best_learning_rates)

# -------------------------------------------------------------------------------
# Make a plot showing training of model with the lowest validation loss
# -------------------------------------------------------------------------------

def plot_training_history(training_history, best_config):
    """
    Plots the training and validation loss over epochs for the best model configuration.

    Parameters:
    - training_history (dict): Dictionary containing training history for each combination of hyperparameters.
    - best_config (tuple): Tuple containing the best configuration of hyperparameters.

    Returns:
    - None
    """
    plt.figure(figsize=(14, 6))

    # Extract training and validation loss from the history for the best configuration
    train_loss = training_history[best_config]['loss']
    val_loss = training_history[best_config]['val_loss']

    # Plot training and validation loss
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')

    # Find the index where the validation loss is minimum
    best_val_loss_idx = val_loss.index(min(val_loss))

    # Draw vertical line at the index of best validation loss
    plt.axvline(x=best_val_loss_idx, color='black', linestyle='--')

    # Annotate minimal validation loss and corresponding epoch
    plt.text(best_val_loss_idx - 1, min(val_loss) * 1.33,
             f'Minimal validation loss: {min(val_loss):.5f} \n Epoch: {best_val_loss_idx}',
             verticalalignment='bottom', horizontalalignment='right', color='black')

    plt.title('Training of MPL consisting of 1 hidden layers with 64 neurons', fontweight="bold")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.grid(color='lightgray')
    plt.gca().spines['top'].set_color('lightgray')
    plt.gca().spines['bottom'].set_color('lightgray')
    plt.gca().spines['left'].set_color('lightgray')
    plt.gca().spines['right'].set_color('lightgray')

    # Save the plot as an image file with high resolution (300 dpi)
    plt.savefig("best_model_training_plot.png", dpi=300, bbox_inches='tight')

    plt.show()


plot_training_history(training_history, best_config)

# -------------------------------------------------------------------------------
# Save best model
# -------------------------------------------------------------------------------

with open("best_model_MLP.pkl", 'wb') as file:
    pickle.dump(best_model, file)
