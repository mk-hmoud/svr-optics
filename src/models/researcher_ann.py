import tensorflow as tf
import os
# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras import layers, models, callbacks
import numpy as np

def build_researcher_ann(input_dim=6, num_neurons=50):
    """
    Builds the ANN architecture described in the researcher's paper:
    5 hidden layers, 50 neurons each, BatchNormalization, ReLU activation.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # Layer 1
    model.add(layers.Dense(num_neurons))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # Layers 2-5
    for _ in range(4):
        model.add(layers.Dense(num_neurons))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        
    # Output Layer
    model.add(layers.Dense(1))
    model.add(layers.Activation('relu')) # As seen in their tf_ann.py: return tf.nn.relu(out)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    return model

def train_ann(model, X_train, y_train, X_val, y_val, epochs=2500, batch_size=48):
    """Trains the ANN with early stopping to prevent extreme overfitting."""
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    return history
