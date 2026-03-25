import os
# Force CPU only to avoid potential GPU memory issues in this environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_ann(input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.1):
    """
    Builds a standard ANN with configurable hidden layers.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for units in hidden_layers:
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
            
    model.add(layers.Dense(1)) # Target is continuous (loss)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def train_ann(model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, patience=50, verbose=1):
    """
    Trains the ANN with early stopping.
    """
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    return history
