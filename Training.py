# Define model training function
import tensorflow as tf

def model_training(model, x_train, y_train, x_valid, y_valid, path_to_save_model, path_to_logs, batch_size_, epochs_):
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Define early stops
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir = path_to_logs),
                 tf.keras.callbacks.EarlyStopping(patience = 5, monitor = 'val_loss'),
                 tf.keras.callbacks.ModelCheckpoint(path_to_save_model, verbose=1, save_best_only = True)]

    # Fit the model
    model.fit(x_train, y_train, validation_data = (x_valid, y_valid), batch_size = batch_size_, epochs = epochs_,
              callbacks=callbacks)

    return model