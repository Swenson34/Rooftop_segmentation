# Define model training function

def model_training(Model, X_train, Y_train, X_valid, Y_valid, path_to_save_model, path_to_logs, Batch_size, Epochs):
    Model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define early stops
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=path_to_logs),
                 tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
                 tf.keras.callbacks.ModelCheckpoint(path_to_save_model, verbose=1, save_best_only=True)]

    # Fit the model
    Model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=Batch_size, epochs=Epochs,
              callbacks=callbacks)

    return Model