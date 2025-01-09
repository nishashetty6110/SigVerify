import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report
import numpy as np
import os

def create_signature_cnn_model():
    """Create a Convolutional Neural Network for signature classification."""
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Second convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Third convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Fourth convolutional block
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        # Output layer (binary classification)
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Train the signature classification model."""
    IMG_SIZE = 150
    BATCH_SIZE = 32
    EPOCHS = 50  # Increased epochs for better training
    DATA_DIR = 'data/'  # Replace with your dataset directory
    MODEL_DIR = 'models/'
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Data augmentation for the images
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalize images to [0, 1]
        validation_split=0.2,  # Split data into training and validation
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'  # Fill missing pixels
    )

    # Training data generator
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',  # Use grayscale images
        class_mode='binary',  # Binary classification
        subset='training'
    )

    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='binary',
        subset='validation'
    )

    # Create the model
    model = create_signature_cnn_model()

    # Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'signature_cnn_model_v{epoch:02d}.keras'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    # Evaluate the model on the validation dataset
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Print classification report
    y_true = validation_generator.classes
    y_pred = (model.predict(validation_generator) > 0.5).astype("int32").flatten()
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))

    # Save the final trained model
    model.save(os.path.join(MODEL_DIR, 'signature_cnn_model_final.h5'))
    print("Model training complete. Final model saved!")

if __name__ == "__main__":
    train_model()
