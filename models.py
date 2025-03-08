import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.models import Model


def create_vgg19_model():
    # Modify input shape for grayscale images
    input_tensor = Input(shape=(224, 224, 1))
    # Convert grayscale to RGB by repeating the channel
    x = tf.keras.layers.Concatenate(axis=-1)([input_tensor, input_tensor, input_tensor])

    base_model = VGG19(weights='imagenet', include_top=False, input_tensor=x)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout to prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


def create_resnet50_model():
    # Modify input shape for grayscale images
    input_tensor = Input(shape=(224, 224, 1))
    # Convert grayscale to RGB by repeating the channel
    x = tf.keras.layers.Concatenate(axis=-1)([input_tensor, input_tensor, input_tensor])

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout to prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model