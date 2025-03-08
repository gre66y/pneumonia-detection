import os
import tensorflow as tf
from PIL import Image
from config import IMG_SIZE, BATCH_SIZE, CLASSES


def is_valid_file(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False


def create_dataset(data_dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        color_mode='grayscale',  # X-ray images are grayscale
        shuffle=True
    )

    # Print dataset information
    print(f"\nLoading data from: {os.path.basename(data_dir)}")

    # Count and validate files in each class
    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            valid_count = 0
            invalid_count = 0
            for file in os.listdir(class_dir):
                filepath = os.path.join(class_dir, file)
                if is_valid_file(filepath):
                    valid_count += 1
                else:
                    invalid_count += 1
                    print(f"Skipping corrupted file: {filepath}")

            print(f"{class_name} - Valid files: {valid_count}, Corrupted files: {invalid_count}")

    return dataset


def prepare_datasets(base_dir):
    print("Preparing datasets...")

    train_dataset = create_dataset(os.path.join(base_dir, 'train'))
    val_dataset = create_dataset(os.path.join(base_dir, 'val'))
    test_dataset = create_dataset(os.path.join(base_dir, 'test'))

    # Normalize the data
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    # Data augmentation for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Apply normalization to all datasets
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Apply augmentation only to training data
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, test_dataset