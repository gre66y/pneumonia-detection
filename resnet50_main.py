import os
import tensorflow as tf
from data_loader import prepare_datasets
from models import create_resnet50_model
from evaluation import evaluate_model, plot_training_history
from config import DATA_DIR, EPOCHS


def main():
    # Enable mixed precision training for better performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Prepare datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(DATA_DIR)

    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Train and evaluate ResNet50
    print("\nTraining ResNet50 model...")
    resnet50_model = create_resnet50_model()
    resnet50_history = resnet50_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate ResNet50
    print("\nEvaluating ResNet50...")
    resnet50_metrics = evaluate_model(resnet50_model, test_dataset, "ResNet50")
    plot_training_history(resnet50_history, "ResNet50")

    # Save model
    print("\nSaving ResNet50 model...")
    resnet50_model.save('resnet50_pneumonia.h5')

    # Print metrics
    precision, recall, f1, accuracy = resnet50_metrics
    print("\nResNet50 Final Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()