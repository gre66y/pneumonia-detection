import os
import tensorflow as tf
from data_loader import prepare_datasets
from models import create_vgg19_model, create_resnet50_model
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

    # Train and evaluate VGG19
    print("\nTraining VGG19 model...")
    vgg19_model = create_vgg19_model()
    vgg19_history = vgg19_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate VGG19
    print("\nEvaluating VGG19...")
    vgg19_metrics = evaluate_model(vgg19_model, test_dataset, "VGG19")
    plot_training_history(vgg19_history, "VGG19")

    # Clear session to free up memory
    tf.keras.backend.clear_session()

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

    # Save models
    print("\nSaving models...")
    vgg19_model.save('vgg19_pneumonia.h5')
    resnet50_model.save('resnet50_pneumonia.h5')

    # Compare models
    print("\nModel Comparison:")
    models = {
        "VGG19": vgg19_metrics,
        "ResNet50": resnet50_metrics
    }

    for model_name, metrics in models.items():
        precision, recall, f1, accuracy = metrics
        print(f"\n{model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()