import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ActiveLearningVGG16:
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = defaultdict(list)
        
    def create_model(self):
        """Create VGG-16 model adapted for CIFAR-10"""
        # Load VGG16 base model
        base_model = VGG16(weights='imagenet', 
                          include_top=False, 
                          input_shape=self.input_shape)
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def preprocess_data(self, x):
        """Preprocess data for VGG-16"""
        # Normalize to [0,1] range
        x = x.astype('float32') / 255.0
        
        # VGG-16 expects images in [0,255] range with ImageNet normalization
        x = x * 255.0
        
        # Apply ImageNet normalization
        x = tf.keras.applications.vgg16.preprocess_input(x)
        
        return x
    
    def calculate_confidence(self, predictions):
        """Calculate confidence scores for predictions"""
        # Use max probability as confidence score
        confidence_scores = np.max(predictions, axis=1)
        return confidence_scores
    
    def low_high_sampling(self, pool_indices, pool_data, n_samples=1000, 
                         low_ratio=0.5, high_ratio=0.5):
        """
        Low + High confidence sampling strategy
        Combines low-confidence (uncertain) and high-confidence samples
        """
        if len(pool_indices) == 0:
            return np.array([])
        
        # Get predictions for pool data
        pool_predictions = self.model.predict(pool_data, verbose=0)
        confidence_scores = self.calculate_confidence(pool_predictions)
        
        # Calculate number of samples for each category
        n_low = int(n_samples * low_ratio)
        n_high = int(n_samples * high_ratio)
        
        # Ensure we don't exceed available samples
        n_low = min(n_low, len(pool_indices))
        n_high = min(n_high, len(pool_indices))
        
        # Adjust if total exceeds n_samples
        if n_low + n_high > n_samples:
            n_low = n_samples // 2
            n_high = n_samples - n_low
        
        selected_indices = []
        
        # Select low-confidence samples (most uncertain)
        if n_low > 0:
            low_conf_indices = np.argsort(confidence_scores)[:n_low]
            selected_indices.extend(pool_indices[low_conf_indices])
        
        # Select high-confidence samples (most certain)
        if n_high > 0:
            high_conf_indices = np.argsort(confidence_scores)[-n_high:]
            selected_indices.extend(pool_indices[high_conf_indices])
        
        return np.array(selected_indices)
    
    def train_iteration(self, train_data, train_labels, val_data, val_labels, 
                       epochs=20, batch_size=32):
        """Train model for one iteration"""
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, test_data, test_labels):
        """Evaluate model on test set"""
        loss, accuracy = self.model.evaluate(test_data, test_labels, verbose=0)
        return accuracy
    
    def active_learning_loop(self, x_train, y_train, x_val, y_val, x_test, y_test,
                           initial_samples=1000, samples_per_iteration=500, 
                           max_iterations=10, epochs_per_iteration=15):
        """
        Main active learning loop with low + high confidence sampling
        """
        print("=== Active Learning with Low + High Confidence Sampling ===")
        print(f"Dataset size: {len(x_train)}")
        print(f"Initial samples: {initial_samples}")
        print(f"Samples per iteration: {samples_per_iteration}")
        print(f"Max iterations: {max_iterations}")
        print("-" * 60)
        
        # Preprocess all data
        x_train_prep = self.preprocess_data(x_train)
        x_val_prep = self.preprocess_data(x_val)
        x_test_prep = self.preprocess_data(x_test)
        
        # Initialize with random samples
        all_indices = np.arange(len(x_train))
        np.random.shuffle(all_indices)
        
        # Split into initial labeled and pool
        labeled_indices = all_indices[:initial_samples]
        pool_indices = all_indices[initial_samples:]
        
        # Create model
        self.create_model()
        
        # Track results
        results = {
            'iteration': [],
            'training_samples': [],
            'test_accuracy': [],
            'val_accuracy': [],
            'training_loss': [],
            'val_loss': []
        }
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            print(f"Training samples: {len(labeled_indices)}")
            print(f"Pool samples: {len(pool_indices)}")
            
            # Get current training data
            current_train_data = x_train_prep[labeled_indices]
            current_train_labels = y_train[labeled_indices]
            
            # Train model
            print("Training model...")
            history = self.train_iteration(
                current_train_data, current_train_labels,
                x_val_prep, y_val,
                epochs=epochs_per_iteration
            )
            
            # Evaluate on test set
            test_accuracy = self.evaluate_model(x_test_prep, y_test)
            val_accuracy = max(history.history['val_accuracy'])
            train_loss = min(history.history['loss'])
            val_loss = min(history.history['val_loss'])
            
            # Store results
            results['iteration'].append(iteration + 1)
            results['training_samples'].append(len(labeled_indices))
            results['test_accuracy'].append(test_accuracy)
            results['val_accuracy'].append(val_accuracy)
            results['training_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Select new samples using low + high confidence sampling
            if len(pool_indices) > 0 and iteration < max_iterations - 1:
                print("Selecting new samples using Low + High confidence sampling...")
                
                pool_data = x_train_prep[pool_indices]
                new_sample_indices = self.low_high_sampling(
                    pool_indices, pool_data, 
                    n_samples=min(samples_per_iteration, len(pool_indices)),
                    low_ratio=0.5, high_ratio=0.5
                )
                
                if len(new_sample_indices) > 0:
                    # Add new samples to labeled set
                    labeled_indices = np.concatenate([labeled_indices, new_sample_indices])
                    
                    # Remove selected samples from pool
                    pool_mask = np.isin(pool_indices, new_sample_indices, invert=True)
                    pool_indices = pool_indices[pool_mask]
                    
                    print(f"Added {len(new_sample_indices)} new samples")
                else:
                    print("No more samples to select")
                    break
            
            # Fine-tune VGG-16 layers in later iterations
            if iteration >= 2:
                print("Fine-tuning VGG-16 layers...")
                for layer in self.model.layers:
                    if 'block5' in layer.name or 'block4' in layer.name:
                        layer.trainable = True
                
                # Recompile with lower learning rate
                self.model.compile(
                    optimizer=Adam(learning_rate=0.0001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot results
        self.plot_results(results_df)
        
        return results_df
    
    def plot_results(self, results_df):
        """Plot training results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test Accuracy
        ax1.plot(results_df['training_samples'], results_df['test_accuracy'], 
                marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Training Samples')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Test Accuracy vs Training Samples\n(Low + High Confidence Sampling)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Validation Accuracy
        ax2.plot(results_df['training_samples'], results_df['val_accuracy'], 
                marker='s', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Training Samples')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy vs Training Samples')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Training Loss
        ax3.plot(results_df['training_samples'], results_df['training_loss'], 
                marker='^', linewidth=2, markersize=6, color='red')
        ax3.set_xlabel('Training Samples')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('Training Loss vs Training Samples')
        ax3.grid(True, alpha=0.3)
        
        # Validation Loss
        ax4.plot(results_df['training_samples'], results_df['val_loss'], 
                marker='d', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('Training Samples')
        ax4.set_ylabel('Validation Loss')
        ax4.set_title('Validation Loss vs Training Samples')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print final results table
        print("\n=== Final Results Summary ===")
        print(results_df.round(4))

def load_and_prepare_cifar10(max_samples=10000):
    """Load and prepare CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    
    # Load CIFAR-10
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Flatten labels
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()
    
    # Use subset of training data
    if max_samples < len(x_train_full):
        indices = np.random.choice(len(x_train_full), max_samples, replace=False)
        x_train_subset = x_train_full[indices]
        y_train_subset = y_train_full[indices]
    else:
        x_train_subset = x_train_full
        y_train_subset = y_train_full
    
    # Split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_subset, y_train_subset, 
        test_size=0.2, random_state=42, stratify=y_train_subset
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def main():
    """Main function to run active learning experiment"""
    # Load data (use a bigger sample size if available)
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_prepare_cifar10(max_samples=60000)
    
    # Create active learning instance
    al_model = ActiveLearningVGG16()
    
    # Run active learning loop with updated config
    results = al_model.active_learning_loop(
        x_train, y_train, x_val, y_val, x_test, y_test,
        initial_samples=10000,         # Start with 10,000 samples
        samples_per_iteration=5000,    # Add 5,000 samples per iteration
        max_iterations=9,             # Run for 9 iterations
        epochs_per_iteration=15
    )
    
    print("\n=== Experiment Complete ===")
    print("Results saved in 'results' DataFrame")
    
    return results


if __name__ == "__main__":
    # Run the experiment
    results = main()
