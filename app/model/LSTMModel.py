import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from app.configuration.Logger_config import setup_logger, logger


class LSTMModel:
    """
    LSTM neural network for stock price prediction
    Handles sequential time series data
    """
    
    def __init__(self, config):
        self.config = config
        self.params = config['models']['lstm']['params']
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = self.params['sequence_length']
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def create_sequences(self, data, target=None):
        """
        Create sequences for LSTM input
        
        Args:
            data: Feature dataframe
            target: Target values (optional, for training)
            
        Returns:
            X: 3D array (samples, sequence_length, features)
            y: Target values (if provided)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            
            if target is not None:
                y.append(target[i + self.sequence_length])
        
        X = np.array(X)
        
        if target is not None:
            y = np.array(y)
            return X, y
        
        return X
    
    def prepare_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Prepare data for LSTM: scaling and sequence creation
        """
        logger.info("Preparing data for LSTM...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train.values)
        
        logger.info(f"Training sequences: {X_train_seq.shape}")
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val.values)
            logger.info(f"Validation sequences: {X_val_seq.shape}")
            return X_train_seq, y_train_seq, X_val_seq, y_val_seq
        
        return X_train_seq, y_train_seq, None, None
    
    def build_model(self, input_shape, task='regression'):
        """
        Build LSTM architecture
        
        Args:
            input_shape: (sequence_length, n_features)
            task: 'regression' or 'classification'
        """
        logger.info(f"Building LSTM model for {task}...")
        
        model = keras.Sequential()
        
        # LSTM layers from config
        lstm_units = self.params['lstm_units']
        dropout = self.params['dropout']
        
        # First LSTM layer
        model.add(layers.LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            model.add(layers.LSTM(units, return_sequences=return_seq))
            model.add(layers.Dropout(dropout))
        
        # Dense layers
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(16, activation='relu'))
        
        # Output layer
        if task == 'regression':
            model.add(layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, task='regression'):
        """
        Train the LSTM model
        """
        logger.info("Training LSTM model...")
        
        # Prepare sequences
        X_train_seq, y_train_seq, X_val_seq, y_val_seq = self.prepare_data(
            X_train, y_train, X_val, y_val
        )
        
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.build_model(input_shape, task)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val_seq is not None else 'loss',
                patience=self.config['training']['early_stopping']['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val_seq is not None else 'loss',
                factor=0.5,
                patience=10,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Train
        validation_data = (X_val_seq, y_val_seq) if X_val_seq is not None else None
        
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("LSTM training complete")
        
        return self.model

    def predict(self, X):
        """
        Make predictions - FIXED VERSION
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Scale and create sequences
        X_scaled = self.scaler.transform(X)
        X_seq = self.create_sequences(X_scaled)

        # CRITICAL FIX: Check if we have sequences
        if len(X_seq) == 0:
            logger.warning("No sequences available for LSTM prediction")
            logger.warning(f"Input length: {len(X)}, Sequence length: {self.sequence_length}")
            # Return array of NaN with correct length
            return np.full(len(X), np.nan)

        # Predict
        try:
            predictions = self.model.predict(X_seq, verbose=0)
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            # Return array of NaN instead of crashing
            return np.full(len(X), np.nan)

        # Flatten if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        # Pad beginning with NaN (lost to sequence creation)
        padding = np.full(self.sequence_length, np.nan)
        predictions = np.concatenate([padding, predictions])

        # Ensure correct length
        if len(predictions) != len(X):
            logger.warning(f"Prediction length mismatch: {len(predictions)} vs {len(X)}")
            # Pad or truncate to match
            if len(predictions) < len(X):
                padding_needed = len(X) - len(predictions)
                predictions = np.concatenate([predictions, np.full(padding_needed, np.nan)])
            else:
                predictions = predictions[:len(X)]

        return predictions
    
    def evaluate(self, X, y, task='regression'):
        """
        Evaluate model performance
        """
        predictions = self.predict(X)
        
        # Remove NaN values
        mask = ~np.isnan(predictions)
        predictions = predictions[mask]
        y_eval = y.values[mask]
        
        if task == 'regression':
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_eval, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_eval, predictions)
            r2 = r2_score(y_eval, predictions)
            
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            logger.info(f"LSTM Evaluation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")
            
        else:  # classification
            from sklearn.metrics import accuracy_score
            
            predictions_binary = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(y_eval, predictions_binary)
            
            correct_direction = np.sum((predictions > 0.5) == (y_eval > 0))
            directional_accuracy = correct_direction / len(y_eval)
            
            metrics = {
                'Accuracy': accuracy,
                'Directional_Accuracy': directional_accuracy
            }
            
            logger.info(f"LSTM Evaluation - Accuracy: {accuracy:.4f}, Directional: {directional_accuracy:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to disk"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"LSTM model loaded from {filepath}")
    
    def plot_training_history(self):
        """
        Plot training history
        Requires matplotlib
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            ax1.plot(self.history.history['loss'], label='Train Loss')
            if 'val_loss' in self.history.history:
                ax1.plot(self.history.history['val_loss'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Model Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Metric
            metric_name = list(self.history.history.keys())[1]
            ax2.plot(self.history.history[metric_name], label=f'Train {metric_name}')
            if f'val_{metric_name}' in self.history.history:
                ax2.plot(self.history.history[f'val_{metric_name}'], label=f'Val {metric_name}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(metric_name)
            ax2.set_title(f'Model {metric_name}')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None
