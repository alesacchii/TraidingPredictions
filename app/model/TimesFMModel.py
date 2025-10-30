import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from configuration.Logger_config import setup_logger, logger
import timesfm


class TimesFMModel:
    """
    Google TimesFM (Time Series Foundation Model) integration
    Pre-trained foundation model for zero-shot time series forecasting
    Model: google/timesfm-2.5-200m-pytorch
    """

    def __init__(self, config):
        self.config = config
        self.params = config['models']['timesfm']['params']
        self.model_name = config['models']['timesfm']['model_name']
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

        logger.info(f"Initializing TimesFM on device: {self.device}")

    def _get_device(self):
        """Determine device to use (GPU/CPU)"""
        backend = self.params.get('backend', 'cpu')

        if backend == 'gpu' and torch.cuda.is_available():
            return 'cuda'
        elif backend == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_model(self):
        """
        Load TimesFM model using the actual working implementation
        Based on: google/timesfm-2.5-200m-pytorch
        """
        logger.info(f"Loading TimesFM model: {self.model_name}")

        try:

            logger.info("timesfm library found, loading model...")

            # Set precision for better performance
            torch.set_float32_matmul_precision("high")

            # Load the pre-trained model
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_name)

            # Compile with forecast config
            context_length = self.params.get('context_length', 512)
            horizon_length = self.params.get('horizon_length', 30)

            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=context_length,
                    max_horizon=horizon_length,
                    normalize_inputs=False,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )

            logger.info("TimesFM model loaded and compiled successfully")

        except ImportError as e:
            logger.warning(f"timesfm library not installed: {e}")
            logger.info("Install with: pip install git+https://github.com/google-research/timesfm.git")
            logger.info("The system will work fine without it using XGBoost + LightGBM")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load TimesFM model: {e}")
            logger.info("The system will continue with XGBoost + LightGBM")
            self.model = None

    def prepare_input(self, data, stock_name=None):
        """
        Prepare input data for TimesFM

        Args:
            data: DataFrame with historical prices/features
            stock_name: Optional stock identifier

        Returns:
            Prepared input in TimesFM format
        """
        # TimesFM expects univariate time series
        # Use Close price as the main signal

        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                time_series = data['Close'].values
            else:
                # Use first numeric column
                time_series = data.select_dtypes(include=[np.number]).iloc[:, 0].values
        else:
            time_series = np.array(data)

        # Normalize to avoid scale issues
        mean = np.mean(time_series)
        std = np.std(time_series)
        normalized_series = (time_series - mean) / (std + 1e-8)

        # Take last context_length points
        context_length = self.params['context_length']
        if len(normalized_series) > context_length:
            normalized_series = normalized_series[-context_length:]

        return {
            'time_series': normalized_series,
            'mean': mean,
            'std': std,
            'length': len(normalized_series)
        }

    def forecast(self, data, horizon=None):
        """
        Generate forecast using TimesFM with proper API (based on working example)

        Args:
            data: Historical data (DataFrame with Close prices)
            horizon: Forecast horizon (number of steps ahead)

        Returns:
            predictions: Array of forecasted values
        """
        if self.model is None:
            logger.error("TimesFM model not loaded")
            return None

        if horizon is None:
            horizon = self.params['horizon_length']

        try:
            from sklearn.preprocessing import StandardScaler

            # Extract time series (use Close prices)
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    time_series = data['Close'].values
                else:
                    time_series = data.select_dtypes(include=[np.number]).iloc[:, 0].values
            else:
                time_series = np.array(data)

            # Take last context_length points
            context_length = self.params['context_length']
            if len(time_series) > context_length:
                input_data = time_series[-context_length:]
            else:
                input_data = time_series

            # Normalize using StandardScaler (as in working example)
            scaler = StandardScaler()
            input_data_reshaped = input_data.reshape(-1, 1)
            normalized_input = scaler.fit_transform(input_data_reshaped)

            # Flatten and convert to float32
            inputs_list = [normalized_input.flatten().astype(np.float32)]

            # Forecast using actual TimesFM API
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=inputs_list,
            )

            # Extract point forecast (already 1D array)
            normalized_forecast = point_forecast[0]

            # De-normalize
            normalized_forecast_2d = normalized_forecast.reshape(-1, 1)
            forecast_values = scaler.inverse_transform(normalized_forecast_2d).flatten()

            return forecast_values

        except Exception as e:
            logger.error(f"Error during TimesFM forecast: {e}")
            return None

    def _generate_fallback(self, time_series, horizon):
        """
        Fallback prediction method if official API not available
        Uses simple extrapolation based on recent trends
        """
        # This is a placeholder - in production, implement proper inference

        # Calculate recent trend
        recent_window = min(20, len(time_series) // 4)
        recent_trend = np.polyfit(range(recent_window), time_series[-recent_window:], deg=1)[0]

        # Project forward with dampening
        predictions = []
        last_value = time_series[-1]

        for i in range(1, horizon + 1):
            # Dampen trend over time
            dampening = 0.95 ** i
            next_value = last_value + recent_trend * dampening
            predictions.append(next_value)
            last_value = next_value

        return np.array(predictions)

    def predict(self, X, horizon=1):
        """
        Make predictions for multiple stocks

        Args:
            X: DataFrame with features (must include Close prices)
            horizon: Forecast horizon

        Returns:
            predictions: Array of predictions
        """
        if self.model is None:
            logger.warning("TimesFM model not available, returning None")
            return None

        try:
            # For each stock, generate forecast
            if 'Stock' in X.columns:
                predictions_list = []

                for stock in X['Stock'].unique():
                    stock_data = X[X['Stock'] == stock]
                    forecast = self.forecast(stock_data, horizon=horizon)

                    if forecast is not None:
                        # Take the prediction at the specified horizon
                        pred_value = forecast[min(horizon - 1, len(forecast) - 1)]
                        predictions_list.extend([pred_value] * len(stock_data))
                    else:
                        predictions_list.extend([np.nan] * len(stock_data))

                return np.array(predictions_list)
            else:
                # Single time series
                forecast = self.forecast(X, horizon=horizon)
                if forecast is not None:
                    pred_value = forecast[min(horizon - 1, len(forecast) - 1)]
                    return np.full(len(X), pred_value)
                else:
                    return np.full(len(X), np.nan)

        except Exception as e:
            logger.error(f"Error in TimesFM prediction: {e}")
            return None

    def evaluate(self, X, y, task='regression'):
        """
        Evaluate TimesFM predictions
        """
        predictions = self.predict(X)

        if predictions is None:
            logger.warning("Cannot evaluate - predictions failed")
            return {}

        # Remove NaN values
        mask = ~np.isnan(predictions)
        predictions = predictions[mask]
        y_eval = y.values[mask] if hasattr(y, 'values') else y[mask]

        if len(predictions) == 0:
            logger.warning("No valid predictions for evaluation")
            return {}

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

        logger.info(f"TimesFM Evaluation - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")

        return metrics

    def train(self, X_train, y_train, X_val=None, y_val=None, task='regression'):
        """
        TimesFM is a pre-trained model (zero-shot)
        No training needed, but we implement this for API compatibility
        """
        logger.info("TimesFM is a pre-trained foundation model - no training required")

        # Only try to load if model is None
        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.warning(f"Could not load TimesFM model: {e}")
                self.model = None

        return self.model


# Utility function for easy instantiation
def create_timesfm_model(config):
    """Create and load TimesFM model"""

    if not config['models']['timesfm']['enabled']:
        logger.info("TimesFM is disabled in config")
        return None

    try:
        model = TimesFMModel(config)
        model.load_model()
        return model
    except Exception as e:
        logger.error(f"Failed to create TimesFM model: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from configuration.Logger_config import load_config

    config = load_config('config.yaml')
    timesfm_model = create_timesfm_model(config)

    if timesfm_model and timesfm_model.model:
        print("TimesFM model loaded successfully.")
    else:
        print("TimesFM model not available.")