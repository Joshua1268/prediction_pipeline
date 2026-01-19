#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FruitsVegetablesPredictionModel:

    def __init__(self):
        """
        Initialize the prediction model with default parameters.
        """
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'quantity_sold'

    def preprocess_data(self, df):
        """
        Preprocess the data for training/prediction with focus on fruits and vegetables.

        Args:
            df (pd.DataFrame): Input dataframe with raw data

        Returns:
            pd.DataFrame: Processed dataframe with engineered features
        """
        df_processed = df.copy()

        # Convert date to datetime if it's not already
        df_processed['date'] = pd.to_datetime(df_processed['date'])

        # Extract date features
        df_processed['year'] = df_processed['date'].dt.year
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['day'] = df_processed['date'].dt.day
        df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
        df_processed['quarter'] = df_processed['date'].dt.quarter
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear

        # Add seasonal features specific to fruits and vegetables
        df_processed['is_summer'] = ((df_processed['month'] >= 6) & (df_processed['month'] <= 8)).astype(int)
        df_processed['is_winter'] = ((df_processed['month'] >= 12) | (df_processed['month'] <= 2)).astype(int)
        df_processed['is_spring'] = ((df_processed['month'] >= 3) & (df_processed['month'] <= 5)).astype(int)
        df_processed['is_fall'] = ((df_processed['month'] >= 9) & (df_processed['month'] <= 11)).astype(int)

        # Encode categorical variables
        categorical_columns = ['product_id', 'product_name', 'category']
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                # Fit on all unique values to handle potential unseen values later
                le.fit(df_processed[col].astype(str))
                df_processed[f'{col}_encoded'] = le.transform(df_processed[col].astype(str))
                self.label_encoders[col] = le

        # Select features for the model with emphasis on perishability and seasonality
        feature_columns = [
            'product_id_encoded', 'product_name_encoded', 'category_encoded',
            'unit_price', 'month', 'day', 'day_of_week', 'quarter', 'day_of_year',
            'is_weekend', 'is_summer', 'is_winter', 'is_spring', 'is_fall',
            'shelf_life_days'
        ]

        # Make sure all feature columns exist in the dataframe
        available_features = [col for col in feature_columns if col in df_processed.columns]
        self.feature_columns = available_features

        return df_processed

    def prepare_features_and_target(self, df_processed, for_training=True):
        """
        Prepare features (X) and target (y) for training or prediction.

        Args:
            df_processed (pd.DataFrame): Processed dataframe
            for_training (bool): Whether preparing for training (True) or prediction (False)

        Returns:
            tuple: Features (X) and target (y) if for_training is True, else just features (X)
        """
        X = df_processed[self.feature_columns]
        if for_training and self.target_column in df_processed.columns:
            y = df_processed[self.target_column]
            return X, y
        else:
            return X, None

    def train(self, df, model_type='random_forest'):
        """
        Train the prediction model with focus on fruits and vegetables characteristics.

        Args:
            df (pd.DataFrame): Training data
            model_type (str): Type of model to train ('random_forest' or 'linear_regression')

        Returns:
            dict: Dictionary with evaluation metrics
        """
        print("Preprocessing fruits and vegetables data...")
        df_processed = self.preprocess_data(df)

        print("Preparing features and target...")
        X, y = self.prepare_features_and_target(df_processed, for_training=True)

        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training {model_type} model for fruits and vegetables...")
        if model_type == 'random_forest':
            # Use parameters suitable for seasonal/perishable goods
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Fruits & Vegetables Model Evaluation Metrics:")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  R² Score: {r2:.2f}")

        return {'mae': mae, 'mse': mse, 'r2': r2}

    def predict(self, df):
        """
        Make predictions on new data.

        Args:
            df (pd.DataFrame): Input data for prediction

        Returns:
            np.array: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        df_processed = self.preprocess_data(df)
        X, _ = self.prepare_features_and_target(df_processed, for_training=False)

        predictions = self.model.predict(X)
        return predictions

    def save_model(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Fruits & Vegetables model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']

        print(f"Fruits & Vegetables model loaded from {filepath}")


class FruitsVegetablesForecastingModel:
    """
    Specialized model for forecasting demand of fruits and vegetables with consideration for seasonality and perishability.
    """

    def __init__(self):
        """
        Initialize the forecasting model.
        """
        self.prediction_model = FruitsVegetablesPredictionModel()
        self.forecast_horizon = 30  # Days to forecast

    def forecast_demand(self, historical_data, product_id, days_ahead=30):
        """
        Forecast demand for a specific fruit or vegetable for the next 'days_ahead' days.

        Args:
            historical_data (pd.DataFrame): Historical sales data
            product_id (str): Product identifier
            days_ahead (int): Number of days to forecast ahead

        Returns:
            pd.DataFrame: Forecast results
        """
        product_data = historical_data[historical_data['product_id'] == product_id].copy()

        if product_data.empty:
            print(f"No historical data found for product {product_id}")
            return None

        
        self.prediction_model.train(product_data)

        last_date = pd.to_datetime(product_data['date']).max()
        future_dates = []

        for i in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=i)
            future_dates.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'product_id': product_id,
                'product_name': product_data['product_name'].iloc[0],
                'category': product_data['category'].iloc[0],
                'unit_price': product_data['unit_price'].mean(),
                'is_weekend': 1 if future_date.weekday() >= 5 else 0,
                'is_summer': 1 if future_date.month in [6, 7, 8] else 0,
                'is_winter': 1 if future_date.month in [12, 1, 2] else 0,
                'is_spring': 1 if future_date.month in [3, 4, 5] else 0,
                'is_fall': 1 if future_date.month in [9, 10, 11] else 0,
                'shelf_life_days': product_data['shelf_life_days'].iloc[0] if 'shelf_life_days' in product_data.columns else 7
            })

        future_df = pd.DataFrame(future_dates)

        predictions = self.prediction_model.predict(future_df)

        forecast_results = []
        for i, pred in enumerate(predictions):
            # Adjust confidence intervals based on perishability
            shelf_life = future_df.iloc[i]['shelf_life_days']
            conf_lower_mult = 0.7 if shelf_life < 7 else 0.8
            conf_upper_mult = 1.3 if shelf_life < 7 else 1.2

            forecast_results.append({
                'date': future_dates[i]['date'],
                'product_id': product_id,
                'product_name': product_data['product_name'].iloc[0],
                'predicted_quantity': max(0, int(round(pred))),  # Ensure non-negative
                'confidence_interval_lower': max(0, int(round(pred * conf_lower_mult))),  # Adjusted for perishability
                'confidence_interval_upper': int(round(pred * conf_upper_mult)),
                'shelf_life_days': shelf_life
            })

        return pd.DataFrame(forecast_results)

    def forecast_inventory_replenishment(self, historical_data, product_id, reorder_level=30):
        """
        Suggest inventory replenishment for fruits and vegetables based on forecast and shelf life.

        Args:
            historical_data (pd.DataFrame): Historical sales data
            product_id (str): Product identifier
            reorder_level (int): Minimum stock level that triggers reorder

        Returns:
            dict: Replenishment recommendation
        """
        forecast = self.forecast_demand(historical_data, product_id, days_ahead=30)

        if forecast is None:
            return None

        # Get product shelf life
        product_info = historical_data[historical_data['product_id'] == product_id].iloc[0]
        shelf_life = product_info.get('shelf_life_days', 7)

        # Find when inventory might fall below reorder level
        forecast['cumulative_demand'] = forecast['predicted_quantity'].cumsum()

        # Simple inventory simulation considering shelf life
        current_stock = historical_data[
            (historical_data['product_id'] == product_id)
        ]['current_stock'].iloc[-1] if 'current_stock' in historical_data.columns else 100

        # Account for spoilage based on shelf life
        forecast['estimated_stock'] = current_stock - forecast['cumulative_demand']

        adjusted_reorder_level = reorder_level if shelf_life > 10 else reorder_level * 1.5

        # Find when to reorder
        reorder_dates = forecast[forecast['estimated_stock'] <= adjusted_reorder_level]

        if not reorder_dates.empty:
            suggested_reorder_date = reorder_dates.iloc[0]['date']
            suggested_quantity = int(adjusted_reorder_level * 2.5) if shelf_life < 7 else int(adjusted_reorder_level * 2)

            return {
                'product_id': product_id,
                'product_name': product_info['product_name'],
                'suggested_reorder_date': suggested_reorder_date,
                'suggested_quantity': suggested_quantity,
                'current_stock': current_stock,
                'reorder_level': adjusted_reorder_level,
                'shelf_life_days': shelf_life
            }

        return None

    def optimize_pricing(self, historical_data, product_id, base_price, days_ahead=7):
        """
        Suggest pricing optimization for fruits and vegetables based on demand forecast and shelf life.

        Args:
            historical_data (pd.DataFrame): Historical sales data
            product_id (str): Product identifier
            base_price (float): Current product price
            days_ahead (int): Number of days to forecast for pricing

        Returns:
            dict: Pricing optimization recommendation
        """
        forecast = self.forecast_demand(historical_data, product_id, days_ahead)

        if forecast is None:
            return None

        # Get product information
        product_info = historical_data[historical_data['product_id'] == product_id].iloc[0]
        shelf_life = product_info.get('shelf_life_days', 7)

        # Calculate average demand over the forecast period
        avg_demand = forecast['predicted_quantity'].mean()

        if shelf_life <= 5:  # Highly perishable
            if avg_demand < 10:  # Low demand
                suggested_price = base_price * 0.8  # Discount to move inventory
            else:
                suggested_price = base_price  # Normal price
        elif shelf_life <= 10:  # Moderately perishable
            if avg_demand < 15:  # Low demand
                suggested_price = base_price * 0.9  # Small discount
            else:
                suggested_price = base_price  # Normal price
        else:  # Longer shelf life
            if avg_demand > 25:  # High demand
                suggested_price = base_price * 1.1  # Premium pricing possible
            else:
                suggested_price = base_price  # Normal price

        return {
            'product_id': product_id,
            'product_name': product_info['product_name'],
            'current_price': base_price,
            'suggested_price': suggested_price,
            'pricing_strategy': 'Discount' if suggested_price < base_price else 'Premium' if suggested_price > base_price else 'Standard',
            'avg_forecasted_demand': avg_demand,
            'shelf_life_days': shelf_life
        }


def evaluate_model_performance(actual_values, predicted_values):
    """
    Evaluate model performance using multiple metrics.

    Args:
        actual_values (np.array): Actual values
        predicted_values (np.array): Predicted values

    Returns:
        dict: Dictionary with evaluation metrics
    """
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)

    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


if __name__ == "__main__":
    print("Fruits & Vegetables Prediction Model - Example Usage")

    print("\nTo use this model:")
    print("1. Load your fruits and vegetables sales data into a pandas DataFrame")
    print("2. Initialize the FruitsVegetablesPredictionModel()")
    print("3. Call model.train(df) to train the model")
    print("4. Use model.predict(new_data) to make predictions")
    print("5. Save the model with model.save_model(filepath)")

    print("\nFruits & Vegetables Forecasting Example:")
    print("1. Initialize FruitsVegetablesForecastingModel()")
    print("2. Call forecast_demand(historical_data, product_id, days_ahead)")
    print("3. Get predictions for future demand")
    print("4. Use forecast_inventory_replenishment() for inventory optimization")
    print("5. Use optimize_pricing() for dynamic pricing strategies")
