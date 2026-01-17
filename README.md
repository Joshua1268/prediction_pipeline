# Fruits & Vegetables Predictive Analytics Platform

## Overview

The Fruits & Vegetables Predictive Analytics Platform is an AI-driven tool that forecasts demand, optimizes inventory, and manages perishable goods for fruits and vegetables. This system provides a comprehensive solution for agricultural businesses to make data-driven decisions about their inventory and pricing strategies.

## Features

- **Dashboard Overview**: Visualize key metrics and trends in your fruits and vegetables data
- **Data Exploration**: Interactive exploration of sales, inventory, and customer data with seasonal insights
- **Fruits & Vegetables Forecasting**: Predict future demand for individual fruits and vegetables
- **Inventory Planning**: Optimize inventory levels and suggest reorder points considering perishability
- **Pricing Optimization**: Dynamic pricing strategies based on demand forecasts and shelf life

## Project Structure

```
sagemaker_code/
├── app.py                    # Streamlit web application
├── generate_synthetic_data.py # Fruits & vegetables data generator
├── prediction_model.py       # Specialized prediction models for perishables
├── requirements.txt          # Dependencies
├── data/                     # Generated synthetic data
│   ├── fv_sales_data.csv
│   ├── fv_inventory_data.csv
│   └── fv_customer_data.csv
└── README.md               # This file
```

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone or download this repository to your local machine.

2. Navigate to the project directory:
```bash
cd sagemaker_code
```

3. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. From the project directory, run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will launch in your default browser at `http://localhost:8501`

3. If the browser doesn't open automatically, navigate to the displayed URL manually.

### Application Sections

- **Dashboard Overview**: Shows key metrics and visualizations of your fruits and vegetables data
- **Data Exploration**: Allows filtering and exploration of historical data with seasonal insights
- **Fruits & Vegetables Forecasting**: Generates demand forecasts for specific fruits and vegetables
- **Inventory Planning**: Provides inventory optimization recommendations considering shelf life
- **Pricing Optimization**: Suggests dynamic pricing strategies based on demand forecasts

### Data Generation

The application automatically generates synthetic fruits and vegetables data on first run. If you need to regenerate the data:

```bash
python generate_synthetic_data.py
```

## Model Architecture

The platform consists of:

1. **Fruits & Vegetables Data Generator**: Creates realistic sales, inventory, and customer data for fruits and vegetables with seasonal patterns
2. **Specialized Prediction Model**: Uses Random Forest or Linear Regression to predict demand with consideration for seasonality and perishability
3. **Forecasting Model**: Extends predictions to future time periods with confidence intervals adjusted for perishable goods
4. **Inventory Planning Module**: Calculates optimal reorder points and quantities considering shelf life
5. **Pricing Optimization Engine**: Suggests dynamic pricing based on demand forecasts and perishability

## Customization

To use your own data instead of synthetic data:

1. Place your CSV files in the `data/` directory with the following names:
   - `fv_sales_data.csv`
   - `fv_inventory_data.csv`
   - `fv_customer_data.csv`

2. Ensure your data follows the same schema as the generated synthetic data

## Troubleshooting

- If you encounter dependency issues, try upgrading pip: `pip install --upgrade pip`
- If Streamlit fails to start, ensure all dependencies are installed: `pip install -r requirements.txt`
- For performance issues with large datasets, consider sampling your data

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computations

## About the Platform

The Fruits & Vegetables Predictive Analytics Platform addresses the key requirements for managing perishable goods:

1. **Demand Forecasting**: Predicts future demand with seasonal adjustments
2. **Inventory Optimization**: Considers shelf life and perishability in recommendations
3. **Dynamic Pricing**: Adjusts pricing based on demand forecasts and shelf life
4. **User Interface**: Streamlit-based web interface for easy interaction
5. **Seasonal Analysis**: Incorporates seasonal patterns in fruits and vegetables demand

## License

This project is created for demonstration purposes. Feel free to adapt and extend it for your specific use case.
