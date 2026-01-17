#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add the project directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_data import FruitsVegetablesDataGenerator
from prediction_model import FruitsVegetablesForecastingModel


def main():
    """
    Main function to run the Streamlit application.
    Sets up the page configuration, header, navigation, and data loading.
    """
    st.set_page_config(
        page_title="Fruits & Vegetables Predictive Analytics Platform",
        layout="wide"
    )
    
    # Professional header
    st.title("Fruits & Vegetables Predictive Analytics Platform")
    st.markdown("""
    Advanced analytics tool for forecasting demand, optimizing inventory, and managing perishable goods for fruits and vegetables.
    """)
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Select a section:", 
                                   ["Dashboard Overview", 
                                    "Data Exploration", 
                                    "Product Forecasting", 
                                    "Inventory Planning", 
                                    "Pricing Optimization"])
    
    # Generate synthetic data if not already present
    data_dir = "data"
    sales_data_path = os.path.join(data_dir, "fv_sales_data.csv")
    inventory_data_path = os.path.join(data_dir, "fv_inventory_data.csv")
    customer_data_path = os.path.join(data_dir, "fv_customer_data.csv")
    
    if not all(os.path.exists(path) for path in [sales_data_path, inventory_data_path, customer_data_path]):
        st.info("Generating synthetic fruits and vegetables data for demonstration...")
        generator = FruitsVegetablesDataGenerator(start_date="2023-01-01", end_date="2024-12-31")
        sales_data, inventory_data, customer_data = generator.generate_all_data()
        
        # Save the generated data
        os.makedirs(data_dir, exist_ok=True)
        sales_df = pd.DataFrame(sales_data)
        inventory_df = pd.DataFrame(inventory_data)
        customer_df = pd.DataFrame(customer_data)
        
        sales_df.to_csv(sales_data_path, index=False)
        inventory_df.to_csv(inventory_data_path, index=False)
        customer_df.to_csv(customer_data_path, index=False)
        
        st.success("Fruits and vegetables data generated successfully!")
    else:
        # Load existing data
        sales_df = pd.read_csv(sales_data_path)
        inventory_df = pd.read_csv(inventory_data_path)
        customer_df = pd.read_csv(customer_data_path)
        
        # Convert date columns to datetime
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        inventory_df["date"] = pd.to_datetime(inventory_df["date"])
        customer_df["date"] = pd.to_datetime(customer_df["date"])
    
    if app_mode == "Dashboard Overview":
        dashboard_overview(sales_df, inventory_df, customer_df)
    elif app_mode == "Data Exploration":
        data_exploration(sales_df, inventory_df, customer_df)
    elif app_mode == "Product Forecasting":
        product_forecasting(sales_df)
    elif app_mode == "Inventory Planning":
        inventory_planning(sales_df, inventory_df)
    elif app_mode == "Pricing Optimization":
        pricing_optimization(sales_df)


def dashboard_overview(sales_df, inventory_df, customer_df):
    """
    Displays the main dashboard with key metrics and visualizations.
    
    Args:
        sales_df (pd.DataFrame): Sales data
        inventory_df (pd.DataFrame): Inventory data
        customer_df (pd.DataFrame): Customer data
    """
    st.header("Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = sales_df["revenue"].sum()
    total_profit = sales_df["profit"].sum()
    avg_daily_sales = sales_df.groupby(sales_df["date"].dt.date)["quantity_sold"].sum().mean()
    top_category = sales_df["category"].mode()[0] if not sales_df.empty else "N/A"
    
    with col1:
        st.metric(label="Total Revenue", value=f"${total_revenue:,.2f}")
    with col2:
        st.metric(label="Total Profit", value=f"${total_profit:,.2f}")
    with col3:
        st.metric(label="Avg Daily Sales", value=f"{avg_daily_sales:.1f}")
    with col4:
        st.metric(label="Top Category", value=top_category)
    
    # Time series chart of sales
    st.subheader("Sales Trend Over Time")
    daily_sales = sales_df.groupby(sales_df["date"].dt.date)["revenue"].sum().reset_index()
    daily_sales.columns = ["Date", "Revenue"]
    
    fig = px.line(daily_sales, x="Date", y="Revenue", title="Daily Revenue Trend")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category distribution
    st.subheader("Sales by Category")
    category_sales = sales_df.groupby("category")["revenue"].sum().reset_index()
    
    fig = px.pie(category_sales, values="revenue", names="category", title="Revenue Distribution by Category")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performing products
    st.subheader("Top Performing Products")
    top_products = sales_df.groupby("product_name")["revenue"].sum().nlargest(10).reset_index()
    
    fig = px.bar(top_products, x="revenue", y="product_name", orientation="h", 
                 title="Top 10 Products by Revenue")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def data_exploration(sales_df, inventory_df, customer_df):
    """
    Provides interactive exploration of sales, inventory, and customer data.
    
    Args:
        sales_df (pd.DataFrame): Sales data
        inventory_df (pd.DataFrame): Inventory data
        customer_df (pd.DataFrame): Customer data
    """
    st.header("Data Exploration")
    
    # Date range selector
    min_date = sales_df["date"].min().date()
    max_date = sales_df["date"].max().date()
    
    date_range = st.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (sales_df["date"].dt.date >= start_date) & (sales_df["date"].dt.date <= end_date)
        filtered_sales = sales_df.loc[mask]
    else:
        filtered_sales = sales_df
    
    # Sales metrics by category
    st.subheader("Sales Metrics by Category")
    category_metrics = filtered_sales.groupby("category").agg({
        "quantity_sold": "sum",
        "revenue": "sum",
        "profit": "sum"
    }).reset_index()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(category_metrics, x="category", y="quantity_sold", 
                     title="Total Quantity Sold by Category")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(category_metrics, x="category", y="revenue", 
                     title="Total Revenue by Category")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.bar(category_metrics, x="category", y="profit", 
                     title="Total Profit by Category")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("Seasonal Sales Analysis")
    seasonal_sales = filtered_sales.groupby("season").agg({
        "quantity_sold": "sum",
        "revenue": "sum"
    }).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seasonal_sales, x="season", y="quantity_sold", 
                     title="Total Quantity Sold by Season")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(seasonal_sales, x="season", y="revenue", 
                     title="Total Revenue by Season")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Quantity scatter plot
    st.subheader("Price vs Quantity Relationship")
    fig = px.scatter(filtered_sales, x="unit_price", y="quantity_sold", 
                     color="category", hover_data=["product_name"],
                     title="Unit Price vs Quantity Sold")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def product_forecasting(sales_df):
    """
    Provides forecasting capabilities for individual products.
    
    Args:
        sales_df (pd.DataFrame): Sales data
    """
    st.header("Product Forecasting")
    
    # Product selection
    products = sales_df["product_name"].unique()
    selected_product = st.selectbox("Select a product to forecast:", options=products)
    
    # Forecast horizon
    forecast_days = st.slider("Select forecast horizon (days):", min_value=7, max_value=90, value=30, step=7)
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            # Prepare data for the selected product
            product_data = sales_df[sales_df["product_name"] == selected_product].copy()
            
            if product_data.empty:
                st.warning(f"No data found for product: {selected_product}")
                return
            
            # Initialize and train the forecasting model
            forecasting_model = FruitsVegetablesForecastingModel()
            forecast_result = forecasting_model.forecast_demand(
                historical_data=product_data, 
                product_id=product_data["product_id"].iloc[0], 
                days_ahead=forecast_days
            )
            
            if forecast_result is not None:
                st.success("Forecast generated successfully!")
                
                # Show forecast data
                st.subheader(f"Forecast for {selected_product}")
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                historical_dates = product_data["date"].dt.date
                historical_values = product_data["quantity_sold"]
                
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_values,
                    mode="lines",
                    name="Historical Sales",
                    line=dict(color="blue")
                ))
                
                # Forecast data
                forecast_dates = pd.to_datetime(forecast_result["date"])
                forecast_values = forecast_result["predicted_quantity"]
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="red", dash="dash")
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result["confidence_interval_upper"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(255, 0, 0, 0.2)",
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result["confidence_interval_lower"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(255, 0, 0, 0.2)",
                    name="Confidence Interval"
                ))
                
                fig.update_layout(
                    title=f"Demand Forecast for {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Quantity Sold",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.subheader("Forecast Table")
                st.dataframe(forecast_result.style.format({
                    "predicted_quantity": "{:.0f}",
                    "confidence_interval_lower": "{:.0f}",
                    "confidence_interval_upper": "{:.0f}"
                }))
            else:
                st.warning("Could not generate forecast for the selected product.")


def inventory_planning(sales_df, inventory_df):
    """
    Provides inventory planning and optimization recommendations.
    
    Args:
        sales_df (pd.DataFrame): Sales data
        inventory_df (pd.DataFrame): Inventory data
    """
    st.header("Inventory Planning")
    
    # Product selection
    products = sales_df["product_name"].unique()
    selected_product = st.selectbox("Select a product for inventory planning:", options=products)
    
    # Get product ID
    product_id = sales_df[sales_df["product_name"] == selected_product]["product_id"].iloc[0]
    
    # Current stock level
    current_stock = inventory_df[inventory_df["product_id"] == product_id]["current_stock"].iloc[-1] if not inventory_df[inventory_df["product_id"] == product_id].empty else 100
    current_stock = st.number_input("Current Stock Level:", value=int(current_stock), min_value=0)
    
    # Reorder level
    reorder_level = st.number_input("Reorder Level:", value=30, min_value=1)
    
    if st.button("Generate Inventory Recommendations"):
        with st.spinner("Analyzing inventory needs..."):
            # Prepare data for the selected product
            product_data = sales_df[sales_df["product_id"] == product_id].copy()
            
            if product_data.empty:
                st.warning(f"No data found for product: {selected_product}")
                return
            
            # Initialize and train the forecasting model
            forecasting_model = FruitsVegetablesForecastingModel()
            recommendation = forecasting_model.forecast_inventory_replenishment(
                historical_data=product_data,
                product_id=product_id,
                reorder_level=reorder_level
            )
            
            if recommendation:
                st.success("Inventory recommendation generated!")
                
                # Display recommendation
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Current Stock", value=recommendation["current_stock"])
                with col2:
                    st.metric(label="Reorder Level", value=recommendation["reorder_level"])
                with col3:
                    st.metric(label="Recommended Quantity", value=recommendation["suggested_quantity"])
                
                st.info(f"Product: {recommendation[product_name]}")
                st.info(f"Suggested reorder date: {recommendation[suggested_reorder_date]}")
                st.info(f"Shelf Life: {recommendation[shelf_life_days]} days")
                
                # Show inventory trend
                st.subheader("Inventory Projection")
                
                # Create a projection based on forecast
                forecast_result = forecasting_model.forecast_demand(
                    historical_data=product_data,
                    product_id=product_id,
                    days_ahead=30
                )
                
                if forecast_result is not None:
                    # Calculate projected inventory
                    forecast_result["cumulative_demand"] = forecast_result["predicted_quantity"].cumsum()
                    forecast_result["projected_inventory"] = current_stock - forecast_result["cumulative_demand"]
                    
                    fig = go.Figure()
                    
                    # Projected inventory
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(forecast_result["date"]),
                        y=forecast_result["projected_inventory"],
                        mode="lines",
                        name="Projected Inventory",
                        line=dict(color="orange")
                    ))
                    
                    # Reorder level
                    fig.add_hline(
                        y=reorder_level,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Reorder Level"
                    )
                    
                    fig.update_layout(
                        title="Projected Inventory Level",
                        xaxis_title="Date",
                        yaxis_title="Inventory Level",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate inventory recommendation for the selected product.")


def pricing_optimization(sales_df):
    """
    Provides dynamic pricing optimization based on demand forecasts.
    
    Args:
        sales_df (pd.DataFrame): Sales data
    """
    st.header("Pricing Optimization")
    
    # Product selection
    products = sales_df["product_name"].unique()
    selected_product = st.selectbox("Select a product for pricing optimization:", options=products)
    
    # Get product data
    product_data = sales_df[sales_df["product_name"] == selected_product].copy()
    
    if product_data.empty:
        st.warning(f"No data found for product: {selected_product}")
        return
    
    # Get current price
    current_price = product_data["unit_price"].iloc[0]
    current_price = st.number_input("Current Unit Price ($):", value=float(current_price), min_value=0.01, step=0.01)
    
    # Forecast horizon for pricing
    forecast_days = st.slider("Forecast horizon for pricing (days):", min_value=7, max_value=30, value=14, step=1)
    
    if st.button("Optimize Pricing"):
        with st.spinner("Analyzing optimal pricing..."):
            # Initialize the forecasting model
            forecasting_model = FruitsVegetablesForecastingModel()
            
            # Get pricing recommendation
            pricing_recommendation = forecasting_model.optimize_pricing(
                historical_data=sales_df,
                product_id=product_data["product_id"].iloc[0],
                base_price=current_price,
                days_ahead=forecast_days
            )
            
            if pricing_recommendation:
                st.success("Pricing optimization completed!")
                
                # Display pricing recommendation
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Current Price", value=f"${pricing_recommendation[current_price]:.2f}")
                with col2:
                    st.metric(label="Suggested Price", value=f"${pricing_recommendation[suggested_price]:.2f}")
                with col3:
                    price_diff = pricing_recommendation["suggested_price"] - pricing_recommendation["current_price"]
                    st.metric(label="Price Change", value=f"${price_diff:.2f}")
                
                st.info(f"Product: {pricing_recommendation[product_name]}")
                st.info(f"Pricing Strategy: {pricing_recommendation[pricing_strategy]}")
                st.info(f"Average Forecasted Demand: {pricing_recommendation[avg_forecasted_demand]:.1f} units/day")
                st.info(f"Shelf Life: {pricing_recommendation[shelf_life_days]} days")
                
                # Show pricing analysis
                st.subheader("Pricing Analysis")
                
                # Create a comparison chart
                comparison_data = pd.DataFrame({
                    "Metric": ["Current Price", "Suggested Price"],
                    "Value": [pricing_recommendation["current_price"], pricing_recommendation["suggested_price"]]
                })
                
                fig = px.bar(comparison_data, x="Metric", y="Value", 
                             title="Current vs Suggested Price Comparison",
                             color="Metric",
                             color_discrete_map={"Current Price": "blue", "Suggested Price": "green"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate pricing optimization for the selected product.")


if __name__ == "__main__":
    main()
