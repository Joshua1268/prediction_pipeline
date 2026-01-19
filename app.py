#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_data import FruitsVegetablesDataGenerator
from prediction_model import FruitsVegetablesForecastingModel


def main():
    
    st.set_page_config(
        page_title="Plateforme d'Analyse Prédictive des Fruits & Légumes",
        layout="wide"
    )

    st.title("Plateforme d'Analyse Prédictive des Fruits & Légumes")
    st.markdown("""
    Outil d'analyse avancé pour la prévision de la demande, l'optimisation des stocks et la gestion des produits périssables pour les fruits et légumes.
    """)

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Sélectionnez une section:",
                                   ["Vue d'ensemble du tableau de bord",
                                    "Exploration des données",
                                    "Prévision des produits",
                                    "Planification des stocks",
                                    "Optimisation des prix"])
    
    data_dir = "data"
    sales_data_path = os.path.join(data_dir, "fv_sales_data.csv")
    inventory_data_path = os.path.join(data_dir, "fv_inventory_data.csv")
    customer_data_path = os.path.join(data_dir, "fv_customer_data.csv")

    if not all(os.path.exists(path) for path in [sales_data_path, inventory_data_path, customer_data_path]):
        with st.spinner("Génération des données synthétiques de fruits et légumes..."):
            st.info("Génération de données synthétiques de fruits et légumes pour la démonstration...")
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

            st.success("Données de fruits et légumes générées avec succès !")
    else:
        # Load existing data with loading indicator
        with st.spinner("Chargement des données..."):
            sales_df = pd.read_csv(sales_data_path)
            inventory_df = pd.read_csv(inventory_data_path)
            customer_df = pd.read_csv(customer_data_path)

            # Convert date columns to datetime if they exist
            if "date" in sales_df.columns:
                sales_df["date"] = pd.to_datetime(sales_df["date"])
            if "date" in inventory_df.columns:
                inventory_df["date"] = pd.to_datetime(inventory_df["date"])
            if "date" in customer_df.columns:
                customer_df["date"] = pd.to_datetime(customer_df["date"])

    if app_mode == "Vue d'ensemble du tableau de bord":
        dashboard_overview(sales_df, inventory_df, customer_df)
    elif app_mode == "Exploration des données":
        data_exploration(sales_df, inventory_df, customer_df)
    elif app_mode == "Prévision des produits":
        product_forecasting(sales_df)
    elif app_mode == "Planification des stocks":
        inventory_planning(sales_df, inventory_df)
    elif app_mode == "Optimisation des prix":
        pricing_optimization(sales_df)


def dashboard_overview(sales_df, inventory_df, customer_df):
    """
    Displays the main dashboard with key metrics and visualizations.

    Args:
        sales_df (pd.DataFrame): Sales data
        inventory_df (pd.DataFrame): Inventory data
        customer_df (pd.DataFrame): Customer data
    """
    st.header("Vue d'ensemble du tableau de bord")

    # Ensure date column is properly converted to datetime
    sales_df["date"] = pd.to_datetime(sales_df["date"])

    # Enhanced KPI cards with better styling
    total_revenue = sales_df["revenue"].sum()
    total_profit = sales_df["profit"].sum()
    avg_daily_sales = sales_df.groupby(sales_df["date"].dt.date)["quantity_sold"].sum().mean()
    top_category = sales_df["category"].mode()[0] if not sales_df.empty else "N/A"

    # Create KPI cards with enhanced styling
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin: 0; font-size: 1.2em;">Revenu Total</h3>
            <p style="margin: 10px 0 0 0; font-size: 1.8em; font-weight: bold;">XOF{total_revenue:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin: 0; font-size: 1.2em;">Profit Total</h3>
            <p style="margin: 10px 0 0 0; font-size: 1.8em; font-weight: bold;">XOF{total_profit:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin: 0; font-size: 1.2em;">Ventes Journalières Moyennes</h3>
            <p style="margin: 10px 0 0 0; font-size: 1.8em; font-weight: bold;">{avg_daily_sales:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin: 0; font-size: 1.2em;">Catégorie Principale</h3>
            <p style="margin: 10px 0 0 0; font-size: 1.8em; font-weight: bold;">{top_category}</p>
        </div>
        """, unsafe_allow_html=True)

    # Additional KPIs
    col5, col6, col7, col8 = st.columns(4)

    avg_order_value = sales_df["revenue"].mean()
    total_products = sales_df["product_id"].nunique()
    avg_profit_margin = (sales_df["profit"] / sales_df["revenue"] * 100).mean() if sales_df["revenue"].sum() > 0 else 0
    peak_season = sales_df.groupby("season")["revenue"].sum().idxmax() if not sales_df.empty and "season" in sales_df.columns else "N/A"

    with col5:
        st.metric(label="Valeur Moyenne des Commandes", value=f"XOF{avg_order_value:.2f}")
    with col6:
        st.metric(label="Nombre de Produits", value=total_products)
    with col7:
        st.metric(label="Marge de Profit Moyenne (%)", value=f"{avg_profit_margin:.1f}%")
    with col8:
        st.metric(label="Saison de Pointe", value=peak_season)

    # Time series chart of sales
    st.subheader("Tendance des Ventes dans le Temps")
    daily_sales = sales_df.groupby(sales_df["date"].dt.date)["revenue"].sum().reset_index()
    daily_sales.columns = ["Date", "Revenue"]

    fig = px.line(daily_sales, x="Date", y="Revenue", title="Tendance du Revenu Quotidien")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Category distribution
    st.subheader("Ventes par Catégorie")
    category_sales = sales_df.groupby("category")["revenue"].sum().reset_index()

    fig = px.pie(category_sales, values="revenue", names="category", title="Distribution du Revenu par Catégorie")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Top performing products
    st.subheader("Produits Performants")
    top_products = sales_df.groupby("product_name")["revenue"].sum().nlargest(10).reset_index()

    fig = px.bar(top_products, x="revenue", y="product_name", orientation="h",
                 title="Top 10 des Produits par Revenu")
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


def data_exploration(sales_df, inventory_df, customer_df):
    """
    Provides interactive exploration of sales, inventory, and customer data.

    Args:
        sales_df (pd.DataFrame): Sales data
        inventory_df (pd.DataFrame): Inventory data
        customer_df (pd.DataFrame): Customer data
    """
    st.header("Exploration des Données")

    # Enhanced KPIs for data exploration
    total_records = len(sales_df)
    date_range_str = f"{sales_df['date'].min().strftime('%Y-%m-%d')} au {sales_df['date'].max().strftime('%Y-%m-%d')}"
    unique_products = sales_df['product_id'].nunique()
    unique_categories = sales_df['category'].nunique()

    # Display enhanced KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #007bff;
        ">
            <h4 style="margin: 0; color: #007bff;">Total Enregistrements</h4>
            <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{total_records:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #28a745;
        ">
            <h4 style="margin: 0; color: #28a745;">Période</h4>
            <p style="font-size: 1.2em; font-weight: bold; margin: 5px 0 0 0;">{date_range_str}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #ffc107;
        ">
            <h4 style="margin: 0; color: #ffc107;">Produits Uniques</h4>
            <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{unique_products}</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div style="
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #dc3545;
        ">
            <h4 style="margin: 0; color: #dc3545;">Catégories</h4>
            <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{unique_categories}</p>
        </div>
        """, unsafe_allow_html=True)

    # Date range selector
    min_date = pd.to_datetime(sales_df["date"]).min().date()
    max_date = pd.to_datetime(sales_df["date"]).max().date()

    date_range = st.date_input(
        "Sélectionnez la plage de dates:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (pd.to_datetime(sales_df["date"]).dt.date >= start_date) & (pd.to_datetime(sales_df["date"]).dt.date <= end_date)
        filtered_sales = sales_df.loc[mask]
    else:
        filtered_sales = sales_df

    # Sales metrics by category
    st.subheader("Indicateurs de Vente par Catégorie")
    category_metrics = filtered_sales.groupby("category").agg({
        "quantity_sold": "sum",
        "revenue": "sum",
        "profit": "sum"
    }).reset_index()

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(category_metrics, x="category", y="quantity_sold",
                     title="Quantité Totale Vendue par Catégorie",
                     color="category",
                     template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(category_metrics, x="category", y="revenue",
                     title="Revenu Total par Catégorie",
                     color="category",
                     template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = px.bar(category_metrics, x="category", y="profit",
                     title="Profit Total par Catégorie",
                     color="category",
                     template="plotly_white")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal analysis if season column exists
    if "season" in filtered_sales.columns:
        st.subheader("Analyse Saisonnière des Ventes")
        seasonal_sales = filtered_sales.groupby("season").agg({
            "quantity_sold": "sum",
            "revenue": "sum"
        }).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(seasonal_sales, x="season", y="quantity_sold",
                         title="Quantité Totale Vendue par Saison",
                         color="season",
                         template="plotly_white")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(seasonal_sales, x="season", y="revenue",
                         title="Revenu Total par Saison",
                         color="season",
                         template="plotly_white")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Price vs Quantity scatter plot
    st.subheader("Relation Prix vs Quantité")
    fig = px.scatter(filtered_sales, x="unit_price", y="quantity_sold",
                     color="category", hover_data=["product_name"],
                     title="Prix Unitaire vs Quantité Vendue",
                     template="plotly_white")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def product_forecasting(sales_df):
    """
    Provides forecasting capabilities for individual products.

    Args:
        sales_df (pd.DataFrame): Sales data
    """
    st.header("Prévision des Produits")

    # Product selection
    products = sales_df["product_name"].unique()
    selected_product = st.selectbox("Sélectionnez un produit à prévoir:", options=products)

    # Forecast horizon
    forecast_days = st.slider("Sélectionnez l'horizon de prévision (jours):", min_value=7, max_value=90, value=30, step=7)

    if st.button("Générer la Prévision"):
        with st.spinner("Entraînement du modèle et génération de la prévision... Cela peut prendre quelques instants."):
            # Prepare data for the selected product
            product_data = sales_df[sales_df["product_name"] == selected_product].copy()

            if product_data.empty:
                st.warning(f"Aucune donnée trouvée pour le produit: {selected_product}")
                return

            # Initialize and train the forecasting model
            forecasting_model = FruitsVegetablesForecastingModel()
            forecast_result = forecasting_model.forecast_demand(
                historical_data=product_data,
                product_id=product_data["product_id"].iloc[0],
                days_ahead=forecast_days
            )

            if forecast_result is not None:
                st.success("Prévision générée avec succès !")

                # Enhanced KPIs for the selected product
                col1, col2, col3, col4 = st.columns(4)

                avg_historical_sales = product_data["quantity_sold"].mean()
                total_historical_revenue = product_data["revenue"].sum()
                shelf_life = product_data["shelf_life_days"].iloc[0] if "shelf_life_days" in product_data.columns else "N/A"
                avg_price = product_data["unit_price"].mean()

                with col1:
                    st.markdown(f"""
                    <div style="
                        background: #e3f2fd;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #2196f3;
                    ">
                        <h4 style="margin: 0; color: #2196f3;">Ventes Moyennes</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{avg_historical_sales:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="
                        background: #f3e5f5;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #9c27b0;
                    ">
                        <h4 style="margin: 0; color: #9c27b0;">Revenu Total</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">XOF{total_historical_revenue:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div style="
                        background: #fff3e0;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #ff9800;
                    ">
                        <h4 style="margin: 0; color: #ff9800;">Durée de Vie</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{shelf_life} jours</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div style="
                        background: #e8f5e8;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #4caf50;
                    ">
                        <h4 style="margin: 0; color: #4caf50;">Prix Moyen</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">XOF{avg_price:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show forecast data
                st.subheader(f"Prévision pour {selected_product}")

                # Plot forecast
                fig = go.Figure()

                # Historical data
                historical_dates = pd.to_datetime(product_data["date"]).dt.date
                historical_values = product_data["quantity_sold"]

                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_values,
                    mode="lines",
                    name="Ventes Historiques",
                    line=dict(color="#1f77b4", width=2)
                ))

                # Forecast data
                forecast_dates = pd.to_datetime(forecast_result["date"])
                forecast_values = forecast_result["predicted_quantity"]

                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_values,
                    mode="lines",
                    name="Prévision",
                    line=dict(color="#ff7f0e", width=2, dash="dash")
                ))

                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result["confidence_interval_upper"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(255, 127, 14, 0.2)",
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_result["confidence_interval_lower"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(255, 127, 14, 0.2)",
                    name="Intervalle de Confiance"
                ))

                fig.update_layout(
                    title=f"Prévision de Demande pour {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Quantité Vendue",
                    height=500,
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show forecast statistics
                st.subheader("Statistiques de Prévision")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Quantité Prévue (Totale)",
                        value=f"{forecast_result['predicted_quantity'].sum():,}"
                    )
                with col2:
                    st.metric(
                        label="Quantité Moyenne/Jour",
                        value=f"{forecast_result['predicted_quantity'].mean():.1f}"
                    )
                with col3:
                    st.metric(
                        label="Confiance Moyenne",
                        value=f"±{(forecast_result['confidence_interval_upper'] - forecast_result['confidence_interval_lower']).mean()/2:.1f}"
                    )

                # Show forecast table
                st.subheader("Tableau de Prévision")
                st.dataframe(forecast_result.style.format({
                    "predicted_quantity": "{:.0f}",
                    "confidence_interval_lower": "{:.0f}",
                    "confidence_interval_upper": "{:.0f}"
                }))
            else:
                st.warning("Impossible de générer une prévision pour le produit sélectionné.")


def inventory_planning(sales_df, inventory_df):
    """
    Provides inventory planning and optimization recommendations.

    Args:
        sales_df (pd.DataFrame): Sales data
        inventory_df (pd.DataFrame): Inventory data
    """
    st.header("Planification des Stocks")

    # Product selection
    products = sales_df["product_name"].unique()
    selected_product = st.selectbox("Sélectionnez un produit pour la planification des stocks:", options=products)

    # Get product ID
    product_id = sales_df[sales_df["product_name"] == selected_product]["product_id"].iloc[0]

    # Current stock level
    current_stock = inventory_df[inventory_df["product_id"] == product_id]["current_stock"].iloc[-1] if not inventory_df[inventory_df["product_id"] == product_id].empty else 100
    current_stock = st.number_input("Niveau Actuel de Stock:", value=int(current_stock), min_value=0)

    # Reorder level
    reorder_level = st.number_input("Niveau de Réapprovisionnement:", value=30, min_value=1)

    if st.button("Générer les Recommandations de Stock"):
        with st.spinner("Analyse des besoins en stock et calcul des recommandations... Cela peut prendre quelques instants."):
            # Prepare data for the selected product
            product_data = sales_df[sales_df["product_id"] == product_id].copy()

            if product_data.empty:
                st.warning(f"Aucune donnée trouvée pour le produit: {selected_product}")
                return

            # Initialize and train the forecasting model
            forecasting_model = FruitsVegetablesForecastingModel()
            recommendation = forecasting_model.forecast_inventory_replenishment(
                historical_data=product_data,
                product_id=product_id,
                reorder_level=reorder_level
            )

            if recommendation:
                st.success("Recommandation de stock générée !")

                # Enhanced KPI cards for inventory
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div style="
                        background: #e8f5e8;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #4caf50;
                    ">
                        <h4 style="margin: 0; color: #4caf50;">Stock Actuel</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{recommendation['current_stock']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="
                        background: #ffebee;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #f44336;
                    ">
                        <h4 style="margin: 0; color: #f44336;">Niveau de Réappro.</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{recommendation['reorder_level']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div style="
                        background: #e3f2fd;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #2196f3;
                    ">
                        <h4 style="margin: 0; color: #2196f3;">Qté Recommandée</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{recommendation['suggested_quantity']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div style="
                        background: #fff3e0;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #ff9800;
                    ">
                        <h4 style="margin: 0; color: #ff9800;">Durée de Vie</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{recommendation['shelf_life_days']} jours</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional info
                st.info(f"**Produit:** {recommendation['product_name']}")
                st.info(f"**Date de réapprovisionnement suggérée:** {recommendation['suggested_reorder_date']}")
                st.info(f"**Délai de Conservation:** {recommendation['shelf_life_days']} jours")

                # Show inventory trend
                st.subheader("Projection des Stocks")

                # Create a projection based on forecast
                with st.spinner("Génération de la projection d'inventaire..."):
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
                        mode="lines+markers",
                        name="Stock Projeté",
                        line=dict(color="#4caf50", width=3),
                        marker=dict(size=6)
                    ))

                    # Reorder level
                    fig.add_hline(
                        y=reorder_level,
                        line_dash="dash",
                        line_color="#f44336",
                        line_width=2,
                        annotation_text="Niveau de Réapprovisionnement",
                        annotation_position="top left"
                    )

                    # Zero stock line
                    fig.add_hline(
                        y=0,
                        line_dash="solid",
                        line_color="#000000",
                        line_width=1
                    )

                    fig.update_layout(
                        title="Projection du Niveau des Stocks",
                        xaxis_title="Date",
                        yaxis_title="Niveau de Stock",
                        height=500,
                        template="plotly_white",
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Impossible de générer une recommandation de stock pour le produit sélectionné.")


def pricing_optimization(sales_df):
    """
    Provides dynamic pricing optimization based on demand forecasts.

    Args:
        sales_df (pd.DataFrame): Sales data
    """
    st.header("Optimisation des Prix")

    # Product selection
    products = sales_df["product_name"].unique()
    selected_product = st.selectbox("Sélectionnez un produit pour l'optimisation des prix:", options=products)

    # Get product data
    product_data = sales_df[sales_df["product_name"] == selected_product].copy()

    if product_data.empty:
        st.warning(f"Aucune donnée trouvée pour le produit: {selected_product}")
        return

    # Get current price
    current_price = product_data["unit_price"].iloc[0]
    current_price = st.number_input("Prix Unitaire Actuel (XOF):", value=float(current_price), min_value=0.01, step=0.01)

    # Forecast horizon for pricing
    forecast_days = st.slider("Horizon de prévision pour les prix (jours):", min_value=7, max_value=30, value=14, step=1)

    if st.button("Optimiser les Prix"):
        with st.spinner("Analyse des prix optimaux et calcul des recommandations... Cela peut prendre quelques instants."):
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
                st.success("Optimisation des prix terminée !")

                # Enhanced KPI cards for pricing
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div style="
                        background: #e3f2fd;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #2196f3;
                    ">
                        <h4 style="margin: 0; color: #2196f3;">Prix Actuel</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">XOF{pricing_recommendation['current_price']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="
                        background: #f3e5f5;
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid #9c27b0;
                    ">
                        <h4 style="margin: 0; color: #9c27b0;">Prix Suggéré</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">XOF{pricing_recommendation['suggested_price']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                price_diff = pricing_recommendation["suggested_price"] - pricing_recommendation["current_price"]
                price_change_pct = (price_diff / pricing_recommendation["current_price"]) * 100

                with col3:
                    bg_color = "#e8f5e8" if price_diff >= 0 else "#ffebee"
                    color = "#4caf50" if price_diff >= 0 else "#f44336"
                    symbol = "+" if price_diff >= 0 else ""
                    st.markdown(f"""
                    <div style="
                        background: {bg_color};
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid {color};
                    ">
                        <h4 style="margin: 0; color: {color};">Variation de Prix</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{symbol}XOF{price_diff:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    bg_color = "#e8f5e8" if price_change_pct >= 0 else "#ffebee"
                    color = "#4caf50" if price_change_pct >= 0 else "#f44336"
                    symbol = "+" if price_change_pct >= 0 else ""
                    st.markdown(f"""
                    <div style="
                        background: {bg_color};
                        padding: 15px;
                        border-radius: 8px;
                        border-left: 5px solid {color};
                    ">
                        <h4 style="margin: 0; color: {color};">Variation (%)</h4>
                        <p style="font-size: 1.5em; font-weight: bold; margin: 5px 0 0 0;">{symbol}{price_change_pct:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional info
                st.info(f"**Produit:** {pricing_recommendation['product_name']}")
                st.info(f"**Stratégie de Prix:** {pricing_recommendation['pricing_strategy']}")
                st.info(f"**Demande Moyenne Prévue:** {pricing_recommendation['avg_forecasted_demand']:.1f} unités/jour")
                st.info(f"**Délai de Conservation:** {pricing_recommendation['shelf_life_days']} jours")

                # Show pricing analysis
                st.subheader("Analyse des Prix")

                # Create a comparison chart
                comparison_data = pd.DataFrame({
                    "Métrique": ["Prix Actuel", "Prix Suggéré"],
                    "Valeur": [pricing_recommendation["current_price"], pricing_recommendation["suggested_price"]]
                })

                fig = px.bar(comparison_data, x="Métrique", y="Valeur",
                             title="Comparaison Prix Actuel vs Prix Suggéré",
                             color="Métrique",
                             color_discrete_map={"Prix Actuel": "#2196f3", "Prix Suggéré": "#9c27b0"},
                             template="plotly_white")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Impossible de générer une optimisation des prix pour le produit sélectionné.")


if __name__ == "__main__":
    main()
