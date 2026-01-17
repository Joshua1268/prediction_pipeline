import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from prediction_model import ProductPredictionModel
from model_trainer import ModelTrainer
import os


def main():
    st.set_page_config(page_title="Assistant IA de Prédictions", layout="wide")

    st.title("🤖 Assistant IA de Prédictions de Produits et Légumes")
    st.markdown("---")

    # Initialiser le modèle
    @st.cache_resource
    def get_trained_model():
        trainer = ModelTrainer()
        return trainer.load_or_train_model()

    predictor = get_trained_model()

    # Charger les données
    data = predictor.load_or_generate_data()

    # Section d'analyse exploratoire (optionnelle, peut être masquée)
    with st.expander("📊 Voir les données historiques"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total des Ventes", f"{data['y'].sum():,.0f}")

        with col2:
            st.metric("Moyenne Quotidienne", f"{data['y'].mean():.2f}")

        with col3:
            st.metric("Écart-type", f"{data['y'].std():.2f}")

        # Graphique des ventes historiques
        fig_historical = px.line(
            data,
            x='ds',
            y='y',
            title='Historique des Ventes',
            labels={'ds': 'Date', 'y': 'Quantité Vendue'}
        )
        fig_historical.update_layout(height=400)
        st.plotly_chart(fig_historical, use_container_width=True)

    # Zone de saisie pour les questions
    st.header("💬 Posez votre question à l'IA")

    # Champ de texte pour la question personnalisée
    user_question = st.text_input("Posez une question sur les prédictions de produits/légumes :",
                                  placeholder="Ex: Quelle sera la demande pour les tomates la semaine prochaine ?")

    # Bouton pour soumettre la question
    if st.button("Envoyer la question"):
        if user_question:
            with st.spinner("L'IA réfléchit à votre question..."):
                # Charger les données et faire une prédiction de base
                data = predictor.load_or_generate_data()

                # Faire une prédiction par défaut pour 30 jours
                forecast_result = predictor.get_prediction_for_product("Produit", 30)
                forecast_df = pd.DataFrame(forecast_result['predictions'])

                # Extraire les insights
                insights = predictor.get_insights()

                # Traiter la question de l'utilisateur
                response = process_user_question(user_question, forecast_df, insights, data)

                # Afficher la réponse
                st.success(response)
        else:
            st.warning("Veuillez poser une question avant de cliquer sur envoyer.")

    # Questions fréquentes
    st.subheader("🔍 Questions fréquentes")

    frequent_questions = [
        "Quelle est la prévision pour les prochains jours ?",
        "Quand les ventes seront-elles les plus élevées ?",
        "Quelle est la tendance générale ?",
        "Quelle est la variation attendue ?",
        "Quels produits auront une forte demande ?"
    ]

    selected_question = st.selectbox("Ou choisissez une question fréquente :", [""] + frequent_questions)

    if selected_question:
        with st.spinner("L'IA réfléchit à votre question..."):
            # Charger les données et faire une prédiction
            data = predictor.load_or_generate_data()
            forecast_result = predictor.get_prediction_for_product("Produit", 30)
            forecast_df = pd.DataFrame(forecast_result['predictions'])
            insights = predictor.get_insights()

            response = process_user_question(selected_question, forecast_df, insights, data)
            st.success(response)

    # Section de prédiction (optionnelle, peut être masquée)
    with st.expander("📈 Voir les prédictions détaillées"):
        # Sidebar pour les contrôles
        st.sidebar.header("Paramètres de Prévision")

        # Sélection du produit
        product_input = st.sidebar.text_input("Nom du produit/légume", "Tomate")

        # Nombre de jours à prédire
        forecast_days = st.sidebar.slider("Nombre de jours à prédire", 7, 90, 30)

        # Bouton pour réentraîner le modèle
        if st.sidebar.button("🔄 Réentraîner le modèle"):
            with st.spinner("Réentraînement du modèle..."):
                trainer = ModelTrainer()
                predictor = trainer.retrain_model()
                st.success("Modèle réentraîné avec succès!")

        # Obtenir les prédictions
        with st.spinner("Calcul des prédictions..."):
            forecast_result = predictor.get_prediction_for_product(product_input, forecast_days)
            forecast_df = pd.DataFrame(forecast_result['predictions'])
            lower_bound = pd.DataFrame(forecast_result['lower_bound'])
            upper_bound = pd.DataFrame(forecast_result['upper_bound'])

        # Afficher les prédictions
        fig_forecast = make_subplots(specs=[[{"secondary_y": False}]])

        # Données historiques
        fig_forecast.add_trace(
            go.Scatter(
                x=data['ds'],
                y=data['y'],
                mode='lines',
                name='Historique',
                line=dict(color='blue')
            )
        )

        # Prédictions
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat1'],
                mode='lines',
                name='Prédictions',
                line=dict(color='red', dash='dash')
            )
        )

        # Intervalles de confiance
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=upper_bound['yhat_upper'],
                mode='lines',
                name='Intervalle Supérieur',
                line=dict(width=0),
                showlegend=False
            )
        )

        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df['ds'],
                y=lower_bound['yhat_lower'],
                mode='lines',
                name='Intervalle Inférieur',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.1)',
                showlegend=False
            )
        )

        fig_forecast.update_layout(
            title=f'Prédictions pour {product_input}',
            xaxis_title='Date',
            yaxis_title='Quantité Vendue',
            height=500
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Afficher les prédictions sous forme de tableau
        st.subheader("Tableau des Prédictions")
        forecast_display = forecast_df.copy()
        forecast_display['Date'] = pd.to_datetime(forecast_display['ds']).dt.date
        forecast_display['Prédiction'] = forecast_display['yhat1'].round(2)
        forecast_table = forecast_display[['Date', 'Prédiction']].set_index('Date')
        st.dataframe(forecast_table, use_container_width=True)


def process_user_question(question, forecast_df, insights, data):
    """Traite la question de l'utilisateur et fournit une réponse pertinente"""
    question_lower = question.lower()

    # Réponses intelligentes basées sur la question posée
    if "prévision" in question_lower or "demain" in question_lower or "semaine" in question_lower or "mois" in question_lower:
        avg_forecast = forecast_df['yhat1'].mean()
        return f"La prévision moyenne pour les prochains jours est de {avg_forecast:.2f} unités par jour."

    elif "plus élevé" in question_lower or "maximum" in question_lower or "pic" in question_lower:
        max_idx = forecast_df['yhat1'].idxmax()
        max_date = forecast_df.loc[max_idx, 'ds']
        max_value = forecast_df.loc[max_idx, 'yhat1']
        return f"Les ventes devraient atteindre leur pic le {max_date.strftime('%Y-%m-%d')} avec environ {max_value:.2f} unités vendues."

    elif "tendance" in question_lower or "évolution" in question_lower:
        trend = insights['trend']
        if trend == "hausse":
            return "La tendance générale indique une augmentation des ventes. Les prévisions montrent une croissance positive."
        else:
            return "La tendance générale indique une diminution des ventes. Les prévisions montrent une baisse potentielle."

    elif "variation" in question_lower or "volatilité" in question_lower:
        std_dev = forecast_df['yhat1'].std()
        return f"La variation attendue (écart-type) des prédictions est de {std_dev:.2f} unités, ce qui indique un niveau de volatilité modéré."

    elif "produit" in question_lower or "demande" in question_lower:
        top_products = data.groupby('ds')['y'].sum().nlargest(5)
        if not top_products.empty:
            peak_date = top_products.index[0]
            peak_value = top_products.iloc[0]
            return f"La demande sera particulièrement forte le {peak_date.strftime('%Y-%m-%d')} avec {peak_value:.2f} unités vendues."

    else:
        # Réponse par défaut si la question n'est pas reconnue
        avg_forecast = forecast_df['yhat1'].mean()
        trend = insights['trend']
        return f"Sur la base des données disponibles, la prévision moyenne est de {avg_forecast:.2f} unités par jour. La tendance est actuellement en {trend}."


if __name__ == "__main__":
    main()