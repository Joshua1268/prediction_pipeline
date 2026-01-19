#!/usr/bin/env python3

import csv
import random
from datetime import datetime, timedelta
import os
from typing import List, Tuple


class FruitsVegetablesDataGenerator:
    def __init__(self, start_date: str = "2023-01-01", end_date: str = "2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.dates = []
        current_date = self.start_date
        while current_date <= self.end_date:
            self.dates.append(current_date)
            current_date += timedelta(days=1)

        # Catégories adaptées
        self.categories = [
            'Fruits Bio', 'Légumes Bio', 'Fruits Exotiques', 'Agrumes',
            'Fruits Rouges', 'Légumes Feuilles', 'Tubercules et Racines',
            'Légumes d\'Été', 'Légumes de Saison des Pluies'
        ]

        self.fruit_names = [
            'Bananes Douces', 'Bananes Plantains', 'Oranges', 'Mangues', 'Ananas',
            'Papayes', 'Citrons Verts', 'Pastèques', 'Goyaves', 'Avocats',
            'Pommes', 'Raisins', 'Noix de Coco', 'Mandarines', 'Fruit de la Passion'
        ]

        self.vegetable_names = [
            'Épinards', 'Laitue', 'Gombo', 'Piment', 'Aubergines', 'Tomates',
            'Oignons', 'Ail', 'Poireaux', 'Carottes', 'Choux', 'Haricots Verts',
            'Concombre', 'Poivrons', 'Ignames', 'Manioc', 'Patates Douces'
        ]

    def generate_products(self, num_fruits: int = 20, num_vegetables: int = 25) -> List[dict]:
        """Générer des produits avec des prix en XOF"""
        products = []

        # Générer les fruits
        for i in range(num_fruits):
            fruit = random.choice(self.fruit_names)
            # Prix en XOF (ex: entre 500 et 3500 XOF le kg ou l'unité)
            price = round(random.uniform(500, 3500), -1) 
            cost = round(price * random.uniform(0.5, 0.7), -1)
            fruit_product = {
                'product_id': f'FRUIT{i+1:03d}',
                'product_name': f'{fruit}',
                'category': random.choice(['Fruits Bio', 'Fruits Exotiques', 'Agrumes']),
                'price_xof': int(price),
                'cost_xof': int(cost),
                'weight_kg': round(random.uniform(0.5, 2.0), 2),
                'seasonal_factor': random.uniform(0.7, 1.6),
                'trend_factor': random.uniform(0.9, 1.2),
                'shelf_life_days': random.randint(3, 10)
            }
            products.append(fruit_product)

        # Générer les légumes
        for i in range(num_vegetables):
            veg = random.choice(self.vegetable_names)
            # Prix en XOF (ex: entre 300 et 2500 XOF)
            price = round(random.uniform(300, 2500), -1)
            cost = round(price * random.uniform(0.5, 0.7), -1)
            veg_product = {
                'product_id': f'VEG{i+1:03d}',
                'product_name': f'{veg}',
                'category': random.choice(['Légumes Feuilles', 'Tubercules et Racines', 'Légumes Bio']),
                'price_xof': int(price),
                'cost_xof': int(cost),
                'weight_kg': round(random.uniform(0.2, 3.0), 2),
                'seasonal_factor': random.uniform(0.7, 1.6),
                'trend_factor': random.uniform(0.9, 1.2),
                'shelf_life_days': random.randint(4, 15)
            }
            products.append(veg_product)

        return products

    def get_season(self, month: int) -> str:
        """Déterminer la saison selon le climat ivoirien"""
        # Simplification : Saison des pluies (Mai à Octobre), Saison Sèche (Novembre à Avril)
        if 5 <= month <= 10:
            return 'Saison des Pluies'
        else:
            return 'Saison Sèche'

    def generate_sales_data(self, products: List[dict]) -> List[dict]:
        """Générer les ventes avec impact des saisons locales"""
        sales_data = []

        for product in products:
            for date in self.dates:
                base_demand = max(0, random.gauss(40, 20))
                month_factor = 1.0
                season = self.get_season(date.month)

                # Ajustement saisonnier
                if season == 'Saison des Pluies':
                    month_factor = 1.3  # Plus d'abondance pour certains produits
                else:
                    month_factor = 1.0

                # Effet week-end (marché plus actif)
                day_factor = 1.25 if date.weekday() >= 5 else 1.0

                demand = base_demand * month_factor * day_factor * product['seasonal_factor']
                actual_sales = max(0, int(demand + random.gauss(0, 5)))

                revenue = actual_sales * product['price_xof']
                cost = actual_sales * product['cost_xof']
                profit = revenue - cost

                sales_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'quantite_vendue': actual_sales,
                    'prix_unitaire_xof': product['price_xof'],
                    'chiffre_affaires_xof': revenue,
                    'cout_total_xof': cost,
                    'profit_xof': profit,
                    'saison': season,
                    'est_weekend': 1 if date.weekday() >= 5 else 0
                })

        return sales_data

    def generate_inventory_data(self, products: List[dict]) -> List[dict]:
        """Générer l'état des stocks"""
        inventory_data = []
        for product in products:
            for date in self.dates:
                inventory_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': product['product_id'],
                    'stock_actuel': random.randint(50, 300),
                    'seuil_reapprovisionnement': 30,
                    'duree_conservation_jours': product['shelf_life_days']
                })
        return inventory_data

    def generate_all_data(self) -> Tuple[List[dict], List[dict]]:
        print("Génération des données en cours (Devise: XOF, Saisons: CI)...")
        products = self.generate_products()
        sales = self.generate_sales_data(products)
        inventory = self.generate_inventory_data(products)
        return sales, inventory


def save_to_csv(sales: List[dict], inventory: List[dict], output_dir: str = "data_ci"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde Ventes
    with open(os.path.join(output_dir, "ventes_ci.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sales[0].keys())
        writer.writeheader()
        writer.writerows(sales)

    # Sauvegarde Inventaire
    with open(os.path.join(output_dir, "inventaire_ci.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)
    
    print(f"Fichiers enregistrés dans le dossier : {output_dir}")


if __name__ == "__main__":
    generator = FruitsVegetablesDataGenerator()
    sales, inventory = generator.generate_all_data()
    save_to_csv(sales, inventory)
    
    print("\nExemple d'une vente :")
    print(sales[0])