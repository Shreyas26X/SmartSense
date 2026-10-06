import pandas as pd
import random
from datetime import datetime, timedelta

# ---------------- NAMES ----------------
names = [
    "John", "Alex", "Rahul", "David", "Sara",
    "Amit", "Sneha", "Ravi", "Pooja", "Michael",
    "Neha", "Priya", "Aryan", "Karan", "Anita"
]

# ---------------- PRODUCTS ----------------
products_map = {
    "sports": ["Football Jersey", "Match Ticket", "Shoes", "Scarf"],
    "healthcare": ["Consultation", "Medicine", "X-Ray", "Blood Test"],
    "business": ["Laptop", "Printer", "Office Chair", "Stationery"],
    "store": ["Milk", "Bread", "Snacks", "Juice"]
}

# ---------------- SEASON FUNCTION ----------------
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

# ---------------- GENERATOR ----------------
def generate_dataset(start_id, category, price_range, qty_range, rows=180):
    data = []

    # Date range
    start_date = datetime(2024, 10, 1)
    end_date   = datetime(2026, 3, 31)

    total_days = (end_date - start_date).days

    for i in range(rows):
        cid = start_id + random.randint(0, 30)
        name = random.choice(names)
        email = f"{name.lower()}{cid}@gmail.com"
        product = random.choice(products_map[category])

        qty = random.randint(*qty_range)
        price = random.randint(*price_range)
        total = qty * price

        # ✅ RANDOM DATE BETWEEN 2025 → MARCH 2026
        random_days = random.randint(0, total_days)
        date = start_date + timedelta(days=random_days)

        season = get_season(date.month)

        data.append([
            cid,
            name,
            email,
            product,
            qty,
            price,
            total,
            date.strftime("%Y-%m-%d"),
            season
        ])

    df = pd.DataFrame(data, columns=[
        "Customer_ID",
        "Customer_Name",
        "Email",
        "Product",
        "Quantity",
        "Price_Per_Item",
        "Total_Price",
        "Purchase_Date",
        "Season"
    ])

    return df


# ---------------- GENERATE FILES ----------------

generate_dataset(1000, "sports", (80, 500), (1, 5), 180).to_csv("sports.csv", index=False)
generate_dataset(2000, "healthcare", (100, 1500), (1, 3), 180).to_csv("healthcare.csv", index=False)
generate_dataset(3000, "business", (50, 800), (2, 15), 180).to_csv("business.csv", index=False)
generate_dataset(4000, "store", (5, 40), (1, 5), 180).to_csv("store.csv", index=False)

print("Datasets generated (2024 → March 2026)")