import csv
from datetime import datetime

def load_positions_from_csv(path: str):
    """
    CSV format (flexible): symbol,quantity,average_price,price
    """
    positions = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append({
                "symbol": row["symbol"],
                "quantity": float(row["quantity"]),
                "average_price": float(row["average_price"]),
                "price": float(row.get("price", row["average_price"])),
            })
    return positions

def load_orders_from_csv(path: str):
    """
    CSV format: date,side,symbol,qty,price
    """
    orders = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row["date"]
            # normalize to ISO
            try:
                dt_iso = datetime.fromisoformat(dt).isoformat()
            except ValueError:
                dt_iso = datetime.strptime(dt, "%Y-%m-%d").isoformat()
            orders.append({
                "created_at": dt_iso,
                "side": row["side"].lower(),
                "symbol": row["symbol"],
                "quantity": float(row["qty"]),
                "price": float(row["price"]),
                "state": "filled",
            })
    return orders
