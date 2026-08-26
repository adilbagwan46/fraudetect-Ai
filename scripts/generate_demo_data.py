from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

TRANSACTION_TYPES = ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN")


def generate_rows(count: int, seed: int) -> list[dict]:
    if count < 30:
        raise ValueError("Generate at least 30 rows so every temporal split is meaningful")
    rng = random.Random(seed)
    customers = [f"C{i:04d}" for i in range(max(20, count // 8))]
    counterparties = [f"M{i:04d}" for i in range(max(12, count // 12))]
    balances = {customer: rng.uniform(800, 50_000) for customer in customers}
    destination_balances = {merchant: rng.uniform(0, 20_000) for merchant in counterparties}
    rows: list[dict] = []

    for index in range(count):
        customer = customers[index % len(customers)]
        counterparty = counterparties[rng.randrange(len(counterparties))]
        step = index // 5
        transaction_type = rng.choices(TRANSACTION_TYPES, weights=(45, 12, 18, 5, 20))[0]
        old_origin = balances[customer]
        amount = round(min(old_origin, max(1.0, rng.lognormvariate(5.2, 0.9))), 2)
        is_fraud = int(index % 47 == 0 and transaction_type in {"TRANSFER", "CASH_OUT"})
        if index % 47 == 0:
            transaction_type = "TRANSFER"
            is_fraud = 1
            amount = round(max(1.0, old_origin * 0.96), 2)
        new_origin = max(0.0, old_origin - amount)
        old_destination = destination_balances[counterparty]
        new_destination = old_destination + amount
        balances[customer] = new_origin if new_origin > 50 else rng.uniform(800, 50_000)
        destination_balances[counterparty] = new_destination
        rows.append(
            {
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "nameOrig": customer,
                "oldbalanceOrg": round(old_origin, 2),
                "newbalanceOrig": round(new_origin, 2),
                "nameDest": counterparty,
                "oldbalanceDest": round(old_destination, 2),
                "newbalanceDest": round(new_destination, 2),
                "isFraud": is_fraud,
                "isFlaggedFraud": int(is_fraud and amount > 200_000),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labeled demo-only PaySim-compatible data"
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/demo_transactions.csv"))
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    rows = generate_rows(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} demo-only transactions at {args.output}")


if __name__ == "__main__":
    main()
