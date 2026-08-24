import json
import random
from datetime import datetime, timedelta

from services.policy_loader import (
    load_policy_rules,
    get_seniority_tier,
    get_regional_multiplier,
    get_spending_limit,
)
from services.audit_engine import audit_expense

USD_TO_INR = 83.0          
EXAMPLES_PER_SCENARIO = 40  
RANDOM_SEED = 42

SYSTEM_PROMPT = "You are a corporate expense auditor."

EMPLOYEE_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L7+"]

TIER1_CITIES = ["New York", "San Francisco", "London", "Tokyo", "Zurich"]
TIER2_CITIES = ["Chicago", "Paris", "Sydney", "Singapore", "Amsterdam"]
TIER3_CITIES = ["Mumbai", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Delhi"]
ALL_CITIES = TIER1_CITIES + TIER2_CITIES + TIER3_CITIES

CATEGORY_DATA = {
    "meals_daily": {
        "purpose": "Daily meals while travelling for client meetings",
        "merchants": ["Saravana Bhavan", "Spice Garden", "Curry Leaf Diner"],
        "items": ["Veg Thali", "Masala Dosa", "Butter Naan", "Dal Makhani"],
    },
    "meals_single_lunch": {
        "purpose": "Lunch during a work trip",
        "merchants": ["Cafe Mocha", "The Sandwich Co", "Green Bowl Cafe"],
        "items": ["Sandwich Combo", "Salad Bowl", "Iced Tea"],
    },
    "business_dinner_per_head": {
        "purpose": "Business dinner with the client team",
        "merchants": ["The Olive Bistro", "Spice Route Restaurant", "Urban Grill"],
        "items": ["Tasting Menu", "Grilled Platter", "Soft Drinks"],
    },
    "local_transport_daily": {
        "purpose": "Local transport for site visits",
        "merchants": ["Metro Card Recharge", "City Bus Pass Counter"],
        "items": ["Metro fare top-up", "Bus pass"],
    },
    "taxi_rideshare_trip": {
        "purpose": "Taxi to client office",
        "merchants": ["Ola Cabs", "Uber India", "Rapido"],
        "items": ["Cab ride"],
    },
    "accommodation_domestic_night": {
        "purpose": "Hotel stay for domestic business trip",
        "merchants": ["Hotel Grand Plaza", "City Inn", "Comfort Suites"],
        "items": ["Room charge - 1 night"],
    },
    "accommodation_international_night": {
        "purpose": "International hotel stay for business trip",
        "merchants": ["Marriott Downtown", "Hilton Garden Inn", "The Grand Hotel"],
        "items": ["Room charge - 1 night (international)"],
    },
    "conference_registration": {
        "purpose": "Conference registration fee for industry summit",
        "merchants": ["TechConf 2026", "FinSummit Registration", "DevCon Tickets"],
        "items": ["Conference registration fee"],
    },
    "client_entertainment_event": {
        "purpose": "Client entertainment event after the meeting",
        "merchants": ["Skybar Lounge", "The Terrace Club", "Riverside Lounge"],
        "items": ["Client entertainment package"],
    },
    "phone_data_monthly": {
        "purpose": "Monthly phone and data plan for work",
        "merchants": ["Airtel", "Jio", "Vodafone Idea"],
        "items": ["Monthly data plan"],
    },
    "software_subscriptions_monthly": {
        "purpose": "Monthly software subscription for work use",
        "merchants": ["Adobe Creative Cloud", "Notion", "Figma"],
        "items": ["Monthly subscription"],
    },
}

ALCOHOL_ITEMS = ["Beer (Kingfisher)", "Red Wine", "Whisky (Single Malt)", "Cocktail - Mojito"]

PROHIBITED_ITEMS = [
    ("Movie Tickets - PVR Cinemas", "PVR Cinemas"),
    ("Gym Membership - Monthly", "FitZone Gym"),
    ("Casino Chips", "Lucky Spin Casino"),
    ("Spa Package", "Serenity Spa"),
    ("Parking Fine", "City Traffic Authority"),
]


def random_date():
    """Return a random date in the last 30 days, as YYYY-MM-DD."""
    days_ago = random.randint(0, 29)
    d = datetime(2026, 6, 12) - timedelta(days=days_ago)
    return d.strftime("%Y-%m-%d")


def make_receipt(merchant_name, total_amount_inr, items):
    """Build a receipt dict in the shape our receipt_parser.py produces."""
    return {
        "merchant_name": merchant_name,
        "date": random_date(),
        "currency": "INR",
        "total_amount": round(total_amount_inr, 2),
        "items": items,
    }


def make_example(employee_level, location, business_purpose, receipt):
    """
    Build one training example by running the real audit_engine and
    packaging the result into the chat-message format used for fine-tuning.
    """
    audit_request = {
        "employee_level": employee_level,
        "location": location,
        "business_purpose": business_purpose,
        "receipt": receipt,
    }
    audit_result = audit_expense(audit_request)

    user_content = (
        "Audit the following expense claim against company policy. "
        "Return ONLY a JSON object with keys 'verdict', 'policy_section', "
        "and 'reason'.\n\n"
        f"Claim details:\n{json.dumps(audit_request, indent=2)}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(audit_result)},
        ]
    }

def generate_within_limit_examples(rules, n):
    examples = []
    categories = [c for c in CATEGORY_DATA if c in rules["spending_limits_usd"]]

    for _ in range(n):
        category = random.choice(categories)
        info = CATEGORY_DATA[category]

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)

        tier = get_seniority_tier(employee_level, rules)
        multiplier = get_regional_multiplier(location, rules)
        limit_usd = get_spending_limit(category, tier, rules)

        if limit_usd is None:
            continue

        limit_inr = limit_usd * multiplier * USD_TO_INR

        total_amount = round(limit_inr * random.uniform(0.5, 0.9), 2)

        merchant = random.choice(info["merchants"])
        item_name = random.choice(info["items"])
        items = [{"name": item_name, "quantity": 1, "amount": total_amount}]

        receipt = make_receipt(merchant, total_amount, items)
        examples.append(make_example(employee_level, location, info["purpose"], receipt))

    return examples


def generate_slightly_over_limit_examples(rules, n):
    examples = []
    categories = [c for c in CATEGORY_DATA if c in rules["spending_limits_usd"]]

    for _ in range(n):
        category = random.choice(categories)
        info = CATEGORY_DATA[category]

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)

        tier = get_seniority_tier(employee_level, rules)
        multiplier = get_regional_multiplier(location, rules)
        limit_usd = get_spending_limit(category, tier, rules)

        if limit_usd is None:
            continue

        limit_inr = limit_usd * multiplier * USD_TO_INR

        total_amount = round(limit_inr * random.uniform(1.01, 1.09), 2)
        if total_amount > rules["receipt_requirements"]["manager_preapproval_required_above_usd"] * USD_TO_INR:
            total_amount = rules["receipt_requirements"]["manager_preapproval_required_above_usd"] * USD_TO_INR * 0.95

        merchant = random.choice(info["merchants"])
        item_name = random.choice(info["items"])
        items = [{"name": item_name, "quantity": 1, "amount": total_amount}]

        receipt = make_receipt(merchant, total_amount, items)
        examples.append(make_example(employee_level, location, info["purpose"], receipt))

    return examples

def generate_well_over_limit_examples(rules, n):
    examples = []
    categories = [c for c in CATEGORY_DATA if c in rules["spending_limits_usd"]]

    for _ in range(n):
        category = random.choice(categories)
        info = CATEGORY_DATA[category]

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)

        tier = get_seniority_tier(employee_level, rules)
        multiplier = get_regional_multiplier(location, rules)
        limit_usd = get_spending_limit(category, tier, rules)

        if limit_usd is None:
            continue

        limit_inr = limit_usd * multiplier * USD_TO_INR

        total_amount = round(limit_inr * random.uniform(1.3, 1.6), 2)
        max_allowed = rules["receipt_requirements"]["manager_preapproval_required_above_usd"] * USD_TO_INR * 0.95
        total_amount = min(total_amount, max_allowed)

        merchant = random.choice(info["merchants"])
        item_name = random.choice(info["items"])
        items = [{"name": item_name, "quantity": 1, "amount": total_amount}]

        receipt = make_receipt(merchant, total_amount, items)
        examples.append(make_example(employee_level, location, info["purpose"], receipt))

    return examples

def generate_alcohol_examples(rules, n):
    examples = []
    categories = ["meals_daily", "business_dinner_per_head", "client_entertainment_event"]

    for _ in range(n):
        category = random.choice(categories)
        info = CATEGORY_DATA[category]

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)

        food_amount = round(random.uniform(500, 2500), 2)
        alcohol_amount = round(random.uniform(300, 1500), 2)
        total_amount = round(food_amount + alcohol_amount, 2)

        merchant = random.choice(info["merchants"])
        food_item = random.choice(info["items"])
        alcohol_item = random.choice(ALCOHOL_ITEMS)

        items = [
            {"name": food_item, "quantity": 1, "amount": food_amount},
            {"name": alcohol_item, "quantity": 1, "amount": alcohol_amount},
        ]

        receipt = make_receipt(merchant, total_amount, items)
        examples.append(make_example(employee_level, location, info["purpose"], receipt))

    return examples

def generate_prohibited_examples(rules, n):
    examples = []

    for _ in range(n):
        item_name, merchant = random.choice(PROHIBITED_ITEMS)

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)
        business_purpose = "Reimbursement request for personal expense incurred during trip"

        total_amount = round(random.uniform(500, 5000), 2)
        items = [{"name": item_name, "quantity": 1, "amount": total_amount}]

        receipt = make_receipt(merchant, total_amount, items)
        examples.append(make_example(employee_level, location, business_purpose, receipt))

    return examples

def generate_missing_field_examples(rules, n):
    examples = []
    categories = [c for c in CATEGORY_DATA]
    fields_to_drop = ["merchant_name", "date", "total_amount", "currency"]

    for _ in range(n):
        category = random.choice(categories)
        info = CATEGORY_DATA[category]

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)

        total_amount = round(random.uniform(300, 1500), 2)
        merchant = random.choice(info["merchants"])
        item_name = random.choice(info["items"])
        items = [{"name": item_name, "quantity": 1, "amount": total_amount}]

        receipt = make_receipt(merchant, total_amount, items)

        field = random.choice(fields_to_drop)
        if field == "total_amount":
            receipt["total_amount"] = 0
        elif field == "currency":
            receipt["currency"] = ""
        else:
            receipt[field] = ""

        examples.append(make_example(employee_level, location, info["purpose"], receipt))

    return examples

def generate_high_value_examples(rules, n):
    examples = []
    categories = ["conference_registration", "accommodation_international_night"]
    threshold_usd = rules["receipt_requirements"]["manager_preapproval_required_above_usd"]

    for _ in range(n):
        category = random.choice(categories)
        info = CATEGORY_DATA[category]

        employee_level = random.choice(EMPLOYEE_LEVELS)
        location = random.choice(ALL_CITIES)

        total_amount = round((threshold_usd * USD_TO_INR) * random.uniform(1.05, 1.5), 2)

        merchant = random.choice(info["merchants"])
        item_name = random.choice(info["items"])
        items = [{"name": item_name, "quantity": 1, "amount": total_amount}]

        receipt = make_receipt(merchant, total_amount, items)
        examples.append(make_example(employee_level, location, info["purpose"], receipt))

    return examples

def main():
    random.seed(RANDOM_SEED)
    rules = load_policy_rules()

    all_examples = []
    all_examples += generate_within_limit_examples(rules, EXAMPLES_PER_SCENARIO)
    all_examples += generate_slightly_over_limit_examples(rules, EXAMPLES_PER_SCENARIO)
    all_examples += generate_well_over_limit_examples(rules, EXAMPLES_PER_SCENARIO)
    all_examples += generate_alcohol_examples(rules, EXAMPLES_PER_SCENARIO)
    all_examples += generate_prohibited_examples(rules, EXAMPLES_PER_SCENARIO)
    all_examples += generate_missing_field_examples(rules, EXAMPLES_PER_SCENARIO)
    all_examples += generate_high_value_examples(rules, EXAMPLES_PER_SCENARIO)

    random.shuffle(all_examples)

    verdict_counts = {}
    for ex in all_examples:
        assistant_json = json.loads(ex["messages"][2]["content"])
        verdict = assistant_json["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    print("Total examples:", len(all_examples))
    print("Verdict breakdown:", verdict_counts)

    n = len(all_examples)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_examples = all_examples[:train_end]
    val_examples = all_examples[train_end:val_end]
    test_examples = all_examples[val_end:]

    write_jsonl("dataset/train.jsonl", train_examples)
    write_jsonl("dataset/val.jsonl", val_examples)
    write_jsonl("dataset/test.jsonl", test_examples)

    print(f"Wrote {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test examples.")


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


if __name__ == "__main__":
    main()