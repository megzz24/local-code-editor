import json
import os

POLICY_FILE = os.path.join(os.path.dirname(__file__), "..", "policy", "policy_rules.json")


def load_policy_rules(path: str = POLICY_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_seniority_tier(employee_level: str, rules: dict) -> str:
    employee_level = employee_level.strip().upper()
    tiers = rules["seniority_tiers"]

    for tier_name, tier_info in tiers.items():
        if tier_name == "section":
            continue
        if employee_level in tier_info["levels"]:
            return tier_name

    if employee_level.startswith("L"):
        try:
            level_number = int(employee_level.replace("+", "").replace("L", ""))
            if level_number >= 7:
                return "Executive"
            elif level_number in (5, 6):
                return "Senior"
            elif level_number in (3, 4):
                return "Mid-Level"
            elif level_number in (1, 2):
                return "Junior"
        except ValueError:
            pass

    return "Junior"


def get_regional_multiplier(location: str, rules: dict) -> float:
    location_lower = location.strip().lower()
    multipliers = rules["regional_multipliers"]

    for tier_name, tier_info in multipliers.items():
        if tier_name == "section":
            continue
        if location_lower in tier_info["example_cities"]:
            return tier_info["multiplier"]

    return multipliers["Tier3"]["multiplier"]


def get_spending_limit(category: str, seniority_tier: str, rules: dict):
    limits = rules["spending_limits_usd"]
    if category not in limits:
        return None
    return limits[category].get(seniority_tier)


def detect_category(text: str, rules: dict):
    text_lower = text.lower()
    for category, keywords in rules["category_keywords"].items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    return None


def convert_to_usd(amount: float, currency: str, rules: dict) -> float:
    rates = rules.get("currency_rates_to_usd", {})
    rate = rates.get(currency.upper(), 1.0)
    return amount * rate


def contains_alcohol(items: list, rules: dict) -> bool:
    alcohol_words = rules["alcohol_keywords"]["keywords"]
    for item in items:
        item_name = item.get("name", "").lower()
        for word in alcohol_words:
            if word in item_name:
                return True
    return False


def contains_prohibited_item(items: list, rules: dict):
    prohibited_words = rules["prohibited_categories"]["keywords"]
    for item in items:
        item_name = item.get("name", "").lower()
        for word in prohibited_words:
            if word in item_name:
                return word
    return None

if __name__ == "__main__":
    rules = load_policy_rules()
    print("Loaded policy:", rules["meta"]["policy_ref"], rules["meta"]["version"])
    print("L5 ->", get_seniority_tier("L5", rules))
    print("Tokyo multiplier ->", get_regional_multiplier("Tokyo", rules))
    print("meals_daily / Senior limit ->", get_spending_limit("meals_daily", "Senior", rules))
    print("detect_category('client dinner with vendor') ->",
          detect_category("client dinner with vendor", rules))