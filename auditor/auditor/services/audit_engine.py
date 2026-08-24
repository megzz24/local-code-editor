from services.policy_loader import (
    load_policy_rules,
    get_seniority_tier,
    get_regional_multiplier,
    get_spending_limit,
    detect_category,
    contains_alcohol,
    contains_prohibited_item,
    convert_to_usd,
)


def audit_expense(request: dict) -> dict:
    rules = load_policy_rules()

    employee_level = request.get("employee_level", "")
    location = request.get("location", "")
    business_purpose = request.get("business_purpose", "")
    receipt = request.get("receipt", {})

    items = receipt.get("items", [])
    total_amount = receipt.get("total_amount", 0)
    currency = receipt.get("currency", "")

    rules_for_conversion = rules  # alias for readability below
    total_amount_usd = convert_to_usd(total_amount, currency or "USD", rules_for_conversion)

    if contains_alcohol(items, rules):
        return {
            "verdict": "Rejected",
            "policy_section": rules["alcohol_keywords"]["section"],  # "4.2"
            "reason": (
                "The receipt contains an alcohol item. Per Section 4.2, "
                "alcohol is strictly non-reimbursable under any circumstances "
                "and alcohol line items must be removed before submission."
            ),
        }

    prohibited_match = contains_prohibited_item(items, rules)
    if prohibited_match:
        return {
            "verdict": "Rejected",
            "policy_section": rules["prohibited_categories"]["section"],  # "8"
            "reason": (
                f"The receipt contains an item matching the prohibited keyword "
                f"'{prohibited_match}'. Per Section 8, this category of expense "
                f"is never reimbursable, regardless of business context."
            ),
        }

    receipt_rules = rules["receipt_requirements"]

    required_fields = ["merchant_name", "date", "total_amount", "currency"]
    missing_fields = [f for f in required_fields if not receipt.get(f)]
    if missing_fields:
        return {
            "verdict": "Flagged",
            "policy_section": receipt_rules["section"],  # "9"
            "reason": (
                f"The receipt is missing required field(s): {', '.join(missing_fields)}. "
                f"Per Section 9, a valid receipt must show merchant name, date, "
                f"itemised charges, total amount, and currency. Please resubmit "
                f"a clearer receipt."
            ),
        }

    if total_amount_usd > receipt_rules["manager_preapproval_required_above_usd"]:
        return {
            "verdict": "Flagged",
            "policy_section": receipt_rules["section"],  # "9"
            "reason": (
                f"The claim amount ({currency} {total_amount}, approx. "
                f"USD {round(total_amount_usd, 2)}) is above "
                f"USD {receipt_rules['manager_preapproval_required_above_usd']}. "
                f"Per Section 9, claims above this amount require receipt + "
                f"manager pre-approval. Please confirm pre-approval has been "
                f"obtained before this claim can be approved."
            ),
        }

    seniority_tier = get_seniority_tier(employee_level, rules)
    regional_multiplier = get_regional_multiplier(location, rules)

    category = detect_category(business_purpose, rules)
    if category is None:
        category = detect_category(receipt.get("merchant_name", ""), rules)

    if category is None:
        return {
            "verdict": "Flagged",
            "policy_section": "3",
            "reason": (
                f"Could not automatically match the business purpose "
                f"'{business_purpose}' to a spending category in Section 3. "
                f"Please clarify the expense category for manual review."
            ),
        }

    base_limit = get_spending_limit(category, seniority_tier, rules)

    if base_limit is None:
        return {
            "verdict": "Flagged",
            "policy_section": "3",
            "reason": (
                f"For category '{category}' at tier '{seniority_tier}', "
                f"the policy specifies 'Reasonable' rather than a fixed limit. "
                f"This claim is flagged for manual judgement on reasonableness."
            ),
        }

    adjusted_limit = round(base_limit * regional_multiplier, 2)

    if total_amount_usd <= adjusted_limit:
        return {
            "verdict": "Approved",
            "policy_section": "3",
            "reason": (
                f"Claim of {currency} {total_amount} (approx. USD {round(total_amount_usd, 2)}) "
                f"for category '{category}' is within the Section 3 limit of USD {base_limit} "
                f"(adjusted to USD {adjusted_limit} for location '{location}', "
                f"tier '{seniority_tier}', multiplier {regional_multiplier}x)."
            ),
        }
    flag_threshold = adjusted_limit * 1.10

    if total_amount_usd <= flag_threshold:
        return {
            "verdict": "Flagged",
            "policy_section": "3",
            "reason": (
                f"Claim of {currency} {total_amount} (approx. USD {round(total_amount_usd, 2)}) "
                f"for category '{category}' slightly exceeds the Section 3 limit of "
                f"USD {adjusted_limit} (base USD {base_limit} x {regional_multiplier} "
                f"for '{location}', tier '{seniority_tier}'). Flagged for manager review - "
                f"please confirm business justification for the overage."
            ),
        }

    return {
        "verdict": "Rejected",
        "policy_section": "3",
        "reason": (
            f"Claim of {currency} {total_amount} (approx. USD {round(total_amount_usd, 2)}) "
            f"for category '{category}' exceeds the Section 3 limit of USD {adjusted_limit} "
            f"(base USD {base_limit} x {regional_multiplier} for '{location}', "
            f"tier '{seniority_tier}') by more than 10%."
        ),
    }

if __name__ == "__main__":
    import json

    example_requests = [
        {
            "employee_level": "L5",
            "location": "Tokyo",
            "business_purpose": "Client dinner with vendor",
            "receipt": {
                "merchant_name": "Sushi Place",
                "date": "2026-06-01",
                "currency": "USD",
                "total_amount": 110,
                "items": [{"name": "Set menu", "quantity": 2, "amount": 110}],
            },
        },
        {
            "employee_level": "L3",
            "location": "Mumbai",
            "business_purpose": "Team lunch",
            "receipt": {
                "merchant_name": "Cafe Mocha",
                "date": "2026-06-02",
                "currency": "USD",
                "total_amount": 18,
                "items": [
                    {"name": "Sandwich", "quantity": 1, "amount": 10},
                    {"name": "Beer", "quantity": 1, "amount": 8},
                ],
            },
        },
    ]

    for req in example_requests:
        result = audit_expense(req)
        print(json.dumps(result, indent=2))
        print("-" * 50)