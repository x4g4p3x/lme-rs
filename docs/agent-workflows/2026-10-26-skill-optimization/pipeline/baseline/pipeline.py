import json


def totals(serialized_records, categories):
    """Sum each requested category; category keys are unique strings.
    Return zero for missing keys. Accept fresh inputs on every call.
    Malformed JSON must raise json.JSONDecodeError, including when categories is empty.
    """
    records = json.loads(serialized_records)
    values_by_category = {category: [] for category in categories}
    if not values_by_category:
        return {}

    for record in records:
        category = record['category']
        if isinstance(category, str) and category in values_by_category:
            values_by_category[category].append(record['value'])

    # Keep builtin sum's numeric behavior and each category's original value order.
    return {category: sum(values) for category, values in values_by_category.items()}
