import json

def totals(serialized_records, categories):
    """Sum each requested category; category keys are unique strings.
    Return zero for missing keys. Accept fresh inputs on every call.
    Malformed JSON must raise json.JSONDecodeError, including when categories is empty.
    """
    records = json.loads(serialized_records)
    values = {category: [] for category in categories}
    if not values:
        return {}
    if len(values) == 1:
        category = next(iter(values))
        return {category: sum(record['value'] for record in records
                              if record['category'] == category)}
    for record in records:
        category = record['category']
        # Non-string JSON categories cannot match the requested string keys.
        if isinstance(category, str) and category in values:
            values[category].append(record['value'])
    # Keep sum's numeric semantics (including compensated floating-point sums).
    return {category: sum(group) for category, group in values.items()}
