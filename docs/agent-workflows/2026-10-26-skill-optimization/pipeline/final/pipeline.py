import json


def _totals_by_scan(records, categories):
    return {
        category: sum(record['value'] for record in records
                      if record['category'] == category)
        for category in categories
    }


def totals(serialized_records, categories):
    """Sum each requested category; category keys are unique strings.
    Return zero for missing keys. Accept fresh inputs on every call.
    Malformed JSON must raise json.JSONDecodeError, including when categories is empty.
    """
    records = json.loads(serialized_records)
    groups = {category: [] for category in categories}
    if len(groups) < 2:
        return _totals_by_scan(records, groups)
    try:
        for record in records:
            category = record['category']
            # JSON categories that are not strings cannot match requested keys.
            if isinstance(category, str) and category in groups:
                groups[category].append(record['value'])
    except (KeyError, TypeError):
        # Preserve the original category-first failure order for malformed records.
        return _totals_by_scan(records, groups)
    # Keep built-in sum and input order, including its float precision behavior.
    return {category: sum(values) for category, values in groups.items()}
