import json


def totals(serialized_records, categories):
    """Sum each requested category; category keys are unique strings.
    Return zero for missing keys. Accept fresh inputs on every call.
    Malformed JSON must raise json.JSONDecodeError, including when categories is empty.
    """
    records = json.loads(serialized_records)
    result = {}
    for category in categories:
        result[category] = sum(record['value'] for record in records if record['category'] == category)
    return result
