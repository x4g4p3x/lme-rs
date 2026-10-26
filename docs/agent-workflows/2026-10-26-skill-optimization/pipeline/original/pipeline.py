import json

def totals(serialized_records, categories):
    """Sum each requested category; category keys are unique strings.
    Return zero for missing keys. Accept fresh inputs on every call.
    Malformed JSON must raise json.JSONDecodeError, including when categories is empty.
    """
    result = {}
    if not categories:
        json.loads(serialized_records)
    for category in categories:
        records = json.loads(serialized_records)
        result[category] = sum(record['value'] for record in records if record['category'] == category)
    return result
