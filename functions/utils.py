# import yaml

def sort_dictionary(data, order="asc", key=None):
    """
    Sorts a dictionary (or a list of dictionaries) based on a specific key.

    Args:
        data (dict | list): Dictionary or list of dictionaries to be sorted.
        key (str): The key name used for sorting.
        order (str): 'asc' for ascending (default), 'desc' for descending.

    Returns:
        list | dict: A list of sorted dictionaries or a sorted dictionary.

    Raises:
        TypeError: If the 'data' parameter is not a dictionary or a list of dictionaries.
        ValueError: If the list contains non-dictionary elements.
    """
    if not isinstance(data, (dict, list)):
        raise TypeError("The 'data' parameter must be a dictionary or a list of dictionaries.")

    reverse_order = order == "desc"

    # If it is a list of dictionaries
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("All items in the list must be dictionaries.")
        return sorted(data, key=lambda x: x.get(key, ""), reverse=reverse_order)

    # If it is a single dictionary
    if isinstance(data, dict):
        return dict(sorted(data.items(), key=lambda item: item[1].get(key, ""), reverse=reverse_order))
