def convert_to_split_edges_format(data):
    """
    Convert a compressed LON dictionary with `edges` to a format with separate `edge_transitions` and `edge_weights`.

    Args:
        data (dict): Original compressed LON data with keys:
            - "local_optima": List of unique local optima.
            - "fitness_values": List of fitness values corresponding to the local optima.
            - "edges": Dictionary of edges with weights {(source, target): weight}.
    Returns:
        dict: Modified compressed LON data with `edge_transitions` and `edge_weights` instead of `edges`.
    """
    converted_data = {
        "local_optima": data["local_optima"],
        "fitness_values": data["fitness_values"],
        "edge_transitions": [],
        "edge_weights": [],
    }

    for (source, target), weight in data["edges"].items():
        converted_data["edge_transitions"].append((source, target))
        converted_data["edge_weights"].append(weight)

    return converted_data

def convert_to_single_edges_format(data):
    """
    Convert a compressed LON dictionary with separate `edge_transitions` and `edge_weights` to a format with `edges`.

    Args:
        data (dict): Modified compressed LON data with keys:
            - "local_optima": List of unique local optima.
            - "fitness_values": List of fitness values corresponding to the local optima.
            - "edge_transitions": List of edges as (source, target).
            - "edge_weights": List of edge weights corresponding to transitions.

    Returns:
        dict: Original compressed LON data with `edges` instead of `edge_transitions` and `edge_weights`.
    """
    converted_data = {
        "local_optima": data["local_optima"],
        "fitness_values": data["fitness_values"],
        "edges": {},
    }

    # print("Edge Transitions Sample:", data["edge_transitions"][:])
    # print("Edge Transition Types:", [type(transition) for transition in data["edge_transitions"][:]])

    for transition, weight in zip(data["edge_transitions"], data["edge_weights"]):
        # Ensure transition is a valid list or tuple with two elements
        if isinstance(transition, (list, tuple)) and len(transition) == 2:
            source, target = map(tuple, transition)  # Convert source and target to tuples
            converted_data["edges"][(source, target)] = weight
        else:
            raise ValueError(f"Invalid transition format: {transition}")


    # # Ensure edge_transitions are tuples
    # for transition, weight in zip(data["edge_transitions"], data["edge_weights"]):
    #     source, target = map(tuple, transition) if isinstance(transition, list) else transition
    #     converted_data["edges"][(source, target)] = weight

    return converted_data