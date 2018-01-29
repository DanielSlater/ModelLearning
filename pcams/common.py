def mean_squared_error(a, b):
    return ((a - b) ** 2).sum()


MAX_DIMENSIONS = 14


def combinations_between_bool_vectors(min_vector, max_vector):
    """
    Examples:
        >> list(combinations_between_bool_vectors([False, False], [False, True]))
        [(False, False), (False, True)]

    Args:
        min_vector (Iterable of bool):
        max_vector (Iterable of bool):

    Returns:
        Generator of Tuple of bool
    """
    if len(min_vector) <= 1:
        if min_vector[0]:
            yield (True,)
        elif max_vector[0]:
            yield (True,)
            yield (False,)
        else:
            yield (False,)
    else:
        for perm in combinations_between_bool_vectors(min_vector[1:], max_vector[1:]):
            if min_vector[0]:
                yield (True,) + perm
            elif max_vector[0]:
                yield (True,) + perm
                yield (False,) + perm
            else:
                yield (False,) + perm


def get_similar_states(position, position_data_pairs, size, distance_func):
    """

    Args:
        position (np.array):
        position_data_pairs ([(np.array, object)]):
        size (int):

    Returns:
        [distance:float, position:np.array, data:object]: list of length ``size`` ordered least to most similar (TODO double check this)
    """
    similar = []

    def inner(new_similarity, position, data):
        for i, (similarity, postiion, item) in enumerate(similar):
            if new_similarity < similarity:
                similar.insert(i, (new_similarity, position, data))
                return

        similar.append((new_similarity, position, data))

    threshold_diff = 9999999
    for other_position, data in position_data_pairs:
        new_similarity = distance_func(position, other_position)
        if len(similar) <= size or new_similarity < threshold_diff:
            inner(new_similarity, other_position, data)
            similar = similar[:size]
            threshold_diff, _, _ = similar[0]

    return similar


def count(iterator):
    return sum(1 for _ in iterator)