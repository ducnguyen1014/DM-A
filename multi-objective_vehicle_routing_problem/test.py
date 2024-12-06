import random
from typing import List
from collections import Counter
import copy

random.seed(30)


@staticmethod
def is_valid_adjacent_route(adjacent_route: List[List[int]]):
    if not adjacent_route:
        return False  # Empty list is not valid

    # Check if the tail of the previous element matches the head of the next element
    for i in range(len(adjacent_route) - 1):
        if adjacent_route[i][1] != adjacent_route[i + 1][0]:
            return False

    # Check if the head of the first element matches the tail of the last element
    if adjacent_route[0][0] != adjacent_route[-1][1]:
        return False

    counts = Counter(num for sublist in adjacent_route for num in sublist)

    # Ensure each number appears at most twice
    if any(count > 2 for count in counts.values()):
        return False

    return True


@staticmethod
def reverse_and_swap(lst, a, b):
    """
    Reverse and swap all elements in a list between indices a and b (exclusive),
    treating the list as cyclic and keeping the list length unchanged.

    Args:
        lst (list): A list of continuous pairs (e.g., [[1, 2], [2, 3], [3, 4]...]).
        a (int): Start index (exclusive).
        b (int): End index (exclusive).

    Returns:
        list: The modified list.
    """
    if not isinstance(lst, list) or len(lst) < 0:
        raise ValueError(f"lst ({lst}) must be a list.")

    for idx, var in zip(("a", "b"), (a, b)):
        if var < 0 or var >= len(lst):
            raise ValueError(f"{idx} ({var}) is invalid.")

    if a == b:
        raise ValueError(f"a ({a}) and b ({b}) must be different.")

    lst = copy.deepcopy(lst)

    # Interal reverse and swap
    if a < b:
        sublist = lst[a + 1 : b]
        reversed_sublist = [[item[1], item[0]] for item in sublist]
        reversed_sublist.reverse()
        lst[a + 1 : b] = reversed_sublist

    # External reverse and swap
    else:
        first_part_length = len(lst) - (a + 1)

        sublist = lst[a + 1 :]
        sublist.extend(lst[:b])
        reversed_sublist = [[item[1], item[0]] for item in sublist]
        reversed_sublist.reverse()

        lst[a + 1 :] = reversed_sublist[:first_part_length]
        lst[:b] = reversed_sublist[first_part_length:]

    return lst


@staticmethod
def swap_elements(list1, index1, list2, index2):
    """
    Swap elements between two lists at the specified indices.

    Parameters:
    - list1: First list
    - list2: Second list
    - index1: Index of the element in list1 to swap
    - index2: Index of the element in list2 to swap
    """
    # Swap elements
    list1[index1], list2[index2] = list2[index2], list1[index1]


@staticmethod
def edge_exchange_crossover(
    adjacent_route_1: List[List[int]], adjacent_route_2: List[List[int]]
):
    if not is_valid_adjacent_route(adjacent_route_1):
        raise ValueError(f"Invalid adjacent_route_1 {adjacent_route_1}.")

    if not is_valid_adjacent_route(adjacent_route_2):
        raise ValueError(f"Invalid adjacent_route_2 {adjacent_route_2}.")

    adjacent_route_1 = copy.deepcopy(adjacent_route_1)
    adjacent_route_2 = copy.deepcopy(adjacent_route_2)

    # i1_index = random.randint(0, len(adjacent_route_1))
    i1_index = 1
    i1 = adjacent_route_1[i1_index]

    # Finding i2 (same head as i1)
    i2_index = next(
        (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[0]), None
    )
    i2 = adjacent_route_2[i2_index]

    # Finding j2 (same tail as i1)
    j2_index = next(
        (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[1]), None
    )

    # Finding j1 (same head as j1)
    j1_index = next(
        (adjacent_route_1.index(x) for x in adjacent_route_1 if x[0] == i2[1]), None
    )

    while True:

        i1 = adjacent_route_1[i1_index]
        i2 = adjacent_route_2[i2_index]

        # Finding j2 (same tail as i1)
        j2_index = next(
            (adjacent_route_2.index(x) for x in adjacent_route_2 if x[0] == i1[1]), None
        )

        # Finding j1 (same head as j1)
        j1_index = next(
            (adjacent_route_1.index(x) for x in adjacent_route_1 if x[0] == i2[1]), None
        )

        # Swap i1 and i2
        swap_elements(adjacent_route_1, i1_index, adjacent_route_2, i2_index)

        adjacent_route_1 = reverse_and_swap(adjacent_route_1, i1_index, j1_index)
        adjacent_route_2 = reverse_and_swap(adjacent_route_2, i2_index, j2_index)

        if is_valid_adjacent_route(adjacent_route_1) and is_valid_adjacent_route(
            adjacent_route_2
        ):
            break

        i1_index = j1_index
        i2_index = j2_index

    return adjacent_route_1, adjacent_route_2


parent_1 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 1]]
parent_2 = [[2, 5], [5, 4], [4, 1], [1, 6], [6, 7], [7, 3], [3, 8], [8, 2]]

edge_exchange_crossover(parent_1, parent_2)
