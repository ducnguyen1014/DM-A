from itertools import product


def generate_permutations_with_last_flag_one(input_list):
    # Extract all odd indices to determine the flag positions
    flag_indices = [i for i in range(len(input_list)) if i % 2 == 1]

    # Ensure the last flag is always 1
    flag_indices_except_last = flag_indices[:-1]
    last_flag_index = flag_indices[-1]

    # Generate all possible permutations for the other flag values (0 or 1)
    all_permutations = product([0, 1], repeat=len(flag_indices_except_last))

    # Generate lists based on the permutations
    result = []
    for perm in all_permutations:
        new_list = input_list[:]  # Make a copy of the original list
        for index, flag_value in zip(flag_indices_except_last, perm):
            new_list[index] = flag_value
        # Set the last flag to 1
        new_list[last_flag_index] = 1
        result.append(new_list)

    # Ensure the given list is included
    if input_list not in result:
        result.append(input_list)

    return result


# Example usage
input_list = [1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 6, 1, 7, 0, 8, 0, 9, 0, 10, 1]
all_flag_permutations = generate_permutations_with_last_flag_one(input_list)

# Print the results
for perm in all_flag_permutations:
    print(perm)

print(len(all_flag_permutations))