import numpy as np
from itertools import combinations


def generate_splits(customers, max_trucks):
    """
    Generate all possible splits of customers into 1 to max_trucks groups.
    Returns a list of lists, where each sublist is a grouping of customers.
    """
    all_splits = []

    for num_trucks in range(1, max_trucks + 1):
        # Generate all combinations of splits for the current number of trucks
        # We create indices for splitting
        indices = range(1, len(customers))  # Indices to split on

        # Find all combinations of splitting indices
        for comb in combinations(
            indices, num_trucks - 1
        ):  # num_trucks - 1 splits create num_trucks groups
            # Create split points based on the current combination of indices
            split_points = (0,) + comb + (len(customers),)
            split = [
                customers[split_points[i] : split_points[i + 1]]
                for i in range(num_trucks)
            ]
            all_splits.append(split)

    return all_splits


# Example list of customers as numbers
customers = list(range(1, 10))  # This will create the list [1, 2, 3, 4]

# Maximum number of trucks
max_trucks = 10

# Generate the splits
customer_splits = generate_splits(customers, max_trucks)

# Print the results
for idx, split in enumerate(customer_splits):
    print(f"Split {idx + 1}: {split}")
