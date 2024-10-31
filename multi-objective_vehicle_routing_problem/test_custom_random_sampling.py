import numpy as np
from pymoo.operators.sampling.rnd import Sampling
from pymoo.core.problem import ElementwiseProblem


class CustomRandomSampling(Sampling):

    def _do(self, problem, n_samples, number_of_customer, number_of_truck, **kwargs):
        X = []  # Start with an empty list to hold samples

        # Generate random permutations for each sample
        for i in range(n_samples):
            # Create an array with numbers from 1 to number_of_customer
            customer_numbers = np.arange(1, number_of_customer + 1)
            customer_permutation = np.random.permutation(
                customer_numbers
            )  # Permute the array
            sample = customer_permutation.tolist(), [
                number_of_truck
            ]  # Create the sample
            X.append(sample)  # Append the sample to the list X

        return X  # Return the list of samples


# Define a mock problem to pass to CustomRandomSampling
class MockProblem(ElementwiseProblem):
    def __init__(self, n_var):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0, xu=1)


# Test the _do function
number_of_customer = 5
number_of_truck = 3
n_samples = 10

# Initialize the problem and sampling
problem = MockProblem(n_var=number_of_customer + 1)  # Additional variable for trucks
sampling = CustomRandomSampling()

# Call the _do function
X = sampling._do(
    problem,
    n_samples,
    number_of_customer=number_of_customer,
    number_of_truck=number_of_truck,
)

# Print the generated samples
print("Generated samples:\n", X)
