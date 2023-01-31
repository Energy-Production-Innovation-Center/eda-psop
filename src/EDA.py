# MIT License

# Copyright (c) 2023 Artur Ferreira Brum

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import numpy as np
import pandas as pd

from collections import Counter
from copy import deepcopy
from scipy.stats import truncnorm
from sklearn.ensemble import RandomForestRegressor


# Modified bisect.insort to sort in decreasing order. Ordered insertion in a list
def reverse_insort(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid + 1
    a.insert(lo, x)


# Determines if there are duplicated rows in a matrix
def has_duplicated_rows(matrix):
    matrix = map(tuple, matrix)
    freq = Counter(matrix)
    for (i, f) in freq.items():
        if f > 1:
            return True
    return False


class Instance:
    def __init__(self, var_type, var_domain):
        assert len(var_type) == len(var_domain)
        self.var_type = var_type
        # Domain is a tuple (min, max) if the type is integer or real, or a set of values if discrete
        self.var_domain = var_domain
        self.var_count = len(var_domain)

    def get_var_count(self):
        return self.var_count

    def get_domain_size(self, var):
        return len(self.var_domain[var])


class Candidate:
    def __init__(self, solution, distributions):
        self.solution = deepcopy(solution)
        self.distributions = deepcopy(distributions)
        self.of_value = float('-inf')

    def __lt__(self, other):
        return self.of_value < other.of_value

    def __gt__(self, other):
        return self.of_value > other.of_value

    def get_of_value(self):
        return self.of_value

    def set_of_value(self, of_value):
        self.of_value = of_value

    def get_var_values(self):
        return deepcopy(self.solution)


class EDA():
    def __init__(self):
        self._name = __name__
        self._iter = 1
        self._solution_count = 0
        
        self._model = None
        self._history_data = []
        self._history_target = []
        self._encoded_columns = None
        self._soft_restarted = False

        # TODO: the following parameters should be initialized here
        self._population_size = 100 
        self._elite_size = 5  
        self._use_model = False          # Whether to use the regression model
        self._use_local_search = False   # Whether to perform the local search
        self._sampling_multiplier = 100  # Sampling factor to increase the sample size when the model is used
        self._budget = 1000              # Stopping criterion
        self._n_ls = 100                 # Limit the local search to the top self._n_ls solutions

        # The problem instance should be read here
        # TODO: instance_types is an array containing the type of each variable in the instance ('i' for integer, 'r' for real, or 'd' for discrete)
        instance_types = []
        # TODO: instance_domains is an array containing the domain of each variable. Each element is a tuple (min, max) if the type is integer or real, or a set of values if discrete
        instance_domains = []
        self._instance = Instance(instance_types, instance_domains)
        self._var_count = self._instance.get_var_count()
        
        # TODO: assign a label for each variable
        self._var_names = []
        
        # Store the inverse mapping between label and index (useful in the sampling procedure to quickly obtain the index of the parent's value)
        self._var_positions = {}
        for i in range(self._var_count):
            domain = self._instance.var_domain[i]
            positions = {}
            for j, value in enumerate(domain):
                positions[value] = j
            self._var_positions[i] = positions

        # Initialize the regression model
        if self._use_model:
            self._model = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=None, min_samples_split=2, n_jobs=4)
        if self._use_local_search and not self._use_model:
            print("The local search requires the regression model! Exiting...")
            sys.exit()

    def run(self):
        population = self._initial_population()
        self._evaluate(population)
        elite = []
        self._elite_candidates(elite, population)
        if self._use_model:
            self._train_model(population)

        while self._budget - self._solution_count >= self._population_size:
            self._iter += 1
            if self._use_model:
                population, prediction = self._sample_population(elite, self._population_size * self._sampling_multiplier)
            else:
                population, prediction = self._sample_population(elite, self._population_size)
            self._evaluate(population)
            self._elite_candidates(elite, population)
            if self._use_model:
                self._train_model(population)

        # Suggestion: print the best solution or the elite set

    def _initial_population(self):
        population = [None] * self._population_size 
        for i in range(self._population_size):
            population[i] = self._sample_candidate_uniform()
        return population

    def _evaluate(self, population):
        # TODO: this method should evaluate the population and set the objective function value for each individual with population[i].set_of_value(value).
        return
        
    def _elite_candidates(self, elite, population):
        # Get the objective function value of each solution in the population
        of_values = [None] * self._population_size
        for i, p in enumerate(population):
            of_values[i] = p.get_of_value()
        # Sort in non-increasing order
        value_cand = zip(of_values, population)
        value_cand = sorted(value_cand, reverse=True)
        # Update the elite set
        for value, cand in value_cand:
            if len(elite) < self._elite_size:
                reverse_insort(elite, cand)
            elif value > elite[-1].get_of_value():
                reverse_insort(elite, cand)
                elite.pop()
            else:
                break

    def _sample_candidate_uniform(self):
        var_count = self._instance.get_var_count()
        candidate = [None] * var_count
        distributions = [None] * var_count

        for i in range(var_count):
            if self._instance.var_type[i] == 'r':
                low, high = self._instance.var_domain[i]
                value = np.random.uniform(low, high)
                candidate[i] = value
                distributions[i] = (high - low) / 2.0
            elif self._instance.var_type[i] == 'i':
                low, high = self._instance.var_domain[i]
                value = np.random.random_integers(low, high)
                candidate[i] = value
                distributions[i] = (high - low) / 2.0
            elif self._instance.var_type[i] == 'd': 
                high = self._instance.get_domain_size(i)
                prob = [1 / high] * high
                distributions[i] = prob
                position = np.random.choice(high, 1, p=prob)[0]
                candidate[i] = self._instance.var_domain[i][position]

        return Candidate(candidate, distributions)

    def _train_model(self, population):
        # Organize the data from the population in a matrix and a separate array with targets
        population_data = []
        population_target = []
        for i in range(self._population_size):
            of_value = population[i].get_of_value()
            if of_value != float('-inf'):
                population_data.append(population[i].get_var_values())
                population_target.append(of_value)
        # Append population to the training data and fit the model
        self._history_data.extend(population_data)
        self._history_target.extend(population_target)
        # Using a pandas DataFrame to facilitate one-hot encoding
        df = pd.DataFrame(data=self._history_data, columns=self._var_names)
        df = pd.get_dummies(df, drop_first=True)
        # Store the encoded columns the first time that the model is trained
        if self._encoded_columns is None:
            self._encoded_columns = df.columns.values.tolist()
        training_data = np.array(df)
        self._model.fit(training_data, self._history_target)

    def _predict(self, population_matrix):
        df = pd.DataFrame(population_matrix, columns=self._var_names)
        df = pd.get_dummies(df, drop_first=True)
        # Reindex with the stored encoded columns to avoid missing columns
        if self._encoded_columns is not None:
            df = df.reindex(columns=self._encoded_columns, fill_value=0)
        data = np.array(df)
        return self._model.predict(data)

    def _sample_population(self, elite, sample_size):
        # Determine the probability of selecting each elite as the parent
        probabilities = []
        for i in range(1, self._elite_size + 1):
            probabilities.append((self._elite_size - i + 1) / (self._elite_size * (self._elite_size + 1) / 2))
        
        # For each elite, compute the updated probability distributions before sampling the new candidates
        if not self._soft_restarted:
            for i in range(self._elite_size):
                for j in range(self._var_count):
                    if self._instance.var_type[j] != 'd':
                        # Slightly reduce the standard deviation
                        stdev = elite[i].distributions[j]
                        stdev *= (1 / self._population_size) ** (1 / self._var_count)
                        elite[i].distributions[j] = stdev
                    else:
                        var_probabilities = elite[i].distributions[j]  # Reference
                        # First, all probabilities are slightly decreased
                        for p in range(len(var_probabilities)):
                            var_probabilities[p] *= 1 - (self._iter - 1) / (self._budget / self._population_size)
                        # And then we slightly increase the probability of selecting the same value as the elite parent
                        parent_position = self._var_positions[j][elite[i].solution[j]]
                        var_probabilities[parent_position] += (self._iter - 1) / (self._budget / self._population_size)

        # Get a parent for each one of the candidates according to the probabilities determined before
        parent_index = list(np.random.choice(self._elite_size, sample_size, p=probabilities))

        # Prepare a matrix to store candidates. A candidate is represented as an array of values here
        candidate_matrix = [None] * sample_size
        for i in range(len(candidate_matrix)):
            candidate_matrix[i] = [None] * self._var_count

        # Sample candidates
        for j in range(self._var_count):
            if self._instance.var_type[j] == 'r':
                low, high = self._instance.var_domain[j]
                for i in range(sample_size):
                    # Get the mean and the st.dev. from the elite parent
                    mean = elite[parent_index[i]].solution[j]
                    stdev = elite[parent_index[i]].distributions[j]
                    low = (low - mean) / stdev
                    high = (high - mean) / stdev
                    candidate_matrix[i][j] = truncnorm.rvs(low, high, loc=mean, scale=stdev)
            elif self._instance.var_type[j] == 'i':
                low, high = self._instance.var_domain[j]
                for i in range(sample_size):
                    # Get the mean and the st.dev. from the elite parent
                    mean = elite[parent_index[i]].solution[j]
                    stdev = elite[parent_index[i]].distributions[j]
                    high += 1
                    mean += 0.5
                    low = (low - mean) / stdev
                    high = (high - mean) / stdev
                    candidate_matrix[i][j] = int(truncnorm.rvs(low, high, loc=mean, scale=stdev))
            elif self._instance.var_type[j] == 'd':
                high = self._instance.get_domain_size(j)
                for i in range(sample_size):
                    # Get the probabilities from the elite parent
                    prob = elite[parent_index[i]].distributions[j]
                    pos = np.random.choice(high, 1, p=prob)[0]
                    candidate_matrix[i][j] = self._instance.var_domain[j][pos]

        if self._use_model:
            # Predict the performance and sort the candidates in non-increasing order
            predicted = self._predict(candidate_matrix)
            value_cand = zip(predicted, candidate_matrix, parent_index)
            value_cand = sorted(value_cand, reverse=True)
            tuples = zip(*value_cand)
            predicted, candidate_matrix, parent_index = [list(t) for t in tuples]

            # Perform a local search
            if self._use_local_search:
                candidate_matrix[:self._n_ls], predicted[:self._n_ls] = self._local_search(candidate_matrix[:self._n_ls], predicted[:self._n_ls])

            # Remove duplicates. Note: could stop at self._population_size. However, checking the whole matrix is necessary for the soft-restart test
            i = 0
            while i < len(candidate_matrix) - 1 and len(candidate_matrix) > self._population_size:
                j = i + 1
                while j < len(candidate_matrix) and candidate_matrix[i] == candidate_matrix[j] and len(candidate_matrix) > self._population_size:
                    candidate_matrix.pop(j)
                    predicted.pop(j)
                    parent_index.pop(j)
                i += 1
        
            assert len(candidate_matrix) >= self._population_size

            if len(candidate_matrix) / sample_size < 0.1:
                self._soft_restart(elite)
                self._soft_restarted = True
            else:
                self._soft_restarted = False

            # Keep the candidates with the best predictions and discard the rest
            candidate_matrix = candidate_matrix[:self._population_size]
            predicted = predicted[:self._population_size]
            parent_index = parent_index[:self._population_size]
        
        # If the model isn't used, perform just a simple check for duplicate candidates to trigger the soft-restart
        else:
            if has_duplicated_rows(candidate_matrix):
                self._soft_restart(elite)
                self._soft_restarted = True
            else:
                self._soft_restarted = False

        # Create Candidates from the arrays of values
        population = [None] * self._population_size
        for i in range(self._population_size):
            candidate = deepcopy(candidate_matrix[i])
            parent_distributions = deepcopy(elite[parent_index[i]].distributions)
            population[i] = Candidate(candidate, parent_distributions)

        if self._use_model:
            return population, predicted
        else:
            return population, []

    # Note: the current version only works with discrete variables
    def _local_search(self, candidate_matrix, predictions):
        assert len(candidate_matrix) == len(predictions)

        for i, (candidate, initial_prediction) in enumerate(zip(candidate_matrix, predictions)):
            incumbent_solution = deepcopy(candidate)
            incumbent_prediction = initial_prediction
            while True:
                # Predict the performance of all neighbors
                neighbors = self._get_neighbors(incumbent_solution)
                neigh_prediction = self._predict(neighbors)
                # Get the best prediction and its index
                index_max = np.argmax(neigh_prediction)
                best_prediction = neigh_prediction[index_max]
                # Stop on local optima
                if best_prediction <= incumbent_prediction:
                    break
                # Update incumbent
                incumbent_solution = deepcopy(neighbors[index_max])
                incumbent_prediction = best_prediction
            
            # Set candidate = best solution (only if the initial solution was improved)
            if incumbent_prediction != initial_prediction:
                predictions[i] = incumbent_prediction
                candidate_matrix[i] = deepcopy(incumbent_solution)

        return candidate_matrix, predictions

    def _get_neighbors(self, candidate_values):
        neighbors = []
        for i in range(self._var_count):
            domain = self._instance.var_domain[i]
            for value in domain:
                if value != candidate_values[i]:
                    neighbor = deepcopy(candidate_values)
                    neighbor[i] = value
                    neighbors.append(neighbor)
        
        return neighbors

    # Note: the current version only works with discrete variables
    def _soft_restart(self, elite):
        for e in elite:
            for i in range(self._var_count):
                if isinstance(e.distributions[i], list):
                    e.distributions[i] = [v * 0.9 + 0.1 * max(e.distributions[i]) for v in e.distributions[i]]
                    e.distributions[i] = [v / sum(e.distributions[i]) for v in e.distributions[i]]
