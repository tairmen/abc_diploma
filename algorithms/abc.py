from algorithms.employee_bee import EmployedBee
from algorithms.onlooker_bee import OnlookerBee
import numpy as np
import random

class ArtificialBeeColony:

    def __init__(self, clf, features, X_train, X_test, y_train, y_test, modification_rate):
        self.food_sources = np.array([])
        self.features = features.copy()
        self.data = X_train.copy()
        self.test_data = X_test.copy()
        self.y_data = y_train.copy()
        self.y_test_data = y_test.copy()
        self.fitness = 0.0
        self.fitnesses = np.array([])
        self.modification_rate = modification_rate
        self.selected_features = self.features
        self.clf = clf

    def initialize_food_source(self, food_source_size, food_source_num):
        self.food_sources = np.empty((0, food_source_size), np.int)
        for i in range(food_source_num):
            self.food_sources = np.append(self.food_sources, np.array([[1 if j == i % food_source_size else 0 for j in range(food_source_size)]]), axis=0)

    def execute(self, cycle, target):
        self.initialize_food_source(len(self.features), len(self.data))

        best_food_source = np.array([random.choice((0, 1)) for _ in range(len(self.features))])
        for _ in range(cycle):
            employed_bees = np.array([])
            for food_source in self.food_sources:
                employed_bees = np.append(employed_bees, EmployedBee(self.clf, self.data, self.y_data, self.test_data, self.y_test_data, food_source, self.modification_rate))

            onlooker_bees = OnlookerBee(employed_bees)
            onlooker_bees.evaluates_nectar()
            self.fitness = onlooker_bees.get_best_fitness()
            self.fitnesses = np.append(self.fitnesses, self.fitness)
            best_food_source = onlooker_bees.get_best_food_source()
            self.selected_features = [f for i, f in enumerate(self.features) if best_food_source[i] == 1]

            if self.fitness >= target:
                return self.fitness, self.selected_features, onlooker_bees.get_best_employed_bee(), self.fitnesses

        return self.fitness, self.selected_features, onlooker_bees.get_best_employed_bee(), self.fitnesses
