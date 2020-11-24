import numpy as np
import random

class OnlookerBee:

    def __init__(self, employed_bees):
        self.employed_bees = employed_bees
        self.bees_distribution = np.array([])
        self.food_source_size = len(self.employed_bees[0].get_current_food_source())
        self.best_fitness = 0
        self.best_food_source = np.array([])
        self.best_employed_bee = None
        print("Init OnlookerBee")

    def get_best_food_source(self):
        return self.best_food_source

    def get_best_employed_bee(self):
        return self.best_employed_bee

    def get_best_fitness(self):
        return self.best_fitness

    def evaluates_nectar(self):
        for i in range(len(self.employed_bees)):
            self.employed_bees[i].calculate_fitness()

        self.roulette_wheel()

    def roulette_wheel(self):
        self.bees_distribution = np.array([])
        num_of_bees = len(self.employed_bees)
        total_fitness = 0
        count = 0
        total_fitness = np.array([])

        for bee in self.employed_bees:
            fitness = bee.get_current_fitness()
            if fitness > self.best_fitness:
                self.best_food_source = bee.get_current_food_source()
                self.best_fitness = fitness
                self.best_employed_bee = bee
            total_fitness = np.append(total_fitness, fitness)

        avg_fitness = np.mean(total_fitness)
        v_fitness = np.max(total_fitness) - np.min(total_fitness)
        if v_fitness == 0:
            v_fitness = 0.1
        var_fitness = np.std(total_fitness)

        for bee in self.employed_bees:
            count += 1
            freq_d = (bee.get_current_fitness() - avg_fitness) / v_fitness
            if freq_d < var_fitness:
                freq_d = 0
            freq = int(freq_d * len(self.employed_bees))    
            # print(f'{count} bee: food source - {bee.get_current_food_source()};')
            # print(f'{count} bee: fitness - {"{:.4f}".format(bee.get_current_fitness())}; importance - {"{:.4f}".format(freq_d)}')
            for i in range(freq):
                self.bees_distribution = np.append(self.bees_distribution, bee)

        num_of_dist = len(self.bees_distribution)

        for i in range(len(self.employed_bees)):
            if num_of_dist > 0:
                freq_d = (bee.get_current_fitness() - avg_fitness) / v_fitness
                if freq_d < var_fitness:
                    self.employed_bees[i] = self.bees_distribution[random.randrange(
                        0, num_of_dist)]
