import pygad
import logging

from drltrader.brain.brain import BrainConfiguration
from drltrader.brain.brain import Brain
from drltrader.data.data_provider import DataProvider

logging.basicConfig(format='%(asctime)s %(message)s',
                    filename='logs/training.log',
                    encoding='utf-8',
                    level=logging.DEBUG)


class TrainingConfiguration:
    def __init__(self,
                 training_scenarios: list,
                 testing_scenarios: list,
                 generations: int = 10,
                 population: int = 8,
                 parents_per_generation: int = 4,
                 elite_per_generation: int = 2,
                 total_timesteps_per_scenario: int = 1000):
        self.generations = generations
        self.population = population
        self.parents_per_generation = parents_per_generation
        self.elite_per_generation = elite_per_generation
        self.training_scenarios = training_scenarios
        self.testing_scenarios = testing_scenarios
        self.total_timesteps_per_scenario = total_timesteps_per_scenario


class EvolutionaryTrainer:
    MAX_LAYER_SIZE = 1024
    MIN_LAYER_SIZE = 32

    MAX_WINDOW_SIZE = 30
    MIN_WINDOW_SIZE = 1

    FIRST_LAYER_SIZE_GENE_IDX = 0
    SECOND_LAYER_SIZE_GENE_IDX = 1
    WINDOW_SIZE_GENE_IDX = 2
    FIRST_INDICATOR_GENE_IDX = 3

    INSTANCE = None

    def __init__(self, data_provider: DataProvider = DataProvider()):
        if EvolutionaryTrainer.INSTANCE is not None:
            raise ValueError("There can be only one instance of Trainer")

        EvolutionaryTrainer.INSTANCE = self

        self.data_provider = data_provider

        self.training_configuration: TrainingConfiguration = None
        self.generation = None

    def train(self, training_configuration: TrainingConfiguration) -> BrainConfiguration:
        self.generation = 0
        self.training_configuration = training_configuration
        self._initialize_genetic_algorithm()

        self.genetic_algorithm.run()
        return EvolutionaryTrainer._get_brain_configuration_from_dna(self, self.genetic_algorithm.best_solutions[0])

    def _initialize_genetic_algorithm(self):
        genes = len(self.data_provider.indicator_column_names) + 3
        self.genetic_algorithm = pygad.GA(num_generations=self.training_configuration.generations,
                                          num_genes=genes,
                                          init_range_low=0.0,
                                          init_range_high=1.0,
                                          sol_per_pop=self.training_configuration.population,
                                          save_best_solutions=True,

                                          fitness_func=EvolutionaryTrainer._evaluate_fitness,
                                          on_generation=EvolutionaryTrainer._on_generation,

                                          crossover_type='uniform',
                                          num_parents_mating=self.training_configuration.parents_per_generation,
                                          keep_parents=self.training_configuration.elite_per_generation,

                                          mutation_type='random',
                                          mutation_by_replacement=True,
                                          mutation_percent_genes=30.0,
                                          random_mutation_min_val=0.0,
                                          random_mutation_max_val=1.0)

    @staticmethod
    def _on_generation(ga_instance):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE

        trainer.generation += 1

        logging.info(f"Generation {trainer.generation} finished")

    @staticmethod
    def _evaluate_fitness(solution, solution_idx):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE
        brain_configuration = EvolutionaryTrainer._get_brain_configuration_from_dna(trainer, solution)

        logging.info(f"Solution {solution_idx} of generation {trainer.generation}")
        logging.info(f"Brain Configuration: {brain_configuration}")
        logging.info("Calculating fitness")

        brain: Brain = Brain(data_provider=trainer.data_provider,
                             brain_configuration=brain_configuration)

        for training_scenario in trainer.training_configuration.training_scenarios:
            logging.info(f"Training on scenario {training_scenario}")
            brain.learn(training_scenario=training_scenario,
                        total_timesteps=trainer.training_configuration.total_timesteps_per_scenario)
            logging.info(f"Training finished")

        mean_testing_profit = 0.0
        for testing_scenario in trainer.training_configuration.testing_scenarios:
            logging.info(f"Testing on scenario {testing_scenario}")
            profit = brain.test(testing_scenario=testing_scenario)['total_profit']
            logging.info(f"Testing finished with profit {profit}")
            mean_testing_profit += profit

        mean_testing_profit = mean_testing_profit / len(trainer.training_configuration.testing_scenarios)

        logging.info(f"Evaluation Profit: {mean_testing_profit}")

        return mean_testing_profit

    @staticmethod
    def _get_brain_configuration_from_dna(trainer, dna):
        window_size = EvolutionaryTrainer._calculate_value(EvolutionaryTrainer.MIN_WINDOW_SIZE,
                                                           EvolutionaryTrainer.MAX_WINDOW_SIZE,
                                                           EvolutionaryTrainer.WINDOW_SIZE_GENE_IDX,
                                                           dna)

        first_layer_size = EvolutionaryTrainer._calculate_value(EvolutionaryTrainer.MIN_LAYER_SIZE,
                                                                EvolutionaryTrainer.MAX_LAYER_SIZE,
                                                                EvolutionaryTrainer.FIRST_LAYER_SIZE_GENE_IDX,
                                                                dna)

        second_layer_size = EvolutionaryTrainer._calculate_value(EvolutionaryTrainer.MIN_LAYER_SIZE,
                                                                 EvolutionaryTrainer.MAX_LAYER_SIZE,
                                                                 EvolutionaryTrainer.SECOND_LAYER_SIZE_GENE_IDX,
                                                                 dna)

        signal_feature_names = []

        for indicator_idx in range(0, len(trainer.data_provider.indicator_column_names)):
            if dna[EvolutionaryTrainer.FIRST_INDICATOR_GENE_IDX + indicator_idx] > 0.5:
                signal_feature_names.append(trainer.data_provider.indicator_column_names[indicator_idx])

        return BrainConfiguration(first_layer_size=first_layer_size,
                                  second_layer_size=second_layer_size,
                                  window_size=window_size,
                                  signal_feature_names=signal_feature_names)

    @staticmethod
    def _calculate_value(min, max, gene_idx, dna):
        return int(min + (max - min) * dna[gene_idx])