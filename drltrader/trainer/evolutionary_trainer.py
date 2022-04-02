import pandas as pd
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
                 generations: int = 2,
                 parents_per_generation: int = 2,
                 start_population: int = 6,
                 stop_population: int = 4,
                 step_population: int = -1,
                 start_timesteps: int = 1000,
                 stop_timesteps: int = 1500,
                 step_timesteps: int = 500,
                 solutions_statistics_filename: str = None):
        self.training_scenarios = training_scenarios
        self.testing_scenarios = testing_scenarios

        self.generations = generations
        self.parents_per_generation = parents_per_generation

        self.start_population = start_population
        self.stop_population = stop_population + 1
        self.step_population = step_population

        self.start_timesteps = start_timesteps
        self.stop_timesteps = stop_timesteps + 1
        self.step_timesteps = step_timesteps

        self.solutions_statistics_filename = solutions_statistics_filename


class EvolutionaryTrainer:
    MAX_LAYER_SIZE = 1024
    MIN_LAYER_SIZE = 32

    MAX_WINDOW_SIZE = 30
    MIN_WINDOW_SIZE = 1

    FIRST_LAYER_SIZE_GENE_IDX = 0
    SECOND_LAYER_SIZE_GENE_IDX = 1
    WINDOW_SIZE_GENE_IDX = 2
    USE_NORMALIZED_OBS_GENE_IDX = 3
    FIRST_INDICATOR_GENE_IDX = 4
    INDICATOR_GENE_ACTIVATION_THRESHOLD = 0.8

    INSTANCE = None

    def __init__(self, data_provider: DataProvider = DataProvider()):
        if EvolutionaryTrainer.INSTANCE is not None:
            raise ValueError("There can be only one instance of Trainer")

        EvolutionaryTrainer.INSTANCE = self

        self.data_provider = data_provider

        self.training_configuration: TrainingConfiguration = None
        self.fitness_cache = None
        self.solutions_statistics: pd.DataFrame = None
        self.current_population = None
        self.current_timestep = None

    def train(self, training_configuration: TrainingConfiguration) -> BrainConfiguration:
        self.fitness_cache = {}
        self.solutions_statistics = pd.DataFrame(columns=['Profit', 'Brain Configuration'])
        self.training_configuration = training_configuration
        self.current_population = self.training_configuration.start_population
        self.current_timesteps = self.training_configuration.start_timesteps
        self._initialize_genetic_algorithm()

        self.genetic_algorithm.run()
        return EvolutionaryTrainer._get_brain_configuration_from_dna(self, self.genetic_algorithm.best_solutions[0])

    def _initialize_genetic_algorithm(self):
        genes = len(self.data_provider.indicator_column_names) + EvolutionaryTrainer.FIRST_INDICATOR_GENE_IDX
        self.genetic_algorithm = pygad.GA(num_generations=self.training_configuration.generations,
                                          num_genes=genes,
                                          init_range_low=0.0,
                                          init_range_high=1.0,
                                          sol_per_pop=self.current_population,
                                          save_best_solutions=True,

                                          fitness_func=EvolutionaryTrainer._evaluate_fitness,
                                          on_generation=EvolutionaryTrainer._on_generation,

                                          crossover_type='uniform',
                                          num_parents_mating=self.training_configuration.parents_per_generation,
                                          keep_parents=0,

                                          mutation_type='random',
                                          mutation_by_replacement=True,
                                          mutation_percent_genes=30.0,
                                          random_mutation_min_val=0.0,
                                          random_mutation_max_val=1.0)

    @staticmethod
    def _on_generation(ga_instance):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE

        logging.info(f"Generation {trainer.genetic_algorithm.generations_completed} finished. Here are the results:")
        logging.info('\t' + trainer.solutions_statistics.to_string().replace('\n', '\n\t'))

        # Update Population
        current_population_values = list(range(trainer.training_configuration.start_population,
                                               trainer.training_configuration.stop_population,
                                               trainer.training_configuration.step_population))
        current_population_idx = min(trainer.genetic_algorithm.generations_completed,
                                     len(current_population_values) - 1)
        trainer.current_population = current_population_values[current_population_idx]
        trainer.genetic_algorithm.num_offspring = trainer.current_population

        # Update Timesteps
        current_timesteps_values = list(range(trainer.training_configuration.start_timesteps,
                                              trainer.training_configuration.stop_timesteps,
                                              trainer.training_configuration.step_timesteps))
        current_timesteps_idx = min(trainer.genetic_algorithm.generations_completed,
                                    len(current_timesteps_values) - 1)
        trainer.current_timesteps = current_timesteps_values[current_timesteps_idx]

        logging.info(f"New population: {trainer.current_population}. New timesteps: {trainer.current_timesteps}")

    @staticmethod
    def _evaluate_fitness(solution, solution_idx):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE
        brain_configuration = EvolutionaryTrainer._get_brain_configuration_from_dna(trainer, solution)
        solution_name = f"{trainer.genetic_algorithm.generations_completed}_{solution_idx}"

        logging.info(f"Solution {solution_name}")
        logging.info(f"Brain Configuration: {brain_configuration}")

        if solution_name not in trainer.fitness_cache:
            logging.info("Fitness not in cache, calculating fitness...")

            brain: Brain = Brain(data_provider=trainer.data_provider,
                                 brain_configuration=brain_configuration)

            for training_scenario in trainer.training_configuration.training_scenarios:
                logging.info(f"Training on scenario {training_scenario} with {trainer.current_timesteps} timesteps")
                brain.learn(training_scenario=training_scenario,
                            total_timesteps=trainer.current_timesteps)
                logging.info(f"Training finished")

            mean_testing_profit = 0.0
            for testing_scenario in trainer.training_configuration.testing_scenarios:
                logging.info(f"Testing on scenario {testing_scenario}")
                profit = brain.test(testing_scenario=testing_scenario)['total_profit']
                logging.info(f"Testing finished with profit {profit}")
                mean_testing_profit += profit

            mean_testing_profit = mean_testing_profit / len(trainer.training_configuration.testing_scenarios)

            trainer.solutions_statistics.at[solution_name, 'Profit'] = mean_testing_profit
            trainer.solutions_statistics.at[solution_name, 'Brain Configuration'] = brain_configuration
            trainer.solutions_statistics.sort_values(by='Profit', inplace=True, ascending=False)
            if trainer.training_configuration.solutions_statistics_filename is not None:
                trainer.solutions_statistics.to_csv(trainer.training_configuration.solutions_statistics_filename)

            logging.info(f"Evaluation Profit: {mean_testing_profit}")
            trainer.fitness_cache[solution_name] = mean_testing_profit

            return mean_testing_profit
        else:
            logging.info("Fitness in cache, returning saved fitness...")
            return trainer.fitness_cache[solution_name]

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

        use_normalized_observations = True if dna[EvolutionaryTrainer.USE_NORMALIZED_OBS_GENE_IDX] > 0.5 else False

        signal_feature_names = []

        for indicator_idx in range(0, len(trainer.data_provider.indicator_column_names)):
            if dna[EvolutionaryTrainer.FIRST_INDICATOR_GENE_IDX + indicator_idx] > \
                    EvolutionaryTrainer.INDICATOR_GENE_ACTIVATION_THRESHOLD:
                signal_feature_names.append(trainer.data_provider.indicator_column_names[indicator_idx])

        return BrainConfiguration(first_layer_size=first_layer_size,
                                  second_layer_size=second_layer_size,
                                  window_size=window_size,
                                  signal_feature_names=signal_feature_names,
                                  use_normalized_observations=use_normalized_observations)

    @staticmethod
    def _calculate_value(min, max, gene_idx, dna):
        return int(min + (max - min) * dna[gene_idx])