import datetime
import math
import pandas as pd
import pygad
import logging
import logging.config

from drltrader.brain.brain import BrainConfiguration
from drltrader.brain.brain import Brain
from drltrader.data import DataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.trainer.dna_decoder import DnaDecoder

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TrainingConfiguration:
    def __init__(self,
                 training_scenarios: list,
                 validation_scenarios: list,
                 generations: int = 2,
                 start_population: int = 6,
                 stop_population: int = 4,
                 step_population: int = -1,
                 start_timesteps: int = 1000,
                 stop_timesteps: int = 1500,
                 step_timesteps: int = 500,
                 solutions_statistics_filename: str = None):
        self.training_scenarios = training_scenarios
        self.validation_scenarios = validation_scenarios

        self.generations = generations

        self.start_population = start_population
        self.stop_population = stop_population + 1
        self.step_population = step_population

        self.start_timesteps = start_timesteps
        self.stop_timesteps = stop_timesteps + 1
        self.step_timesteps = step_timesteps

        self.solutions_statistics_filename = solutions_statistics_filename


class EvolutionaryTrainer:
    INSTANCE = None

    def __init__(self, data_repository: DataRepository = IndicatorsDataRepository(AlpacaOHLCVDataRepository())):
        if EvolutionaryTrainer.INSTANCE is not None:
            raise ValueError("There can be only one instance of Trainer")

        EvolutionaryTrainer.INSTANCE = self

        self.data_repository = data_repository

        self.training_configuration: TrainingConfiguration = None
        self.fitness_cache = None
        self.solutions_statistics: pd.DataFrame = None
        self.current_population = None
        self.current_timesteps = None
        self.training_timestamp = None
        self._dna_brain_config_mapper = DnaDecoder(features_per_symbol=self.data_repository.get_columns_per_symbol())

    def train(self, training_configuration: TrainingConfiguration) -> BrainConfiguration:
        self.fitness_cache = {}
        self.solutions_statistics = pd.DataFrame(columns=['Profit', 'Brain Configuration'])
        self.training_configuration = training_configuration
        self.current_population = self.training_configuration.start_population
        self.current_timesteps = self.training_configuration.start_timesteps
        self.training_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self._initialize_genetic_algorithm()

        self.genetic_algorithm.run()
        return self._dna_brain_config_mapper.get_brain_configuration_from_dna(self.genetic_algorithm.best_solutions[0])

    def _initialize_genetic_algorithm(self):
        genes = self._dna_brain_config_mapper.get_genes_size()
        self.genetic_algorithm = pygad.GA(num_generations=self.training_configuration.generations,
                                          num_genes=genes,
                                          init_range_low=0.0,
                                          init_range_high=1.0,
                                          sol_per_pop=self.current_population,
                                          save_best_solutions=True,

                                          fitness_func=EvolutionaryTrainer._evaluate_fitness,
                                          on_generation=EvolutionaryTrainer._on_generation,

                                          crossover_type='uniform',
                                          num_parents_mating=int(self.current_population / 2.0),
                                          keep_parents=0,

                                          mutation_type='random',
                                          mutation_by_replacement=True,
                                          mutation_percent_genes=30.0,
                                          random_mutation_min_val=0.0,
                                          random_mutation_max_val=1.0)

    @staticmethod
    def _on_generation(ga_instance):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE

        logger.info(f"Generation {trainer.genetic_algorithm.generations_completed} finished. Here are the results:")
        logger.info('\t' + trainer.solutions_statistics.to_string().replace('\n', '\n\t'))

        # Update Population
        current_population_values = list(range(trainer.training_configuration.start_population,
                                               trainer.training_configuration.stop_population,
                                               trainer.training_configuration.step_population))
        current_population_idx = min(trainer.genetic_algorithm.generations_completed,
                                     len(current_population_values) - 1)
        trainer.current_population = current_population_values[current_population_idx]
        trainer.genetic_algorithm.num_offspring = trainer.current_population

        # Update Parents
        trainer.genetic_algorithm.num_parents_mating = int(trainer.genetic_algorithm.num_offspring / 2.0)

        # Update Timesteps
        current_timesteps_values = list(range(trainer.training_configuration.start_timesteps,
                                              trainer.training_configuration.stop_timesteps,
                                              trainer.training_configuration.step_timesteps))
        current_timesteps_idx = min(trainer.genetic_algorithm.generations_completed,
                                    len(current_timesteps_values) - 1)
        trainer.current_timesteps = current_timesteps_values[current_timesteps_idx]

        logger.info(f"New population: {trainer.current_population}. "
                    f"New parents: {trainer.genetic_algorithm.num_parents_mating}. "
                    f"New timesteps: {trainer.current_timesteps}")

    @staticmethod
    def _evaluate_fitness(solution, solution_idx):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE

        solution_name = f"{trainer.genetic_algorithm.generations_completed}_{solution_idx}"
        logger.info(f"Solution {solution_name}")

        brain_configuration = trainer._dna_brain_config_mapper.get_brain_configuration_from_dna(solution)
        logger.info(f"Brain Configuration: {brain_configuration}")

        if brain_configuration is None:
            return -math.inf

        if solution_name not in trainer.fitness_cache:
            logger.info("Fitness not in cache, calculating fitness...")

            brain: Brain = Brain(data_repository=trainer.data_repository,
                                 brain_configuration=brain_configuration)

            for training_scenario in trainer.training_configuration.training_scenarios:
                # FIXME: Since the scenario might not have Symbols nor Interval, it prints "None_None"
                logger.info(f"Training on scenario {training_scenario} with {trainer.current_timesteps} timesteps")
                brain.learn(training_scenario=training_scenario,
                            total_timesteps=trainer.current_timesteps)
                logger.info(f"Training finished")

            mean_testing_profit = 0.0
            for validation_scenario in trainer.training_configuration.validation_scenarios:
                logger.info(f"Testing on scenario {validation_scenario}")
                info = brain.evaluate(testing_scenario=validation_scenario)
                profit = info['total_profit'] if 'total_profit' in info else info['current_profit']
                logger.info(f"Testing finished with profit {profit}")
                mean_testing_profit += profit

            mean_testing_profit = mean_testing_profit / len(trainer.training_configuration.validation_scenarios)

            trainer.solutions_statistics.at[solution_name, 'Profit'] = mean_testing_profit
            trainer.solutions_statistics.at[solution_name, 'Brain Configuration'] = brain_configuration
            trainer.solutions_statistics.sort_values(by='Profit', inplace=True, ascending=False)
            if trainer.training_configuration.solutions_statistics_filename is not None:
                trainer.solutions_statistics.to_csv(f"{trainer.training_configuration.solutions_statistics_filename}-{trainer.training_timestamp}.csv")

            logger.info(f"Evaluation Profit: {mean_testing_profit}")
            trainer.fitness_cache[solution_name] = mean_testing_profit

            return mean_testing_profit
        else:
            logger.info("Fitness in cache, returning saved fitness...")
            return trainer.fitness_cache[solution_name]
