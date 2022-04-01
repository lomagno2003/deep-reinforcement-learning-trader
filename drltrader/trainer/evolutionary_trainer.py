import pygad
import json
from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import BrainConfiguration
from drltrader.brain.brain import Brain
from drltrader.data.data_provider import DataProvider
from drltrader.data.scenario import Scenario


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

        genes = len(self.data_provider.indicator_column_names) + 3
        self.genetic_algorithm = pygad.GA(num_generations=2,
                                          num_genes=genes,
                                          init_range_low=0.0,
                                          init_range_high=1.0,
                                          sol_per_pop=2,
                                          save_best_solutions=True,

                                          fitness_func=EvolutionaryTrainer._evaluate_fitness,
                                          on_generation=EvolutionaryTrainer._on_generation,

                                          crossover_type='uniform',
                                          num_parents_mating=2,
                                          keep_parents=2,

                                          mutation_type='random',
                                          mutation_by_replacement=True,
                                          mutation_percent_genes=30.0,
                                          random_mutation_min_val=0.0,
                                          random_mutation_max_val=1.0)

        # TODO: Clean this
        self.training_scenario = Scenario(symbol='TSLA',
                                          start_date=datetime.now() - timedelta(days=30),
                                          end_date=datetime.now())
        self.testing_scenario = Scenario(symbol='TSLA',
                                         start_date=datetime.now() - timedelta(days=1),
                                         end_date=datetime.now())

    def train(self) -> BrainConfiguration:
        self.genetic_algorithm.run()

        return EvolutionaryTrainer._get_brain_configuration_from_dna(self, self.genetic_algorithm.best_solutions[0])

    @staticmethod
    def _on_generation(ga_instance):
        pass

    @staticmethod
    def _evaluate_fitness(solution, solution_idx):
        trainer: EvolutionaryTrainer = EvolutionaryTrainer.INSTANCE
        brain_configuration = EvolutionaryTrainer._get_brain_configuration_from_dna(trainer, solution)

        brain: Brain = Brain(data_provider=trainer.data_provider,
                             brain_configuration=brain_configuration)

        brain.learn(training_scenario=trainer.training_scenario,
                    total_timesteps=1000)

        results = brain.test(testing_scenario=trainer.testing_scenario)

        print(f"Brain Configuration: {json.dumps(brain_configuration.__dict__)}")
        print(f"Evaluation Profit: {results['total_profit']}")

        return results['total_profit']

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