from datetime import datetime
from datetime import timedelta

from drltrader.data.data_provider import DataProvider, Scenario
from drltrader.brain.brain import Brain, BrainConfiguration
from drltrader.trainer.evolutionary_trainer import EvolutionaryTrainer, TrainingConfiguration


class TrainingRunner:
    def __init__(self):
        self._symbols = ['SPY', 'TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP']
        self._initiate_scenarios()
        self._initiate_training_configuration()

        self._data_provider = DataProvider()

    def run(self):
        # Find best brain configuration
        print("Finding best brain configuration")
        trainer: EvolutionaryTrainer = EvolutionaryTrainer(data_provider=self._data_provider)
        best_brain_configuration: BrainConfiguration = trainer.train(self._training_configuration)

        # Train brain
        print("Training best brain")
        best_brain = Brain(data_provider=self._data_provider,
                           brain_configuration=best_brain_configuration)
        best_brain.learn(self._training_scenarios[0], total_timesteps=200000)

        # Save brain
        print("Saving best brain")
        best_brain.save("temp/best_brain", override=True)

    def _initiate_scenarios(self):
        self._training_scenarios = [Scenario(symbols=self._symbols,
                                             interval='1d',
                                             start_date=datetime.now() - timedelta(days=720),
                                             end_date=datetime.now() - timedelta(days=50))]
        self._testing_scenarios = [Scenario(symbols=self._symbols,
                                            interval='1d',
                                            start_date=datetime.now() - timedelta(days=90),
                                            end_date=datetime.now() - timedelta(days=10))]

    def _initiate_training_configuration(self):
        self._training_configuration = TrainingConfiguration(training_scenarios=self._training_scenarios,
                                                             testing_scenarios=self._testing_scenarios,
                                                             generations=20,
                                                             start_population=15,
                                                             stop_population=6,
                                                             step_population=-1,
                                                             start_timesteps=1000,
                                                             stop_timesteps=20000,
                                                             step_timesteps=500,
                                                             solutions_statistics_filename='logs/solutions.csv')


if __name__ == '__main__':
    training_runner: TrainingRunner = TrainingRunner()

    training_runner.run()
