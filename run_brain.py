from datetime import datetime
from datetime import timedelta

from drltrader.brain.brain import Brain
from drltrader.observers.simple_observer import CompositeObserver
from drltrader.observers.alpaca_observer import AlpacaObserver
from drltrader.observers.telegram_observer import TelegramObserver
from drltrader.data.data_provider import DataProvider, Scenario


class BrainRunner:
    def __init__(self):
        self._symbols = ['TSLA', 'AAPL', 'MSFT', 'SPY', 'SHOP']

    def run(self):
        # Load Brain
        print("Loading brain")
        brain: Brain = Brain.load("temp/best_brain")

        # Start Observing
        print("Starting observation")
        start_date = datetime.now() - timedelta(days=30)
        observation_scenario: Scenario = Scenario(symbols=self._symbols,
                                                  start_date=start_date)
        brain.start_observing(scenario=observation_scenario,
                              observer=CompositeObserver([AlpacaObserver(), TelegramObserver()]))

        print("Finish observation")


if __name__ == '__main__':
    training_runner: BrainRunner = BrainRunner()

    training_runner.run()
