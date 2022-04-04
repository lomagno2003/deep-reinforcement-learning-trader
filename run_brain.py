from threading import Thread
from datetime import datetime
from datetime import timedelta
from flask import Flask

from drltrader.brain.brain import Brain
from drltrader.brain.brain_repository_file import BrainRepositoryFile
from drltrader.observers.simple_observer import CompositeObserver
from drltrader.observers.alpaca_observer import AlpacaObserver
from drltrader.observers.telegram_observer import TelegramObserver
from drltrader.data.data_provider import Scenario

app = Flask("GCR Port Listener")


class BrainRunner:
    @staticmethod
    def run():
        # Load Brain
        print("Loading brain")
        symbols = ['SPY', 'TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP']

        brain: Brain = BrainRepositoryFile().load("best_brain")

        # Start Observing
        print("Starting observation")
        start_date = datetime.now() - timedelta(days=30)
        observation_scenario: Scenario = Scenario(symbols=symbols,
                                                  start_date=start_date)
        brain.start_observing(scenario=observation_scenario,
                              observer=CompositeObserver([AlpacaObserver(), TelegramObserver()]))

        print("Finish observation")

    def launch_brain_async(self):
        thread = Thread(target=BrainRunner.run, args=())
        thread.start()

    # This needs to be done to avoid failures on GCR
    def launch_flask_and_block(self):
        port = 8080
        print(f"Listening to port {port}")
        app.run(debug=True, host="0.0.0.0", port=port)

    @staticmethod
    @app.route("/")
    def http_listener():
        return "I'm still running..."


if __name__ == '__main__':
    training_runner: BrainRunner = BrainRunner()
    training_runner.launch_brain_async()
    training_runner.launch_flask_and_block()
