import logging
import logging.config
from threading import Thread
from datetime import datetime
from datetime import timedelta
from flask import Flask

from drltrader.brain.brain import Brain
from drltrader.brain.brain_repository_file import BrainRepositoryFile
from drltrader.observers.simple_observer import CompositeObserver
from drltrader.observers.alpaca_observer import AlpacaObserver
from drltrader.observers.telegram_observer import TelegramObserver
from drltrader.data import Scenario

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

app = Flask("GCR Port Listener")


class BrainRunner:
    @staticmethod
    def run_brain():
        # Load Brain
        logger.info("Loading brain")
        symbols = ['SPY', 'TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP']

        brain: Brain = BrainRepositoryFile().load("best_brain")

        # Start Observing
        logger.info("Starting observation")
        start_date = datetime.now() - timedelta(days=30)
        observation_scenario: Scenario = Scenario(symbols=symbols,
                                                  start_date=start_date,
                                                  interval='1h')
        brain.start_observing(scenario=observation_scenario,
                              observer=CompositeObserver([AlpacaObserver(), TelegramObserver()]))

        logger.info("Finish observation")

    @staticmethod
    def run_flask():
        port = 8080
        logger.info(f"Listening to port {port}")
        app.run(debug=False, host="0.0.0.0", port=port)

    def launch_brain_async(self):
        thread = Thread(target=BrainRunner.run_brain, args=())
        thread.start()

        return thread

    # This needs to be done to avoid failures on GCR
    def launch_flask_and_block(self):
        thread = Thread(target=BrainRunner.run_flask, args=())
        thread.start()

        return thread

    @staticmethod
    @app.route("/")
    def http_listener():
        return "I'm still running..."


if __name__ == '__main__':
    training_runner: BrainRunner = BrainRunner()
    brain_thread = training_runner.launch_brain_async()
    flask_thread = training_runner.launch_flask_and_block()

    brain_thread.join()
    flask_thread.join()
