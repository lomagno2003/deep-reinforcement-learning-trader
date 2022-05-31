import logging
import logging.config
import traceback

from drltrader.observers import Observer, Order

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class PrintEnvObserver(Observer):
    def notify_new_data(self):
        print(f"There's new data available")

    def notify_portfolio_change(self, old_portfolio: dict, new_portfolio: dict):
        print(f"There was a portfolio change from {str(old_portfolio)} to {str(new_portfolio)}")

    def notify_begin_of_observation(self, portfolio: dict):
        print(f"There was a portfolio change: {str(portfolio)}")


class CallbackObserver(Observer):
    def __init__(self,
                 new_data_callback_function=None,
                 order_callback_function=None,
                 portfolio_callback_function=None):
        self._new_data_callback_function = new_data_callback_function
        self._order_callback_function = order_callback_function
        self._portfolio_callback_function = portfolio_callback_function

    def notify_new_data(self):
        self._new_data_callback_function

    def notify_portfolio_change(self, old_portfolio: dict, new_portfolio: dict):
        self._portfolio_callback_function(old_portfolio, new_portfolio)

    def notify_begin_of_observation(self, portfolio: dict):
        self._portfolio_callback_function(portfolio)


class CompositeObserver(Observer):
    def __init__(self, observers: list):
        self._observers = observers

    def notify_new_data(self):
        for observer in self._observers:
            observer.notify_new_data()

    def notify_portfolio_change(self, old_portfolio: dict, new_portfolio: dict):
        for observer in self._observers:
            observer.notify_portfolio_change(old_portfolio, new_portfolio)

    def notify_begin_of_observation(self, portfolio: dict):
        for observer in self._observers:
            observer.notify_portfolio_change(portfolio)


class SafeObserver(Observer):
    def __init__(self, observer: Observer):
        self._observer = observer

    def notify_new_data(self):
        try:
            self._observer.notify_new_data()
        except:
            logging.warning(traceback.format_exc())

    def notify_portfolio_change(self, old_portfolio: dict, new_portfolio: dict):
        try:
            self._observer.notify_portfolio_change(old_portfolio, new_portfolio)
        except:
            logging.warning(traceback.format_exc())

    def notify_begin_of_observation(self, portfolio: dict):
        try:
            self._observer.notify_portfolio_change(portfolio)
        except:
            logging.warning(traceback.format_exc())
