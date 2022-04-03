import unittest
import time

from drltrader.envs.observers.alpaca_env_observer import AlpacaEnvObserver


class TelegramEnvObserverTestCase(unittest.TestCase):
    def test_init(self):
        # Act
        alpaca_env_observer: AlpacaEnvObserver = AlpacaEnvObserver()

        # Assert
        self.assertIsNotNone(alpaca_env_observer)

    def test_notify_stock_buy_and_notify_stock_sell(self):
        # Arrange
        alpaca_env_observer: AlpacaEnvObserver = AlpacaEnvObserver()

        # Act/Assert
        alpaca_env_observer.notify_stock_buy('FB', 1.0, 1.0)
        time.sleep(10)
        alpaca_env_observer.notify_stock_sell('FB', 1.0, 1.0)

        # Assert
        # FIXME: In here you'll need to check the alpaca API to see if it worked


if __name__ == '__main__':
    unittest.main()
