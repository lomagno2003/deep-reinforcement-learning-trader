import unittest
import time

from drltrader.observers import Order, Sides
from drltrader.observers.alpaca_observer import AlpacaObserver


class TelegramObserverTestCase(unittest.TestCase):
    def test_init(self):
        # Act
        alpaca_observer: AlpacaObserver = AlpacaObserver()

        # Assert
        self.assertIsNotNone(alpaca_observer)

    def test_notify_stock_buy_and_notify_stock_sell(self):
        # Arrange
        alpaca_observer: AlpacaObserver = AlpacaObserver()

        # Act/Assert
        alpaca_observer.notify_order(Order(symbol='FB', qty=1.0, price=225.24, side=Sides.Buy))
        time.sleep(3)
        alpaca_observer.notify_order(Order(symbol='FB', qty=1.0, price=225.24, side=Sides.Sell))

        # Assert
        # FIXME: In here you'll need to check the alpaca API to see if it worked
        # FIXME: This test might fail if run outside of market hours since the orders are accepted but never processed


if __name__ == '__main__':
    unittest.main()
