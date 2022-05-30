import unittest

from drltrader.observers.alpaca_observer import AlpacaObserver


class TelegramObserverTestCase(unittest.TestCase):
    def test_init(self):
        # Act
        alpaca_observer: AlpacaObserver = AlpacaObserver()

        # Assert
        self.assertIsNotNone(alpaca_observer)

    def test_notify_begin_of_observation(self):
        # Arrange
        alpaca_observer: AlpacaObserver = AlpacaObserver()

        # Act
        alpaca_observer.notify_begin_of_observation(portfolio={'TSLA': 10})

        # Assert
        # FIXME: In here you'll need to check the alpaca API to see if it worked
        # FIXME: This test might fail if run outside of market hours since the orders are accepted but never processed

if __name__ == '__main__':
    unittest.main()
