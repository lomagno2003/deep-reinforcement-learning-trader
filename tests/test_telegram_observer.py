import unittest

from drltrader.observers import Order, Sides
from drltrader.observers.telegram_observer import TelegramEnvObserver


class TelegramEnvObserverTestCase(unittest.TestCase):
    def test_notify_stock_buy(self):
        # Arrange
        telegram_observer: TelegramEnvObserver = TelegramEnvObserver()

        # Act/Assert
        telegram_observer.notify_order(Order(symbol='TSLA', qty=1.0, price=1.0, side=Sides.Buy))

        # Assert
        # FIXME: In here you'll need to check the group to ensure it was sent

    def test_start_stop_polling(self):
        # Arrange
        telegram_observer: TelegramEnvObserver = TelegramEnvObserver()

        # Act/Assert
        telegram_observer.start_polling()
        telegram_observer.stop_polling()


if __name__ == '__main__':
    unittest.main()
