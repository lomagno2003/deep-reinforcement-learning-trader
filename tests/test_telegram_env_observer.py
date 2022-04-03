import unittest

from drltrader.envs.observers.telegram_env_observer import TelegramEnvObserver


class TelegramEnvObserverTestCase(unittest.TestCase):
    def test_notify_stock_buy(self):
        # Arrange
        telegram_env_observer: TelegramEnvObserver = TelegramEnvObserver()

        # Act/Assert
        telegram_env_observer.notify_stock_sell('TSLA', 1.0, 1.0)

        # Assert
        # FIXME: In here you'll need to check the group to ensure it was sent

    def test_start_stop_polling(self):
        # Arrange
        telegram_env_observer: TelegramEnvObserver = TelegramEnvObserver()

        # Act/Assert
        telegram_env_observer.start_polling()
        telegram_env_observer.stop_polling()


if __name__ == '__main__':
    unittest.main()
