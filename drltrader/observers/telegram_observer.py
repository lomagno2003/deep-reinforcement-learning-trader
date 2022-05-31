from telegram import *
from telegram.ext import *
import json

from drltrader.observers import Observer, Order


class TelegramObserver(Observer):
    def __init__(self,
                 config_file_name: str = 'config.json'):
        with open(config_file_name) as config_file:
            config = json.load(config_file)

        self._telegram_token = config['telegram']['token']
        self._char_group_id = config['telegram']['chat_id']
        self._updater = Updater(token=self._telegram_token)

        self.send_message_to_group(text="Bot started, waiting for portfolio updates")

    def notify_new_data(self):
        self.send_message_to_group(text="There's new data available")

    def notify_portfolio_change(self, old_portfolio: dict, new_portfolio: dict):
        text = f"Portfolio changed from {str(old_portfolio)} to {str(new_portfolio)}"
        self.send_message_to_group(text)

    def notify_begin_of_observation(self, portfolio: dict):
        text = f"Portfolio changed to: {str(portfolio)}"
        self.send_message_to_group(text)

    def start_polling(self):
        dispatcher = self._updater.dispatcher
        dispatcher.add_handler(CommandHandler("portfolio", TelegramObserver.portfolio_command))
        self._updater.start_polling()

    def stop_polling(self):
        self._updater.stop()

    @staticmethod
    def portfolio_command(update: Update, context: CallbackContext):
        context.bot.send_message(chat_id=update.effective_chat.id, text="Not implemented yet")

    def send_message_to_group(self, text: str):
        self._updater.bot.send_message(chat_id=self._char_group_id, text=text)


