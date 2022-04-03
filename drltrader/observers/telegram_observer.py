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

    def notify_order(self, order: Order):
        text = f"There was an order to {order.side} {order.qty} stocks of {order.symbol}"
        self._updater.bot.send_message(chat_id=self._char_group_id,
                                       text=text)

    def start_polling(self):
        dispatcher = self._updater.dispatcher
        dispatcher.add_handler(CommandHandler("portfolio", TelegramObserver.portfolio_command))
        self._updater.start_polling()

    def stop_polling(self):
        self._updater.stop()

    @staticmethod
    def portfolio_command(update: Update, context: CallbackContext):
        context.bot.send_message(chat_id=update.effective_chat.id, text="Not implemented yet")

