import alpaca_trade_api as tradeapi
import math
import json
import logging
import logging.config

from drltrader.observers import Observer, Order

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class AlpacaObserver(Observer):
    def __init__(self,
                 config_file_name: str = 'config.json'):
        with open(config_file_name) as config_file:
            config = json.load(config_file)

        self._alpaca_key = config['alpaca']['key']
        self._alpaca_secret = config['alpaca']['secret']
        self._alpaca_url = config['alpaca']['url']

        self._alpaca_api = tradeapi.REST(self._alpaca_key,
                                         self._alpaca_secret,
                                         self._alpaca_url,
                                         api_version='v2')

        # Test it works
        logger.info(f"The status of the alpaca account is {self._alpaca_api.get_account()}")

    def notify_portfolio_change(self, old_portfolio: dict, new_portfolio: dict):
        self._update_portfolio(portfolio=new_portfolio)

    def notify_begin_of_observation(self, portfolio: dict):
        self._update_portfolio(portfolio=portfolio)

    def _update_portfolio(self, portfolio: dict):
        # TODO: For now we are assuming only 1 symbol can be found
        non_marginable_buying_power = float(self._alpaca_api.get_account().non_marginable_buying_power)

        selected_symbol = None
        selected_side = None
        for symbol in portfolio:
            if portfolio[symbol] > 0.0:
                selected_symbol = symbol
                selected_side = 'long'
            elif portfolio[symbol] < 0.0:
                selected_symbol = symbol
                selected_side = 'short'

        for position in self._alpaca_api.list_positions():
            if position.symbol != selected_symbol and int(position.qty_available) != 0:
                self._alpaca_api.close_position(symbol=position.symbol)

        current_stock_price = float(self._alpaca_api.get_latest_bar(selected_symbol).c)
        new_stocks = int(non_marginable_buying_power / current_stock_price)

        if new_stocks == 0:
            return

        if selected_side == 'long':
            order = self._alpaca_api.submit_order(symbol=selected_symbol,
                                                  qty=new_stocks,
                                                  side='buy',
                                                  type="market",
                                                  time_in_force="day")
        elif selected_side == 'short':
            order = self._alpaca_api.submit_order(symbol=selected_symbol,
                                                  qty=new_stocks,
                                                  side='sell',
                                                  type="market",
                                                  time_in_force="day")

        logger.info(f"Order to {order.side} {order.qty} {order.symbol} stocks submitted")
        logger.debug(order.__dict__)
