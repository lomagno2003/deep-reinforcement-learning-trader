import alpaca_trade_api as tradeapi
import json
import logging


from drltrader.observers import Observer, Order

logging.basicConfig(format='%(asctime)s %(message)s',
                    filename='logs/training.log',
                    encoding='utf-8',
                    level=logging.DEBUG)


class AlpacaObserver(Observer):
    def __init__(self,
                 config_file_name: str = 'config.json'):
        with open(config_file_name) as config_file:
            config = json.load(config_file)

        self._alpaca_key = config['alpaca']['key']
        self._alpaca_secret = config['alpaca']['secret']
        self._alpaca_url = config['alpaca']['url']

        self._alpaca_api = api = tradeapi.REST(self._alpaca_key,
                                               self._alpaca_secret,
                                               self._alpaca_url,
                                               api_version='v2')

        # Test it works
        logging.info(f"The status of the alpaca account is {api.get_account()}")

    def notify_order(self, order: Order):
        quantity = str(int(10000.0 / order.price))

        order = self._alpaca_api.submit_order(symbol=order.symbol,
                                              qty=quantity,
                                              side=order.side,
                                              type="market",
                                              time_in_force="day")

        logging.info(f"Order to {order.side} {quantity} {order.symbol} stocks submitted")
        logging.debug(order.__dict__)
