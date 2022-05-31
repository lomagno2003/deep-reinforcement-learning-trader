import gym
import numpy as np
import pandas as pd
import logging
import logging.config
from gym import spaces
import matplotlib.pyplot as plt

from drltrader.observers import Observer, Order, Sides

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class PortfolioStocksEnv(gym.Env):
    ALLOCATION_PENALTY_LONG = 0.002
    ALLOCATION_PENALTY_SHORT = 0.003

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 window_size: int,
                 dataframe_per_symbol: dict,
                 initial_portfolio_allocation: dict,
                 prices_feature_name: str = 'Close',
                 signal_feature_names: list = ['RSI_4', 'RSI_16']):
        super(PortfolioStocksEnv, self).__init__()

        # Save Configurations
        self._window_size = window_size
        self._initial_portfolio_allocation_size = initial_portfolio_allocation
        self._dataframe_per_symbol = dataframe_per_symbol
        self._prices_feature_name = prices_feature_name
        self._signal_feature_names = signal_feature_names

        # Initialize Custom Configurations
        self._reset_enabled = True
        self._process_dataframe_per_symbol()

        self._action_to_symbol = {}
        self._action_to_side = {}
        for action in range(len(dataframe_per_symbol)):
            self._action_to_symbol[action * 2] = list(dataframe_per_symbol.keys())[action]
            self._action_to_side[action * 2] = 'long'

            self._action_to_symbol[action * 2 + 1] = list(dataframe_per_symbol.keys())[action]
            self._action_to_side[action * 2 + 1] = 'short'

        # Initialize Gym Configurations
        self.action_space = spaces.Discrete(len(self._action_to_symbol))
        self.shape = self._get_observation(self._window_size).shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # Initialize Runtime Variables
        self._observer = None
        self._done = None
        self._current_tick = None
        self._portfolio_allocation_size = None
        self._portfolio_allocation_tick = None
        self._allocations_history = None

        self.reset()

    def append_data(self, dataframe_per_symbol: dict):
        if self._observer is not None:
            self._observer.notify_new_data()

        new_dataframe_per_symbol = {}

        for symbol in dataframe_per_symbol:
            old_data_for_symbol = self._dataframe_per_symbol[symbol]
            new_data_for_symbol = dataframe_per_symbol[symbol].loc[self._timestamps[-1]:, ].iloc[1:, :]
            new_dataframe_per_symbol[symbol] = pd.concat([old_data_for_symbol, new_data_for_symbol])

        self._dataframe_per_symbol = new_dataframe_per_symbol
        self._process_dataframe_per_symbol()

        self._done = self._current_tick >= self._frame_bound[1]

        if not self._done:
            logger.info("New data was added and the brain can check it out")

    def observe(self, observer: Observer):
        self._observer = observer
        if observer is not None:
            self._observer.notify_begin_of_observation(self._portfolio_allocation_size)

    def step(self, action):
        if self._done:
            return None, 0.0, self._done, self._get_info()

        self._current_tick += 1

        # Process Action
        selected_symbol = self._action_to_symbol[action]
        selected_side = self._action_to_side[action]
        allocated_symbol, allocated_side = self._get_allocated_position()

        if selected_symbol != allocated_symbol or selected_side != allocated_side:
            self._transfer_allocations(allocated_symbol, allocated_side, selected_symbol, selected_side, self._current_tick)

        # Calculate Gym Responses
        return self.get_step_outputs()

    def get_step_outputs(self):
        self._done = self._current_tick == self._frame_bound[1]
        observation = self._get_observation(self._current_tick)
        reward = self.current_profit()

        return observation, reward, self._done, self._get_info()

    def reset(self):
        self._done = False
        if not self._reset_enabled:
            return

        self._current_tick = self._frame_bound[0]
        self._portfolio_allocation_size = {}
        self._portfolio_allocation_tick = {}
        self._allocations_history = []

        for symbol in self._dataframe_per_symbol:
            self._portfolio_allocation_size[symbol] = 0.0
            self._portfolio_allocation_tick[symbol] = None

        for symbol in self._initial_portfolio_allocation_size:
            self._portfolio_allocation_size[symbol] = self._initial_portfolio_allocation_size[symbol]
            self._portfolio_allocation_tick[symbol] = self._frame_bound[0]

        return self._get_observation(self._current_tick)

    def disable_reset(self):
        self._reset_enabled = False

    def render(self, mode='human'):
        # TODO: Stub-method
        pass

    def render_all(self):
        for symbol in self._prices_per_symbol:
            plt.plot(self._prices_per_symbol[symbol], label=symbol)

        allocated = []
        allocated_prices = []

        deallocated = []
        deallocated_prices = []
        for allocation_details in self._allocations_history:
            deallocated.append(allocation_details['allocation_tick'])
            deallocated_prices.append(allocation_details['source_symbol_price'])
            allocated.append(allocation_details['allocation_tick'])
            allocated_prices.append(allocation_details['target_symbol_price'])

        plt.plot(allocated, allocated_prices, 'go')
        plt.plot(deallocated, deallocated_prices, 'ro')

        plt.suptitle(
            "Total Profit: %.6f" % self.current_profit()
        )
        plt.legend()

    def close(self):
        # TODO: Stub-method
        pass

    def initial_portfolio_value(self):
        initial_portfolio_allocation_tick = {}
        for symbol in self._initial_portfolio_allocation_size:
            if self._initial_portfolio_allocation_size[symbol] > 0:
                initial_portfolio_allocation_tick[symbol] = self._frame_bound[0]
            else:
                initial_portfolio_allocation_tick[symbol] = None

        return self._portfolio_value(self._initial_portfolio_allocation_size,
                                     initial_portfolio_allocation_tick,
                                     self._frame_bound[0])

    def current_portfolio_value(self):
        return self._portfolio_value(self._portfolio_allocation_size,
                                     self._portfolio_allocation_tick,
                                     self._current_tick)

    def current_profit(self):
        return self.current_portfolio_value() / self.initial_portfolio_value()

    def _process_dataframe_per_symbol(self):
        timestamps = self._dataframe_per_symbol[list(self._dataframe_per_symbol.keys())[0]].index.tolist()

        first_symbol = list(self._dataframe_per_symbol.keys())[0]
        self._frame_bound = (self._window_size, len(self._dataframe_per_symbol[first_symbol].index) - 1)

        start = self._frame_bound[0] - self._window_size
        end = self._frame_bound[1]

        prices_per_symbol = {}
        signal_features_per_symbol = {}

        for symbol in self._dataframe_per_symbol:
            symbol_dataframe = self._dataframe_per_symbol[symbol]
            prices_per_symbol[symbol] = symbol_dataframe[self._prices_feature_name].to_numpy()
            signal_features_per_symbol[symbol] = symbol_dataframe.loc[:, self._signal_feature_names].to_numpy()[start:end]

        # FIXME: If prices_per_symbol[ANY] <  self._window_size, the model will fail since
        #  observation < observation_space
        self._timestamps = timestamps
        self._prices_per_symbol = prices_per_symbol
        self._signal_features_per_symbol = signal_features_per_symbol

    def _transfer_allocations(self, source_symbol, source_side, target_symbol, target_side, allocation_tick):
        # Temporary Variables
        old_portfolio_allocation_size = self._portfolio_allocation_size

        source_symbol_original_price = self._prices_per_symbol[source_symbol][self._portfolio_allocation_tick[source_symbol]]
        source_symbol_shares = self._portfolio_allocation_size[source_symbol]
        source_symbol_price = self._prices_per_symbol[source_symbol][allocation_tick]

        # Deallocation
        if source_side == 'long':
            deallocated_funds = source_symbol_shares * source_symbol_price
            self._portfolio_allocation_size[source_symbol] = 0.0
        else:
            source_symbol_original_market_value = -2 * source_symbol_shares * source_symbol_original_price
            source_symbol_market_value = -1 * source_symbol_shares * source_symbol_price
            deallocated_funds = source_symbol_original_market_value - source_symbol_market_value
            self._portfolio_allocation_size[source_symbol] = 0.0

        # Penalty
        penalty = PortfolioStocksEnv.ALLOCATION_PENALTY_LONG if target_side == 'long' \
            else PortfolioStocksEnv.ALLOCATION_PENALTY_SHORT
        deallocated_funds = deallocated_funds * (1.0 - penalty)

        # Reallocation
        target_symbol_original_shares = self._portfolio_allocation_size[target_symbol]
        target_symbol_price = self._prices_per_symbol[target_symbol][allocation_tick]

        target_symbol_sign = 1.0 if target_side == 'long' else -1.0
        target_symbol_new_shares = target_symbol_sign * deallocated_funds / target_symbol_price
        self._portfolio_allocation_size[target_symbol] = target_symbol_original_shares + target_symbol_new_shares
        self._portfolio_allocation_tick[source_symbol] = None
        self._portfolio_allocation_tick[target_symbol] = allocation_tick

        # Notify Observer
        if self._observer is not None:
            self._observer.notify_portfolio_change(old_portfolio=old_portfolio_allocation_size,
                                                   new_portfolio=self._portfolio_allocation_size)

        # Statistics
        allocation_details = {
            'allocation_tick': allocation_tick,

            'source_symbol': source_symbol,
            'source_symbol_price': source_symbol_price,
            'source_symbol_shares': source_symbol_shares,

            'target_symbol': target_symbol,
            'target_symbol_price': target_symbol_price,
            'target_symbol_original_shares': target_symbol_original_shares,
            'target_symbol_new_shares': target_symbol_new_shares,
        }
        self._allocations_history.append(allocation_details)

    def _portfolio_value(self, portfolio_allocation_count, portfolio_allocation_tick, current_tick):
        portfolio_value = 0.0

        for symbol in portfolio_allocation_count:
            symbol_shares = portfolio_allocation_count[symbol]
            if portfolio_allocation_tick[symbol] is not None:
                symbol_original_price = self._prices_per_symbol[symbol][portfolio_allocation_tick[symbol]]
            else:
                symbol_original_price = None

            symbol_current_price = self._prices_per_symbol[symbol][current_tick]

            if symbol_shares >= 0:
                portfolio_value += symbol_shares * symbol_current_price
            else:
                portfolio_value += -1 * (symbol_original_price * 2 - symbol_current_price) * symbol_shares

        return portfolio_value

    def _get_info(self):
        return {
            'current_profit': self.current_profit(),
            'current_portfolio_value': self.current_portfolio_value(),
            'portfolio_allocation': self._portfolio_allocation_size,
            'allocations_history': self._allocations_history
        }

    def _get_observation(self, current_tick):
        result = None

        for symbol in self._signal_features_per_symbol:
            signals_for_symbol = self._signal_features_per_symbol[symbol]
            signals_for_symbol_for_tick = signals_for_symbol[(current_tick - self._window_size):current_tick]
            result = signals_for_symbol_for_tick if result is None else np.hstack([result, signals_for_symbol_for_tick])

        return np.array(result).transpose()

    def _get_allocated_position(self):
        # FIXME: This function will go away once multiple allocations are allowed
        for symbol in self._portfolio_allocation_size:
            if self._portfolio_allocation_size[symbol] > 0.0:
                return symbol, 'long'
            elif self._portfolio_allocation_size[symbol] < 0.0:
                return symbol, 'short'

        raise ValueError("There are no allocated symbols")
