import gym
import numpy as np
import pandas as pd
from gym import spaces
import matplotlib.pyplot as plt

from drltrader.envs.observers import EnvObserver


class PortfolioStocksEnv(gym.Env):
    ALLOCATION_PENALTY = 0.002

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 window_size: int,
                 dataframe_per_symbol: dict,
                 initial_portfolio_allocation: dict,
                 prices_feature_name: str = 'Close',
                 signal_feature_names: list = ['RSI_4', 'RSI_16'],
                 env_observer: EnvObserver = None):
        super(PortfolioStocksEnv, self).__init__()

        # Save Configurations
        self._window_size = window_size
        self._initial_portfolio_allocation = initial_portfolio_allocation
        self._dataframe_per_symbol = dataframe_per_symbol
        self._prices_feature_name = prices_feature_name
        self._signal_feature_names = signal_feature_names
        self._env_observer = env_observer

        # Initialize Custom Configurations
        self._reset_enabled = True
        self._process_dataframe_per_symbol()
        self._action_to_symbol = {}
        for symbol in dataframe_per_symbol:
            self._action_to_symbol[len(self._action_to_symbol)] = symbol

        # Initialize Gym Configurations
        self.action_space = spaces.Discrete(len(self._dataframe_per_symbol))
        self.shape = self._get_observation(self._window_size).shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # Initialize Runtime Variables
        self._done = None
        self._current_tick = None
        self._portfolio_allocation = None
        self._allocations_history = None

        self.reset()

    def append_data(self, dataframe_per_symbol: dict):
        new_dataframe_per_symbol = {}

        for symbol in dataframe_per_symbol:
            old_data_for_symbol = self._dataframe_per_symbol[symbol]
            new_data_for_symbol = dataframe_per_symbol[symbol].loc[self._timestamps[-1]:, ].iloc[1:, :]
            new_dataframe_per_symbol[symbol] = pd.concat([old_data_for_symbol, new_data_for_symbol])

        self._dataframe_per_symbol = new_dataframe_per_symbol
        self._process_dataframe_per_symbol()

        self._done = self._current_tick >= self._frame_bound[1]

    def step(self, action):
        if self._done:
            return None, 0.0, self._done, self._get_info()

        self._current_tick += 1

        # Process Action
        selected_symbol = self._action_to_symbol[action]
        allocated_symbol = self._get_allocated_symbol()

        if selected_symbol != allocated_symbol:
            self._transfer_allocations(allocated_symbol, selected_symbol, self._current_tick)

        # Calculate Gym Responses
        return self.get_step_outputs()

    def get_step_outputs(self):
        self._done = self._current_tick == self._frame_bound[1]

        if self._done:
            observation = None
            reward = 0.0
        else:
            observation = self._get_observation(self._current_tick)
            reward = self.current_profit()

        return observation, reward, self._done, self._get_info()

    def reset(self):
        self._done = False
        if not self._reset_enabled:
            return

        self._current_tick = self._frame_bound[0]
        self._portfolio_allocation = {}
        self._allocations_history = []

        for symbol in self._dataframe_per_symbol:
            self._portfolio_allocation[symbol] = 0.0

        for symbol in self._initial_portfolio_allocation:
            self._portfolio_allocation[symbol] = self._initial_portfolio_allocation[symbol]

        return self._get_observation(self._current_tick)

    def disable_reset(self):
        self._reset_enabled = False

    def render(self, mode='human'):
        # TODO: Stub-method
        pass

    def render_all(self):
        for symbol in self._prices_per_symbol:
            plt.plot(self._prices_per_symbol[symbol])

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

    def close(self):
        # TODO: Stub-method
        pass

    def initial_portfolio_value(self):
        return self._portfolio_value(self._initial_portfolio_allocation, self._frame_bound[0])

    def current_portfolio_value(self):
        return self._portfolio_value(self._portfolio_allocation, self._current_tick)

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

        self._timestamps = timestamps
        self._prices_per_symbol = prices_per_symbol
        self._signal_features_per_symbol = signal_features_per_symbol

    def _transfer_allocations(self, source_symbol, target_symbol, allocation_tick):
        # Temporary Variables
        source_symbol_shares = self._portfolio_allocation[source_symbol]
        source_symbol_price = self._prices_per_symbol[source_symbol][allocation_tick]

        target_symbol_original_shares = self._portfolio_allocation[target_symbol]
        target_symbol_price = self._prices_per_symbol[target_symbol][allocation_tick]

        # Deallocation
        deallocated_funds = source_symbol_shares * source_symbol_price
        self._portfolio_allocation[source_symbol] = 0.0

        # Penalty
        deallocated_funds = deallocated_funds * (1.0 - PortfolioStocksEnv.ALLOCATION_PENALTY)

        # Reallocation
        target_symbol_new_shares = deallocated_funds / target_symbol_price
        self._portfolio_allocation[target_symbol] = target_symbol_original_shares + target_symbol_new_shares

        # Notify Observer
        if self._env_observer is not None:
            self._env_observer.notify_stock_buy(target_symbol, target_symbol_new_shares, target_symbol_price)
            self._env_observer.notify_stock_sell(source_symbol, source_symbol_shares, source_symbol_price)

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

    def _portfolio_value(self, portfolio_allocation, current_tick):
        portfolio_value = 0.0

        for symbol in portfolio_allocation:
            symbol_shares = portfolio_allocation[symbol]
            symbol_price = self._prices_per_symbol[symbol][current_tick]

            portfolio_value += symbol_shares * symbol_price

        return portfolio_value

    def _get_info(self):
        return {
            'current_profit': self.current_profit(),
            'current_portfolio_value': self.current_portfolio_value(),
            'portfolio_allocation': self._portfolio_allocation,
            'allocations_history': self._allocations_history
        }

    def _get_observation(self, current_tick):
        result = None

        for symbol in self._signal_features_per_symbol:
            signals_for_symbol = self._signal_features_per_symbol[symbol]
            signals_for_symbol_for_tick = signals_for_symbol[(current_tick - self._window_size):current_tick]
            result = signals_for_symbol_for_tick if result is None else np.hstack([result, signals_for_symbol_for_tick])

        return np.array(result)

    def _get_allocated_symbol(self):
        # FIXME: This function will go away once multiple allocations are allowed
        for symbol in self._portfolio_allocation:
            if self._portfolio_allocation[symbol] > 0.0:
                return symbol

        raise ValueError("There are no allocated symbols")
