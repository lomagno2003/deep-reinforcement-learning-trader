import gym
import math
import numpy as np
import pandas as pd
import logging
import logging.config
from gym import spaces
import matplotlib.pyplot as plt

from drltrader.observers import Observer
from drltrader.envs.portfolio_reward_strategy import RewardStrategy, MixedRewardStrategy

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class PortfolioStocksEnv(gym.Env):
    ALLOCATION_PENALTY_LONG = 0.005
    ALLOCATION_PENALTY_SHORT = 0.01

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 window_size: int,
                 dataframe_per_symbol: dict,
                 initial_portfolio: dict,
                 prices_feature_name: str = 'Close',
                 signal_feature_names: list = ['RSI_4', 'RSI_16'],
                 reward_strategy: RewardStrategy = MixedRewardStrategy(),
                 rendering_enabled: bool = False):
        super(PortfolioStocksEnv, self).__init__()

        # Save Configurations
        self._window_size = window_size
        self._initial_portfolio = initial_portfolio
        self._dataframe_per_symbol = dataframe_per_symbol
        self._prices_feature_name = prices_feature_name
        self._signal_feature_names = signal_feature_names
        self._reward_strategy = reward_strategy
        self._rendering_enabled = rendering_enabled

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

        # Initialize Runtime Variables
        self._observer = None
        self._done = None
        self.current_tick = None

        self._rewards_history = None
        self._portfolio_history = None
        self._portfolio_profit_history = None
        self._portfolio_position_age_history = None

        self.reset()

        # Initialize Gym Configurations
        self.action_space = spaces.Discrete(len(self._action_to_symbol))
        self.shape = self._get_observation(self._window_size).shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def reset(self):
        self._done = False
        if not self._reset_enabled:
            return

        self.current_tick = self._frame_bound[0]
        self._portfolio_history = {self.current_tick - 1: self._initial_portfolio}
        self._portfolio_profit_history = np.zeros(self._window_size)
        self._portfolio_position_age_history = np.zeros(self._window_size)
        self._rewards_history = np.zeros(self._window_size) # FIXME: Refactor this

        return self._get_observation(self.current_tick)

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

        self._done = self.current_tick >= self._frame_bound[1]

        if not self._done:
            logger.info("New data was added and the brain can check it out")

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

            # FIXME: I'm sure there's a better way to do this
            prices_per_symbol[symbol] = np.array(list(map(lambda x: x if x > 0.0 else np.nan, prices_per_symbol[symbol])))
            mask = np.isnan(prices_per_symbol[symbol])
            prices_per_symbol[symbol][mask] = np.interp(np.flatnonzero(mask),
                                                        np.flatnonzero(~mask),
                                                        prices_per_symbol[symbol][~mask])

        # FIXME: If prices_per_symbol[ANY] <  self._window_size, the model will fail since
        #  observation < observation_space
        self._timestamps = timestamps
        self._prices_per_symbol = prices_per_symbol
        self._signal_features_per_symbol = signal_features_per_symbol

    def observe(self, observer: Observer):
        self._observer = observer
        if observer is not None:
            self._observer.notify_begin_of_observation(self._find_portfolio_on_tick(query_tick=0))

    def disable_reset(self):
        self._reset_enabled = False

    def close(self):
        # TODO: Stub-method
        pass

    def profit(self, query_tick: int = None):
        return self.portfolio_value(query_tick=query_tick) / self.portfolio_value(query_tick=0)

    def portfolio_value(self, query_tick: int = None):
        portfolio_value = 0.0

        if query_tick is None:
            query_tick = self.current_tick

        portfolio_last_update_tick, portfolio_on_tick = self._find_portfolio_on_tick(query_tick=query_tick)

        for symbol in portfolio_on_tick:
            symbol_shares = portfolio_on_tick[symbol]
            if portfolio_on_tick[symbol] is not None:
                symbol_original_price = self._prices_per_symbol[symbol][portfolio_last_update_tick]
            else:
                symbol_original_price = None

            symbol_current_price = self._prices_per_symbol[symbol][query_tick]

            if symbol_shares >= 0:
                portfolio_value += symbol_shares * symbol_current_price
            else:
                portfolio_value += -1 * (symbol_original_price * 2 - symbol_current_price) * symbol_shares

        return portfolio_value

    def _find_portfolio_on_tick(self, query_tick):
        last_portfolio_tick_before_query_tick = None

        for updates_tick in self._portfolio_history:
            if last_portfolio_tick_before_query_tick is None:
                last_portfolio_tick_before_query_tick = updates_tick
            else:
                if query_tick > updates_tick > last_portfolio_tick_before_query_tick:
                    last_portfolio_tick_before_query_tick = updates_tick

        last_portfolio_before_query_tick = self._portfolio_history[last_portfolio_tick_before_query_tick]

        return last_portfolio_tick_before_query_tick, last_portfolio_before_query_tick

    def step(self, action):
        if self._done:
            return None, self._calculate_reward(), self._done, self._get_info()

        self.current_tick += 1

        # Precalculate runtime signals
        self._calculate_runtime_signals()

        # Process Action
        selected_symbol = self._action_to_symbol[action]
        selected_side = self._action_to_side[action]
        _, current_portfolio = self._find_portfolio_on_tick(self.current_tick)
        allocated_symbol, allocated_side = self._get_allocated_position(current_portfolio)

        if selected_symbol != allocated_symbol or selected_side != allocated_side:
            self._transfer_allocations(allocated_symbol,
                                       allocated_side,
                                       selected_symbol,
                                       selected_side)

        return self.get_step_outputs()

    def _calculate_runtime_signals(self):
        portfolio_tick, portfolio = self._find_portfolio_on_tick(query_tick=self.current_tick)
        symbol, side = self._get_allocated_position(portfolio)

        current_price = self._prices_per_symbol[symbol][self.current_tick]
        last_price = self._prices_per_symbol[symbol][portfolio_tick]

        profit = current_price - last_price if side == 'long' else last_price - current_price
        profit = (profit * 10) / last_price

        position_age_ticks = self.current_tick - portfolio_tick
        position_age = position_age_ticks/1000

        # FIXME: What about shorting?
        self._portfolio_profit_history = np.append(self._portfolio_profit_history, profit)
        self._portfolio_position_age_history = np.append(self._portfolio_position_age_history, position_age)

    def _transfer_allocations(self, source_symbol, source_side, target_symbol, target_side):
        # Temporary Variables
        old_portfolio_tick, current_portfolio = self._find_portfolio_on_tick(self.current_tick)
        old_portfolio = current_portfolio.copy()
        new_portfolio = old_portfolio.copy()

        source_symbol_original_price = self._prices_per_symbol[source_symbol][old_portfolio_tick]
        source_symbol_shares = old_portfolio[source_symbol]
        source_symbol_price = self._prices_per_symbol[source_symbol][self.current_tick]

        # Deallocation
        if source_side == 'long':
            deallocated_funds = source_symbol_shares * source_symbol_price
            new_portfolio[source_symbol] = 0.0
        else:
            source_symbol_original_market_value = -2 * source_symbol_shares * source_symbol_original_price
            source_symbol_market_value = -1 * source_symbol_shares * source_symbol_price
            deallocated_funds = source_symbol_original_market_value - source_symbol_market_value
            new_portfolio[source_symbol] = 0.0

        # Penalty
        penalty = PortfolioStocksEnv.ALLOCATION_PENALTY_LONG if target_side == 'long' \
            else PortfolioStocksEnv.ALLOCATION_PENALTY_SHORT
        deallocated_funds = deallocated_funds * (1.0 - penalty)

        # Reallocation
        if target_symbol not in new_portfolio:
            new_portfolio[target_symbol] = 0.0

        target_symbol_original_shares = new_portfolio[target_symbol]
        target_symbol_price = self._prices_per_symbol[target_symbol][self.current_tick]

        target_symbol_sign = 1.0 if target_side == 'long' else -1.0
        target_symbol_new_shares = target_symbol_sign * deallocated_funds / target_symbol_price
        new_portfolio[target_symbol] = target_symbol_original_shares + target_symbol_new_shares

        # Notify Observer
        if self._observer is not None:
            self._observer.notify_portfolio_change(old_portfolio=old_portfolio,
                                                   new_portfolio=new_portfolio)

        # Statistics
        source_position_profit = None
        if source_side == 'long':
            percentual_price_difference = (source_symbol_price - source_symbol_original_price) / source_symbol_original_price
            source_position_profit = percentual_price_difference - PortfolioStocksEnv.ALLOCATION_PENALTY_LONG
        else:
            percentual_price_difference = (source_symbol_original_price - source_symbol_price) / source_symbol_original_price
            source_position_profit = percentual_price_difference - PortfolioStocksEnv.ALLOCATION_PENALTY_SHORT

        allocation_details = {
            'allocation_tick': self.current_tick,

            'source_symbol': source_symbol,
            'source_symbol_price': source_symbol_price,
            'source_symbol_shares': source_symbol_shares,
            'source_position_profit': source_position_profit,

            'target_symbol': target_symbol,
            'target_symbol_price': target_symbol_price,
            'target_symbol_original_shares': target_symbol_original_shares,
            'target_symbol_new_shares': target_symbol_new_shares,
        }
        self._portfolio_history[self.current_tick] = new_portfolio

        return allocation_details

    def get_step_outputs(self):
        self._done = self.current_tick == self._frame_bound[1]

        if self._done and self._rendering_enabled:
            self.render_all()

        # FIXME: Remove this, is just for testing
        if self._done:
            # Log stats
            portfolio_history_dataframe = pd.DataFrame.from_dict(self._portfolio_history, orient='index')
            logger.info(f"The profit is {self.profit()}. Here are the portfolio updates:")
            logger.info(portfolio_history_dataframe.to_string())

        observation = self._get_observation(self.current_tick)
        reward = self._calculate_reward()
        done = self._done
        info = self._get_info()

        return observation, reward, done, info

    def _get_observation(self, query_tick):
        result = None

        # Add signals to result
        for symbol in self._signal_features_per_symbol:
            signals_for_symbol = self._signal_features_per_symbol[symbol]
            signals_for_symbol_for_tick = signals_for_symbol[(query_tick - self._window_size):query_tick]
            result = signals_for_symbol_for_tick if result is None else np.hstack([result, signals_for_symbol_for_tick])

        # Add runtime signals to result
        portfolio_profit_window = self._portfolio_profit_history[(query_tick - self._window_size):query_tick]
        portfolio_position_age_window = self._portfolio_position_age_history[(query_tick - self._window_size):query_tick]

        portfolio_profit_window_mean = portfolio_profit_window.mean()
        portfolio_profit_window_oscilator = list(map(lambda x: x - portfolio_profit_window_mean, portfolio_profit_window))

        result = np.hstack([result, np.array([portfolio_position_age_window]).transpose()])
        result = np.hstack([result, np.array([portfolio_profit_window]).transpose()])
        result = np.hstack([result, np.array([portfolio_profit_window_oscilator]).transpose()])

        # Transpose to match model and return
        return np.array(result).transpose()

    def _calculate_reward(self):
        if len(self._rewards_history) <= self.current_tick:
            self._rewards_history = np.append(self._rewards_history, self._reward_strategy.get_reward(self))

        return self._rewards_history[-1]

    def _get_info(self):
        _, portfolio = self._find_portfolio_on_tick(query_tick=self.current_tick)
        return {
            'current_profit': self.profit(),
            'current_portfolio_value': self.portfolio_value(),
            'portfolio_allocation': portfolio,
        }

    def _get_allocated_position(self, portfolio):
        # FIXME: This function will go away once multiple allocations are allowed
        symbol = None
        side = None

        for symbol in portfolio:
            if portfolio[symbol] > 0.0:
                symbol = symbol
                side = 'long'
                break
            elif portfolio[symbol] < 0.0:
                symbol = symbol
                side = 'short'
                break

        return symbol, side

        raise ValueError("There are no allocated symbols")

    def render(self, mode='human'):
        # TODO: Stub-method
        pass

    def render_all(self,
                   render_signals: bool = True,
                   render_rewards: bool = True,
                   render_runtime_signals: bool = True,
                   render_exclusion_feature_names: list = ['5m_Close']):
        prices_per_symbol = {}
        plt.figure(figsize=(15, 6))
        plt.cla()

        for symbol in self._prices_per_symbol:
            prices_array = self._prices_per_symbol[symbol]
            plt.plot(prices_array[prices_array != 0], label=symbol)

            prices_per_symbol[symbol] = prices_array[prices_array != 0]

        for symbol in prices_per_symbol:
            long_symbol_index = []
            short_symbol_index = []
            long_symbol_prices = []
            short_symbol_prices = []

            long_symbol_start_index = []
            short_symbol_start_index = []
            long_symbol_start_prices = []
            short_symbol_start_prices = []

            last_allocated_symbol = None
            last_allocated_side = None
            for i in range(len(prices_per_symbol[symbol])):
                if i in self._portfolio_history:
                    new_portfolio = self._portfolio_history[i]
                    symbol, side = self._get_allocated_position(new_portfolio)
                    last_allocated_symbol = symbol
                    last_allocated_side = side

                    if last_allocated_side == 'long':
                        long_symbol_start_index.append(i)
                        long_symbol_start_prices.append(prices_per_symbol[symbol][i])
                    else:
                        short_symbol_start_index.append(i)
                        short_symbol_start_prices.append(prices_per_symbol[symbol][i])

                if last_allocated_symbol is not None:
                    if last_allocated_side == 'long':
                        long_symbol_index.append(i)
                        long_symbol_prices.append(prices_per_symbol[symbol][i])
                    else:
                        short_symbol_index.append(i)
                        short_symbol_prices.append(prices_per_symbol[symbol][i])

            plt.plot(long_symbol_index, long_symbol_prices, 'b')
            plt.plot(short_symbol_index, short_symbol_prices, 'r')
            plt.plot(long_symbol_start_index, long_symbol_start_prices, 'b^')
            plt.plot(short_symbol_start_index, short_symbol_start_prices, 'rv')

        plt.twinx()
        if render_signals:
            for symbol in self._signal_features_per_symbol:
                transposed_signal_features = self._signal_features_per_symbol[symbol].transpose()
                for signal_feature_name in self._signal_feature_names:
                    if signal_feature_name in render_exclusion_feature_names:
                        continue

                    signal_feature_name_idx = list(self._signal_feature_names).index(signal_feature_name)
                    signals_array = transposed_signal_features[signal_feature_name_idx]
                    plt.plot(signals_array, alpha=0.7, label=signal_feature_name)

        if render_rewards:
            plt.plot(self._rewards_history, alpha=0.7, label='Reward')

        if render_runtime_signals:
            plt.plot(self._portfolio_profit_history, alpha=0.7, label='Profit')
            plt.plot(self._portfolio_position_age_history, alpha=0.7, label='Position Age')
            # plt.plot(self._portfolio_profit_history, alpha=0.7, label='Reward')

        plt.suptitle(
            "Total Profit: %.6f" % self.profit()
        )
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

        # Log stats
        portfolio_history_dataframe = pd.DataFrame.from_dict(self._portfolio_history, orient='index')
        logger.info(f"The profit is {self.profit()}. Here are the portfolio updates:")
        logger.info(portfolio_history_dataframe.to_string())
