from gym_anytrading.envs import StocksEnv, Positions, Actions


class SingleStockEnv(StocksEnv):
    def __init__(self,
                 df,
                 window_size,
                 frame_bound,
                 prices_feature_name,
                 signal_feature_names,
                 reset_enabled: bool = True):
        self._prices_feature_name = prices_feature_name
        self._signal_feature_names = signal_feature_names
        self._reset_enabled = reset_enabled

        super().__init__(df, window_size, frame_bound)

        self.trade_fee_bid_percent = 0.001  # unit
        self.trade_fee_ask_percent = 0.001  # unit

    def _process_data(self):
        # FIXME: Probably there's a better way to do this
        self.df.insert(0, "current_profit", 0.0, allow_duplicates=False)
        self.df.insert(0, "position_start_time", 0.0, allow_duplicates=False)

        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, self._prices_feature_name].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['current_profit', 'position_start_time'] + self._signal_feature_names].to_numpy()[start:end]
        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            original_profit = self._total_profit
            shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
            new_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

            if self._position == Positions.Long:
                step_reward += new_profit - original_profit

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:

            if self._position == Positions.Long:
                self._total_profit = self._calculate_current_profit()

        if self._position == Positions.Long:
            self.signal_features[self._current_tick][0] = 1.0 - self._calculate_current_profit()
            self.signal_features[self._current_tick][1] = self._current_tick - self._last_trade_tick

    def _calculate_current_profit(self):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
        return (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def disable_reset(self):
        self._reset_enabled = False

    def reset(self):
        if self._reset_enabled:
            return super().reset()
        else:
            return self._get_observation()
