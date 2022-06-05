class RewardStrategy:
    def get_reward(self, env):
        pass


class MixedRewardStrategy(RewardStrategy):
    def get_reward(self, env):
        total_profit = env.profit() - 1.0
        period_incentive = self._calculate_period_incentive(env)
        greedy_trader_penalty = self._calculate_greedy_trader_penalty(env)
        passive_trader_penalty = self._calculate_passive_trader_penalty(env)

        return total_profit # + period_incentive + greedy_trader_penalty + passive_trader_penalty

    def _calculate_greedy_trader_penalty(self, env):
        current_tick = env.current_tick
        last_tick, _ = env._find_portfolio_on_tick(env.current_tick)

        greedy_trader_penalty = 0.0
        if current_tick - last_tick < 50:
            greedy_trader_penalty = -1.0

        return greedy_trader_penalty

    def _calculate_passive_trader_penalty(self, env):
        current_tick = env.current_tick
        last_tick, _ = env._find_portfolio_on_tick(env.current_tick)

        passive_trader_penalty = 0.0
        if current_tick - last_tick > 200:
            passive_trader_penalty = -1.0
        return passive_trader_penalty

    def _calculate_period_incentive(self, env):
        period = 50
        period_incentive = 0.0
        current_tick = env.current_tick
        previous_tick = max(0, current_tick - period)
        period_checkpoint = current_tick % period == 0
        if period_checkpoint:
            period_increased = env.profit(current_tick) > env.profit(previous_tick)
            period_incentive = 1.0 if period_increased else -1.0

        return period_incentive / 4
