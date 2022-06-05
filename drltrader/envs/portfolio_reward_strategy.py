class RewardStrategy:
    def get_reward(self, env):
        pass


class MixedRewardStrategy(RewardStrategy):
    def get_reward(self, env):
        total_profit = env.profit() - 1.0

        period = 50
        period_incentive = 0.0

        current_tick = env.current_tick
        previous_tick = max(0, current_tick - period)

        period_checkpoint = current_tick % period == 0

        if period_checkpoint:
            period_increased = env.profit(current_tick) > env.profit(previous_tick)
            period_incentive = 1.0 if period_increased else -1.0

        return total_profit + period_incentive / 4
