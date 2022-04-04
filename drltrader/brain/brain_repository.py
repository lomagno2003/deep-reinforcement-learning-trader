from drltrader.brain.brain import Brain


class BrainRepository:
    def load(self, brain_id: str):
        pass

    def save(self, brain_id: str, brain: Brain, override: bool = False):
        pass
