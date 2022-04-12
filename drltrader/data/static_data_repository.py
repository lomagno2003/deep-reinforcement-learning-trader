from drltrader.data import DataRepository, Scenario


class StaticDataRepository(DataRepository):
    def __init__(self, dataframe_per_symbol: dict):
        self._dataframe_per_symbol = dataframe_per_symbol

    def retrieve_datas(self, scenario: Scenario):
        return self._dataframe_per_symbol