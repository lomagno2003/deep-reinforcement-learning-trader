import pandas as pd

from drltrader.data import DataRepository, Scenario


class CompositeDataRepository(DataRepository):
    def __init__(self, data_repositories: list):
        self._data_repositories = data_repositories

    def retrieve_datas(self, scenario: Scenario):
        dr_results = []

        for data_repository in self._data_repositories:
            dr_results.append(data_repository.retrieve_datas(scenario))

        # FIXME: This causes loss of data when the interval of the main DF captures multiple sentiments
        # FIXME: This causes weird situations when sentiments are outside the market window
        tolerance = pd.Timedelta('1d')

        results = {}
        for symbol in scenario.symbols:
            for i in range(len(dr_results)):
                if symbol not in results:
                    results[symbol] = dr_results[i][symbol]
                    results[symbol] = results[symbol].sort_index()
                else:
                    # FIXME: Probably this shouldn't be done here
                    dr_results[i][symbol].index = pd.to_datetime(dr_results[i][symbol].index, utc=True)
                    dr_results[i][symbol].tz_localize(None)
                    results[symbol].tz_localize(None)

                    dr_results[i][symbol] = dr_results[i][symbol].sort_index()
                    results[symbol] = pd.merge_asof(results[symbol],
                                                    dr_results[i][symbol],
                                                    direction='forward',
                                                    left_index=True,
                                                    right_index=True,
                                                    tolerance=tolerance,
                                                    suffixes=('', ''))

        return results
