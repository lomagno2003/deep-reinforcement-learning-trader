import pandas as pd
from datetime import timedelta
from datetime import datetime

from drltrader.data import DataRepository, Scenario


class CompositeDataRepository(DataRepository):
    def __init__(self, data_repositories: list, combinator_operation: str = 'sum'):
        self._data_repositories = data_repositories
        self._combinator_operation = combinator_operation

    def get_repository_name(self):
        repository_name = f"Composite({self._combinator_operation}, "
        first = True

        for data_repository in self._data_repositories:
            if first:
                repository_name += f"{data_repository.get_repository_name()}"
                first = False
            else:
                repository_name += f", {data_repository.get_repository_name()}"

        repository_name += ")"

        return repository_name

    def retrieve_datas(self, scenario: Scenario):
        dr_results = []

        for data_repository in self._data_repositories:
            dr_results.append(data_repository.retrieve_datas(scenario))

        if self._combinator_operation == 'sum':
            return self._combine_using_sum(dr_results=dr_results, scenario=scenario)
        else:
            # FIXME: Assuming repeat
            return self._combine_using_repeat(dr_results=dr_results, scenario=scenario)

    def _combine_using_sum(self, dr_results: list, scenario: Scenario):
        results = {}
        for symbol in scenario.symbols:
            for i in range(len(dr_results)):
                if symbol not in results:
                    results[symbol] = dr_results[i][symbol]
                    results[symbol] = results[symbol].sort_index()
                    results[symbol] = results[symbol].tz_localize(None)

                    original_index = results[symbol].index

                    if results[symbol].empty:
                        interval = timedelta(hours=1)
                        origin = datetime.fromtimestamp(0)
                    else:
                        interval = results[symbol].index[1] - results[symbol].index[0]
                        origin = results[symbol].index[0]
                        origin = origin.tz_localize(None)

                    results[symbol] = results[symbol].resample(interval, origin=origin).sum()
                else:
                    # FIXME: Probably this shouldn't be done here
                    dr_results[i][symbol].index = pd.to_datetime(dr_results[i][symbol].index, utc=True)
                    dr_results[i][symbol] = dr_results[i][symbol].tz_localize(None)
                    results[symbol] = results[symbol].tz_localize(None)

                    dr_results[i][symbol] = dr_results[i][symbol].sort_index()
                    resampled_dataframe = dr_results[i][symbol].resample(interval, origin=origin).sum()

                    results[symbol] = pd.concat([results[symbol], resampled_dataframe], axis=1).fillna(0.0)

            # FIXME: This causes weird situations when sentiments are outside the market window
            # FIXME: This might cause data loss
            results[symbol] = results[symbol].reindex(original_index)

        return results

    def _combine_using_repeat(self, dr_results: list, scenario: Scenario):
        results = {}
        for symbol in scenario.symbols:
            dataframes_for_symbol = []

            for i in range(len(dr_results)):
                dataframes_for_symbol.append(dr_results[i][symbol])

            results[symbol] = pd.concat(dataframes_for_symbol, axis=1).fillna(method='ffill').fillna(0.0)

        return results
