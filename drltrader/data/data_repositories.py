from drltrader.data import DataRepository
from drltrader.data.composite_data_repository import CompositeDataRepository
from drltrader.data.indicators_data_repository import IndicatorsDataRepository
from drltrader.data.ohlcv_data_repository import AlpacaOHLCVDataRepository
from drltrader.data.cached_data_repository import CachedDataRepository
from drltrader.data.prefix_data_repository import PrefixDataRepository
from drltrader.data.resampled_data_repository import ResampleDataRepository
from drltrader.data.normalized_data_repository import NormalizedDataRepository


class DataRepositories:
    @staticmethod
    def build_multi_time_interval_data_repository():
        data_repository_5m = IndicatorsDataRepository(
            ResampleDataRepository(
                CachedDataRepository(AlpacaOHLCVDataRepository()),
                '5m'
            )
        )
        data_repository_60m = IndicatorsDataRepository(
            ResampleDataRepository(
                CachedDataRepository(AlpacaOHLCVDataRepository()),
                '60m'
            )
        )
        data_repository_1d = IndicatorsDataRepository(
            ResampleDataRepository(
                CachedDataRepository(AlpacaOHLCVDataRepository()),
                '1d'
            )
        )

        data_repository: DataRepository = CompositeDataRepository(
            combinator_operation='repeat',
            data_repositories=[
                PrefixDataRepository(data_repository_5m, '5m'),
                PrefixDataRepository(data_repository_60m, '60m'),
                PrefixDataRepository(data_repository_1d, '1d')
            ])

        return CachedDataRepository(data_repository)

    @staticmethod
    def build_normalized_multi_time_interval_data_repository(exclude_normalized_columns: list = []):
        return NormalizedDataRepository(source_data_repository=DataRepositories.build_multi_time_interval_data_repository(),
                                        excluded_columns=exclude_normalized_columns)
