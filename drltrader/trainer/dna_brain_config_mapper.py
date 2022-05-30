import logging.config

from drltrader.brain.brain import BrainConfiguration
from drltrader.data import DataRepository

logging.config.fileConfig('log.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class DnaBrainConfigMapper:
    EXCLUDED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    # Notice that there's a co-relation between window-size and kernel-size. Additional considerations are done below.
    MAX_LINEAR_LAYER_SIZE = 2048
    MIN_LINEAR_LAYER_SIZE = 128
    MAX_CNN_KERNEL_COUNT = 128
    MIN_CNN_KERNEL_COUNT = 32
    MAX_CNN_KERNEL_SIZE = 10
    MIN_CNN_KERNEL_SIZE = 8

    MAX_WINDOW_SIZE = 32
    MIN_WINDOW_SIZE = 1

    # FIXME: We can't go over 1d since it triggers bug on portfolio_stocks_env#L175
    INTERVALS = ['5m', '15m', '30m', '1h']

    INDICATOR_GENE_ACTIVATION_THRESHOLD = 0.8
    SYMBOL_GENE_ACTIVATION_THRESHOLD = 0.5

    SYMBOLS = ['TDOC', 'ETSY', 'MELI', 'SE', 'SQ', 'DIS', 'TSLA', 'AAPL', 'MSFT', 'SHOP', 'FB']

    F_CNN1_KERNEL_COUNT_IDX = 0
    F_CNN1_KERNEL_SIZE_IDX = F_CNN1_KERNEL_COUNT_IDX + 1
    F_CNN2_KERNEL_COUNT_IDX = F_CNN1_KERNEL_SIZE_IDX + 1
    F_CNN2_KERNEL_SIZE_IDX = F_CNN2_KERNEL_COUNT_IDX + 1
    F_LINEAR1_SIZE_IDX = F_CNN2_KERNEL_SIZE_IDX + 1
    F_LINEAR2_SIZE_IDX = F_LINEAR1_SIZE_IDX + 1

    WINDOW_SIZE_GENE_IDX = F_LINEAR2_SIZE_IDX + 1
    USE_NORMALIZED_OBS_GENE_IDX = WINDOW_SIZE_GENE_IDX + 1
    INTERVAL_GENE_IDX = USE_NORMALIZED_OBS_GENE_IDX + 1
    DYNAMIC_CONFIGURATIONS_GENE_IDX = INTERVAL_GENE_IDX + 1

    def __init__(self, features_per_symbol: list):
        self._features_per_symbol = features_per_symbol

    def get_genes_size(self):
        return len(self._features_per_symbol) \
               + DnaBrainConfigMapper.DYNAMIC_CONFIGURATIONS_GENE_IDX \
               + len(DnaBrainConfigMapper.SYMBOLS)

    def get_brain_configuration_from_dna(self, dna):
        f_cnn1_kernel_count = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_CNN_KERNEL_COUNT,
                                                                    DnaBrainConfigMapper.MAX_CNN_KERNEL_COUNT,
                                                                    DnaBrainConfigMapper.F_CNN1_KERNEL_COUNT_IDX,
                                                                    dna)
        f_cnn1_kernel_size = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_CNN_KERNEL_SIZE,
                                                                   DnaBrainConfigMapper.MAX_CNN_KERNEL_SIZE,
                                                                   DnaBrainConfigMapper.F_CNN1_KERNEL_SIZE_IDX,
                                                                   dna)

        f_cnn2_kernel_count = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_CNN_KERNEL_COUNT,
                                                                    DnaBrainConfigMapper.MAX_CNN_KERNEL_COUNT,
                                                                    DnaBrainConfigMapper.F_CNN2_KERNEL_COUNT_IDX,
                                                                    dna)
        f_cnn2_kernel_size = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_CNN_KERNEL_SIZE,
                                                                   DnaBrainConfigMapper.MAX_CNN_KERNEL_SIZE,
                                                                   DnaBrainConfigMapper.F_CNN2_KERNEL_SIZE_IDX,
                                                                   dna)

        f_linear1_size = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_LINEAR_LAYER_SIZE,
                                                               DnaBrainConfigMapper.MAX_LINEAR_LAYER_SIZE,
                                                               DnaBrainConfigMapper.F_LINEAR1_SIZE_IDX,
                                                               dna)
        f_linear2_size = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_LINEAR_LAYER_SIZE,
                                                               DnaBrainConfigMapper.MAX_LINEAR_LAYER_SIZE,
                                                               DnaBrainConfigMapper.F_LINEAR2_SIZE_IDX,
                                                               dna)

        window_size = DnaBrainConfigMapper._calculate_value(DnaBrainConfigMapper.MIN_WINDOW_SIZE,
                                                            DnaBrainConfigMapper.MAX_WINDOW_SIZE,
                                                            DnaBrainConfigMapper.WINDOW_SIZE_GENE_IDX,
                                                            dna)

        if window_size < f_cnn1_kernel_size:
            return None

        if window_size < f_cnn2_kernel_size:
            return None

        use_normalized_observations = True if dna[DnaBrainConfigMapper.USE_NORMALIZED_OBS_GENE_IDX] > 0.5 else False

        interval_idx = int(len(DnaBrainConfigMapper.INTERVALS) * dna[DnaBrainConfigMapper.INTERVAL_GENE_IDX])
        interval = DnaBrainConfigMapper.INTERVALS[interval_idx]

        signal_feature_names = []

        for indicator_idx in range(0, len(self._features_per_symbol)):
            if dna[DnaBrainConfigMapper.DYNAMIC_CONFIGURATIONS_GENE_IDX + indicator_idx] > \
                    DnaBrainConfigMapper.INDICATOR_GENE_ACTIVATION_THRESHOLD:
                selected_feature = self._features_per_symbol[indicator_idx]

                if selected_feature not in DnaBrainConfigMapper.EXCLUDED_COLUMNS:
                    signal_feature_names.append(selected_feature)

        symbols_gene_start_idx = DnaBrainConfigMapper.DYNAMIC_CONFIGURATIONS_GENE_IDX \
                                 + len(self._features_per_symbol) - 1

        # FIXME: Hardcoded value
        symbols = ['SPY']

        for symbol_idx in range(0, len(DnaBrainConfigMapper.SYMBOLS)):
            if dna[symbols_gene_start_idx + symbol_idx] > \
                    DnaBrainConfigMapper.SYMBOL_GENE_ACTIVATION_THRESHOLD:
                symbols.append(DnaBrainConfigMapper.SYMBOLS[symbol_idx])

        return BrainConfiguration(f_cnn1_kernel_count=f_cnn1_kernel_count,
                                  f_cnn1_kernel_size=f_cnn1_kernel_size,
                                  f_cnn2_kernel_count=f_cnn2_kernel_count,
                                  f_cnn2_kernel_size=f_cnn2_kernel_size,
                                  f_linear1_size=f_linear1_size,
                                  f_linear2_size=f_linear2_size,
                                  window_size=window_size,
                                  signal_feature_names=signal_feature_names,
                                  use_normalized_observations=use_normalized_observations,
                                  interval=interval,
                                  symbols=symbols)
    
    @staticmethod
    def _calculate_value(min, max, gene_idx, dna):
        return int(min + (max - min) * dna[gene_idx])