
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict

from .kernel import egi_kernel, impact_engine_kernel, integrals_kernel
from .kernel import EgiDataChannel, EgiIntegralChannel, PulsaDataChannel
from .pulsa_data import PulsaData

from .integrals import MultiChannelWrapper, ExtremumCircularWrapper, SuperExtremumCircularWrapper

_is_test = False
is_check_extrema_num = False


@dataclass(frozen=True)
class InconsistencySnapshot:
    """Latest observed-versus-inferred price displacement for every channel."""

    real_move: np.ndarray
    expected_move: np.ndarray
    residual: np.ndarray
    normalized_score: np.ndarray
    excess_intensity: np.ndarray

    @property
    def peak_score(self) -> float:
        """Return the largest absolute normalized inconsistency."""
        if self.normalized_score.size == 0:
            return 0.0
        return float(np.max(np.abs(self.normalized_score)))

def check_extrema_num(rst):
    if not is_check_extrema_num:
        return
    if not hasattr(check_extrema_num, 'seen_peaks'):
        check_extrema_num.seen_peaks = set()

    # Just check channel 0 for simplicity
    if rst.shape[1] > 0:
        abs_time = rst[:, 0, int(EgiDataChannel.abs_time.value)]
        extremas = rst[:, 0, int(EgiDataChannel.extremas.value)]

        valid_idx = np.where(np.abs(extremas) > 0.5)[0]
        peaks_abs_time = abs_time[valid_idx]

        # Add to set
        for t in peaks_abs_time:
            if t > 0:  # Ignore 0 initialization times
                check_extrema_num.seen_peaks.add(t)

        print(f"""Status: {len(check_extrema_num.seen_peaks)}""")

class impact:
    """Fit and evaluate the nonlinear volume-to-price impact relationship."""
    @staticmethod
    def inverse(
        time_square: np.ndarray,
        price_moves: np.ndarray,
        channel_indices: np.ndarray,
        beta_range: tuple = (0.35, 0.99)
    ) -> np.ndarray:
        """Estimate power-law parameters independently for each force channel."""
        return impact_engine_kernel.inverse(time_square, price_moves, channel_indices, beta_range)

    @staticmethod
    def inverse2(
        time_square: np.ndarray,
        price_moves: np.ndarray,
        channel_indices: np.ndarray,
        beta_range: tuple = (0.35, 0.8),
        consider_offset: bool = True,
        steps: int = 10
    ) -> np.ndarray:
        """Estimate power-law parameters with an optional activation threshold."""
        return impact_engine_kernel.inverse2(time_square, price_moves, channel_indices, beta_range, consider_offset, steps)


    @staticmethod
    def forward(
    active_net_flow: np.ndarray,
    params: np.ndarray
    ) -> np.ndarray:
        """Evaluate inferred price movement from signed force and parameters."""
        return impact_engine_kernel.forward(active_net_flow, params)


@dataclass
class Egi:
    """Track excess generalized intensity in circular streaming buffers."""
    capacity: int
    max_window: int = 600
    min_window: int = 4
    peak_factor: int = 10
    peak_factor2: float = 2.0
    minimal_tick: float = 0.01

    increment: int = 0
    data: MultiChannelWrapper | None = None
    integrals: MultiChannelWrapper | None = None
    raw_extremas: ExtremumCircularWrapper | None = None
    extremas: SuperExtremumCircularWrapper | None = None
    extremas2: SuperExtremumCircularWrapper | None = None
    _baseline: float | None = field(default=None, init=False)
    _extrema_total_count: int = field(default=0, init=False)

    def __post_init__(self):

        container_data = np.zeros((self.capacity, len(EgiDataChannel)), dtype=np.float64)
        self.data = MultiChannelWrapper(container_data)

        container_intg = np.zeros((self.capacity + 1, len(EgiIntegralChannel)), dtype=np.float64)
        self.integrals = MultiChannelWrapper(container_intg)

        raw_idx = EgiDataChannel.raw_extremas
        extrema_idx = EgiDataChannel.extremas
        full_extrema_idx = EgiDataChannel.full_extremas

        extrema2_idx = EgiDataChannel.extremas2
        full_extrema2_idx = EgiDataChannel.full_extremas2
        abs_time_idx = EgiDataChannel.abs_time

        self.raw_extremas = ExtremumCircularWrapper(container_data, raw_extrema_id=raw_idx, abs_time_id=abs_time_idx)

        w1 = int(max(3, round(float(self.peak_factor) * float(self.max_window))))
        self.extremas = SuperExtremumCircularWrapper(
            self.raw_extremas,
            extremas_id=extrema_idx,
            full_extrema_id=full_extrema_idx,
            window=w1,
        )

        w2 = int(max(3, round(float(self.peak_factor2) * float(self.max_window))))
        self.extremas2 = SuperExtremumCircularWrapper(
            self.raw_extremas,
            extremas_id=extrema2_idx,
            full_extrema_id=full_extrema2_idx,
            window=w2,
        )

    def load_data(self, data: np.ndarray):
        """Append observed and expected price movements to the EGI buffers."""
        if data is None or data.size == 0:
            self.increment = 0
            return

        if not np.isfinite(data).all():
            self.increment = 0
            return

        self.increment = data.shape[0]
        self._refresh_data(data)

    def _refresh_data(self, data: np.ndarray):
        gap = data[:, EgiDataChannel.real_delta] - data[:, EgiDataChannel.expect_delta]

        start_head = int(self.data.heads[int(EgiDataChannel.real_value)])


        self.data.update(data, [EgiDataChannel.real_value, EgiDataChannel.real_delta, EgiDataChannel.expect_delta], calc_type='original')
        self.data.update(gap, [EgiDataChannel.real_expect_delta_gap], calc_type='original')



        self.integrals.update(data[:, EgiDataChannel.real_delta], [EgiIntegralChannel.real_normal], calc_type='normal')
        self.integrals.update(data[:, EgiDataChannel.real_delta], [EgiIntegralChannel.real_abs], calc_type='abs')


        self.integrals.update(data[:, EgiDataChannel.expect_delta], [EgiIntegralChannel.expect_normal], calc_type='normal')
        self.integrals.update(data[:, EgiDataChannel.expect_delta], [EgiIntegralChannel.expect_abs], calc_type='abs')


        self.integrals.update(gap, [EgiIntegralChannel.real_expect_gap_normal], calc_type='normal')
        self.integrals.update(gap, [EgiIntegralChannel.real_expect_gap_abs], calc_type='abs')



        current_head = int(self._extrema_total_count)

        self.raw_extremas.head = start_head
        self.raw_extremas.total_count = current_head
        self.raw_extremas.update(data[:, EgiDataChannel.real_value])

        self.extremas.update()
        self.extremas2.update()

        if self._baseline is None:
            self._baseline = integrals_kernel.get_mean_by_integral_logic(
                self.integrals[:, EgiIntegralChannel.real_expect_scale_gap_abs],
                0, self.increment,
                current_head
            )



        if _is_test:
            print(f"--- [DEBUG] EGI _refresh_data: current_head={current_head}, increment={self.increment}")

        egi_kernel.egi_update(
            self.data.get(),
            self.integrals.get(),
            current_head,
            self.increment,
            self._baseline,
            self.min_window,
            self.max_window,
            self.minimal_tick,
        )



        self.raw_extremas.advance_pointer()
        self._extrema_total_count = int(self.raw_extremas.total_count)



@dataclass
class PulsaEngine:
    """Infer price impact and compute instantaneous volume-price mismatch."""
    capacity: int
    tick_size: float = 0.01
    max_iceberg_ratio: float = field(default=2.0)
    is_estimated_paras: bool = False

    min_window: int = 4
    max_window: int = 600
    peak_factor: int = 10
    peak_factor2: float = 2.0

    _inv_paras: np.ndarray | None = None # (K, 3)

    _last_head: int = 0
    _current_head: int = 0
    _current_len: int = 0

    @property
    def bin_size(self):
        return self.tick_size

    def __post_init__(self):
        self.pulsa_data = PulsaData(self.tick_size, self.max_iceberg_ratio)

        self.egis = defaultdict(lambda: Egi(
            capacity=self.capacity,
            max_window=self.max_window,
            min_window=self.min_window,
            peak_factor=self.peak_factor,
            peak_factor2=self.peak_factor2,
            minimal_tick=self.tick_size,
        ))

    def _init(self) -> bool:
        if self._inv_paras is not None:
            return True

        data = self.pulsa_data.newest_data
        if data.size == 0:
            return False

        energy_idx = self.pulsa_data.energy_idxs
        inv = impact.inverse(data, data[:, PulsaDataChannel.trade_price_idx], energy_idx)
        if inv is None or not np.isfinite(inv).all():
            return False

        if self.is_estimated_paras:
            inv[:] = inv[0]

        self._inv_paras = inv
        if _is_test:
            print(f'Estimated paras are: \n{self._inv_paras} and is_estimated_paras is : {self.is_estimated_paras}')

        return True

    def _to_egis(self) -> bool:
        new_data = self.pulsa_data.newest_data
        if new_data.size == 0:
            return False

        if self._inv_paras is None:
            return False

        estimates = impact.forward(self.pulsa_data.newest_energy, self._inv_paras)
        if estimates is None or not np.isfinite(estimates).all():
            return False

        if not np.isfinite(new_data[:, [PulsaDataChannel.trade_price_idx, PulsaDataChannel.trade_price_diff]]).all():
            return False

        for idx in range(estimates.shape[1]):
            data = np.column_stack((
                new_data[:, PulsaDataChannel.trade_price_idx],
                new_data[:, PulsaDataChannel.trade_price_diff],
                estimates[:, idx]
            ))
            self.egis[idx].load_data(data)

        return True

    def newest_energy(self) -> np.ndarray:
        return self.pulsa_data.newest_energy

    def latest_inconsistency(self) -> InconsistencySnapshot | None:
        """Return the latest instantaneous volume-price mismatch.

        The engine compares the observed price move with the move inferred from
        order-book and trade-flow features. The signed normalized score is
        ``(real - expected) / (abs(real) + abs(expected) + epsilon)``.
        """
        if not self.egis:
            return None
        results = self.to_egi_results(total_fetch_len=1)
        if results.size == 0:
            return None

        latest = results[-1]
        real_move = latest[:, EgiDataChannel.real_delta.value].copy()
        expected_move = latest[:, EgiDataChannel.expect_delta_scale.value].copy()
        residual = latest[
            :, EgiDataChannel.real_expect_delta_scale_gap.value
        ].copy()
        denominator = np.abs(real_move) + np.abs(expected_move) + np.finfo(np.float64).eps
        normalized_score = residual / denominator
        excess_intensity = latest[:, EgiDataChannel.egi_value.value].copy()
        return InconsistencySnapshot(
            real_move=real_move,
            expected_move=expected_move,
            residual=residual,
            normalized_score=normalized_score,
            excess_intensity=excess_intensity,
        )

    def load_data(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0

        df2 = df

        prev_state = self.pulsa_data._snapshot_state()
        prev_inv_paras = None if self._inv_paras is None else self._inv_paras.copy()
        prev_head = int(self._current_head)
        prev_last_head = int(self._last_head)
        prev_current_len = int(self._current_len)

        try:
            self._last_head = self._current_head
            self.pulsa_data.load_data(df2)
            self._current_len = int(self.pulsa_data.current_len)
            if self._current_len <= 0:
                self.pulsa_data._restore_state(prev_state)
                self._inv_paras = prev_inv_paras
                self._current_head = prev_head
                self._last_head = prev_last_head
                self._current_len = prev_current_len
                return 0

            if not self._init():
                self.pulsa_data._restore_state(prev_state)
                self._inv_paras = prev_inv_paras
                self._current_head = prev_head
                self._last_head = prev_last_head
                self._current_len = prev_current_len
                return 0

            if not self._to_egis():
                self.pulsa_data._restore_state(prev_state)
                self._inv_paras = prev_inv_paras
                self._current_head = prev_head
                self._last_head = prev_last_head
                self._current_len = prev_current_len
                return 0

            self._current_head += int(self._current_len)
            return int(self._current_len)

        except Exception:
            self.pulsa_data._restore_state(prev_state)
            self._inv_paras = prev_inv_paras
            self._current_head = prev_head
            self._last_head = prev_last_head
            self._current_len = prev_current_len
            raise

    def to_egi_results(self, total_fetch_len: int = None) -> np.ndarray:
        """Return recent EGI frames in chronological order for every channel."""
        egi = next(iter(self.egis.values()))
        data = egi.data.get()

        head = egi.data.heads[0]
        capacity = egi.data.capacity

        if total_fetch_len is None:
            total_fetch_len = capacity

        total_fetch_len = min(total_fetch_len, capacity)

        rst = np.zeros((total_fetch_len, len(self.egis), len(EgiDataChannel)), dtype=data.dtype)

        start = integrals_kernel.n_get_next_head(head, -total_fetch_len, capacity)
        end = integrals_kernel.n_get_next_head(head, 0, capacity)

        for idx, egi_inst in self.egis.items():
            data = egi_inst.data.get()
            if start >= end and total_fetch_len > 0:
                part1_len = capacity - start
                rst[0: part1_len, idx, :] = data[start: capacity, :]
                rst[part1_len:, idx, :] = data[0:end, :]
            else:
                rst[:, idx, :] = data[start: end, :]

        check_extrema_num(rst)
        return rst
