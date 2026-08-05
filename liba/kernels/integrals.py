from .kernel import integrals_kernel

import numpy as np
from dataclasses import dataclass, field




float64_LIMIT = 1.0e30

_is_test = False
is_check_extrema_num = False


@dataclass
class MultiChannelWrapper:
    """Maintain multiple circular data or integral channels in one array."""
    out_integrals: np.ndarray

    def __post_init__(self):

        self.capacity = self.out_integrals.shape[0]
        self.n_channels = self.out_integrals.shape[1]

        self.heads = np.zeros(self.n_channels, dtype=np.int64)
        self.last_heads = np.zeros(self.n_channels, dtype=np.int64)
        self.currents = np.zeros(self.n_channels, dtype=np.float64)

    def __getitem__(self, key):
        return self.out_integrals[key]

    def get(self):
        return self.out_integrals

    def update(self, data: np.ndarray, channels: list | np.ndarray, calc_type: str = 'normal'):
        channels = np.asarray(channels, dtype=np.int64)


        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._normalize_channels(channels)

        if calc_type == 'normal' or calc_type == 'abs':
            is_abs = (calc_type == 'abs')

            self.currents = integrals_kernel.update_integral_multi(
                self.out_integrals,
                self.currents,
                data,
                self.heads,
                channels,
                is_abs
            )
        elif calc_type == 'original':
            self.currents = integrals_kernel.fill_original_multi(self.out_integrals, self.currents, data, self.heads, channels)
        else:
            raise ValueError(f'Current calc type does not support: {calc_type}')

        for i in channels:
            self.last_heads[i] = self.heads[i]
            self.heads[i] = integrals_kernel.n_get_next_head(self.heads[i], data.shape[0], self.capacity)

    def _normalize_channels(self, channels: np.ndarray):
        for i in channels:

            if abs(self.currents[i]) > float64_LIMIT:


                integrals_kernel.shift_integral_channel(self.out_integrals, i, self.currents[i])


                self.currents[i] = 0.0



@dataclass
class ExtremumCircularWrapper:
    """Detect streaming local extrema without advancing shared state early."""
    out_cube: np.ndarray

    raw_extrema_id: int
    abs_time_id: int = -1

    head: int = field(default=0, init=False)
    last_head: int = field(default=0, init=False)
    total_count: int = field(default=0, init=False)

    current_batch_size: int = field(default=0, init=False)

    def __post_init__(self):

        capacity = self.out_cube.shape[0]
        if self.out_cube.ndim != 2:
            raise ValueError("out_cube must be a two-dimensional array")
        if capacity < 3:
            raise ValueError(
                f"out_cube capacity must be at least 3; received {capacity}"
            )

        self.values = np.zeros(capacity, dtype=np.float64)
        self._semi_status = np.zeros(capacity, dtype=np.int64)

        self.raw_extremas_view = self.out_cube[:, self.raw_extrema_id]

        if self.abs_time_id >= 0:
            self.abs_time_view = self.out_cube[:, self.abs_time_id]
        else:
            self.abs_time_view = None

        self.head = 0
        self.last_head = 0
        self.total_count = 0
        self.current_batch_size = 0

    def update(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float64).flatten()
        total_batch_size = len(data)
        self.current_batch_size = total_batch_size
        if total_batch_size == 0:
            return


        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("input data contains NaN or infinite values")

        capacity = self.out_cube.shape[0]


        if total_batch_size >= capacity:
            raise ValueError(
                f"batch length {total_batch_size} exceeds safe capacity "
                f"{capacity - 1}"
            )


        integrals_kernel.compute_extremums_inplace_circular(
            data,
            self.values,
            self.raw_extremas_view,
            self._semi_status,
            self.head,
            self.total_count
        )


        if self.abs_time_view is not None:
            integrals_kernel.write_abs_time_circular(self.abs_time_view, self.head, total_batch_size, self.total_count)

    def advance_pointer(self):
        if self.current_batch_size == 0:
            return

        capacity = self.out_cube.shape[0]
        self.last_head = self.head
        self.head = integrals_kernel.n_get_next_head(self.head, self.current_batch_size, capacity)
        self.total_count += self.current_batch_size


        self.current_batch_size = 0

    def get(self):
        return self.raw_extremas_view




@dataclass
class SuperExtremumCircularWrapper:
    """Track windowed super extrema over the raw-extrema circular buffer."""
    raw_wrapper: ExtremumCircularWrapper

    extremas_id: int
    full_extrema_id: int

    window: int = 300

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError("window must be greater than zero")

        self.extremas_view = self.raw_wrapper.out_cube[:, self.extremas_id]
        self.full_extremas_view = self.raw_wrapper.out_cube[:, self.full_extrema_id]

        self.seen_raw_peaks = set()
        self.seen_super_peaks = set()
        self.seen_full_peaks = set()

    def update(self):
        batch_size = self.raw_wrapper.current_batch_size
        if batch_size == 0:
            return


        head = self.raw_wrapper.head

        new_total_count = self.raw_wrapper.total_count + batch_size

        integrals_kernel.compute_super_extremas_realtime_circular(
            self.raw_wrapper.values,
            self.raw_wrapper.raw_extremas_view,
            self.extremas_view,
            self.full_extremas_view,
            head,
            batch_size,
            self.window,
            new_total_count
        )

        self.track_cumulative_peaks()
        self.print_debug_info()

    def track_cumulative_peaks(self):
        if not is_check_extrema_num:
            return

        abs_time = self.raw_wrapper.abs_time_view
        if abs_time is None:
            return

        raw_ext = self.raw_wrapper.raw_extremas_view
        sup_ext = self.extremas_view
        full_ext = self.full_extremas_view

        valid_raw = np.where(np.abs(raw_ext) > 0.5)[0]
        for t in abs_time[valid_raw]:
            if t > 0:
                self.seen_raw_peaks.add(t)

        valid_sup = np.where(np.abs(sup_ext) > 0.5)[0]
        for t in abs_time[valid_sup]:
            if t > 0:
                self.seen_super_peaks.add(t)

        valid_full = np.where(np.abs(full_ext) > 0.5)[0]
        for t in abs_time[valid_full]:
            if t > 0:
                self.seen_full_peaks.add(t)


        head = self.raw_wrapper.head
        batch_size = self.raw_wrapper.current_batch_size
        capacity = self.raw_wrapper.out_cube.shape[0]

        if batch_size > 0:

            idx = integrals_kernel.n_get_next_head(head, batch_size - 1, capacity)
            last_price = self.raw_wrapper.values[idx]
            price_str = f"| Current Batch Last Price: {last_price}"
        else:
            price_str = "| No new prices in this batch"

        print(f"""Status: {self.window} | {len(self.seen_raw_peaks)} | {len(self.seen_super_peaks)} | {len(self.seen_full_peaks)} | {price_str}""")

    def print_debug_info(self):
        if not _is_test:
            return

        raw_peaks = np.sum(self.raw_wrapper.raw_extremas_view == 1)
        raw_valleys = np.sum(self.raw_wrapper.raw_extremas_view == -1)
        raw_total = raw_peaks + raw_valleys

        super_peaks = np.sum(self.extremas_view == 1)
        super_valleys = np.sum(self.extremas_view == -1)
        super_total = super_peaks + super_valleys

        hist_peaks = np.sum(self.full_extremas_view == 1)
        hist_valleys = np.sum(self.full_extremas_view == -1)
        hist_total = hist_peaks + hist_valleys

        overlap_peaks = np.sum((self.full_extremas_view == 1) & (self.raw_wrapper.raw_extremas_view == 1))
        overlap_valleys = np.sum((self.full_extremas_view == -1) & (self.raw_wrapper.raw_extremas_view == -1))
        overlap_total = overlap_peaks + overlap_valleys

        total_batch_size = self.raw_wrapper.current_batch_size
        print(
            f"""Status: {total_batch_size} | {self.window} | {raw_total} | {raw_peaks} | {raw_valleys} | {super_total} | {super_peaks} | {super_valleys} | {hist_total} | {hist_peaks} | {hist_valleys} | {overlap_total} | {overlap_peaks} | {overlap_valleys}"""
            f"Unified Window: {self.window}\n"
            f"  -> Raw extrema: {raw_total} (peaks:{raw_peaks}, valleys:{raw_valleys})\n"
            f"  -> Active super extrema: {super_total} (peaks:{super_peaks}, valleys:{super_valleys})\n"
            f"  -> Full super-extrema history: {hist_total} (peaks:{hist_peaks}, valleys:{hist_valleys})\n"
            f"  -> Super/raw overlaps: {overlap_total} (peaks:{overlap_peaks}, valleys:{overlap_valleys})"
        )
