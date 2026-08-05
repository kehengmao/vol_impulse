import time
import threading
import numpy as np
from collections import deque
from .chart_core_utils import test_log, _is_test
from .chart_geometry import ChartGeometry

class RingBufferPointerManager:
    @staticmethod
    def get_write_slices(logical_start, logical_end, capacity):
        length = logical_end - logical_start
        if length <= 0:
            return []

        if length > capacity:
            logical_start = logical_end - capacity
            length = capacity

        phys_start = (logical_start % capacity + capacity) % capacity
        phys_end = (logical_end % capacity + capacity) % capacity

        if phys_start < phys_end and length == (phys_end - phys_start):
            return [(phys_start, phys_end, 0, length)]
        else:
            part1_len = capacity - phys_start
            return [
                (phys_start, capacity, 0, part1_len),
                (0, length - part1_len, part1_len, length)
            ]

    @staticmethod
    def get_read_slices(current_ptr, logical_start, logical_end, capacity):
        logical_start = max(0, current_ptr - capacity, int(logical_start))
        logical_end = min(current_ptr, int(logical_end))

        if logical_start >= logical_end:
            return []

        phys_start = (logical_start % capacity + capacity) % capacity
        phys_end = (logical_end % capacity + capacity) % capacity
        length = logical_end - logical_start

        if phys_start < phys_end and length == (phys_end - phys_start):
            return [(phys_start, phys_end)]
        else:
            return [(phys_start, capacity), (0, phys_end)]

class RingBufferDataStore:
    def __init__(self, num_channels, base_capacity, capacity_multiplier=2):
        self.num_channels = num_channels
        self.capacity = base_capacity * capacity_multiplier

        self.queue_lock = threading.Lock()

        test_log(f"Initializing RingBufferDataStore: capacity={self.capacity}, channels={num_channels}")

        self.data_x = np.zeros(self.capacity, dtype=float)
        self.data_bid = np.zeros(self.capacity, dtype=float)
        self.data_ask = np.zeros(self.capacity, dtype=float)
        self.data_ma = np.zeros(self.capacity, dtype=float)

        self.data_e_val = np.zeros((num_channels, 2, self.capacity), dtype=np.float64)
        self.data_e_flag = np.full((num_channels, 2, self.capacity), -1, dtype=np.int8)

        self.data_ptr = 0
        self.global_idx = 0
        self.current_avg = np.full((num_channels, 2), np.nan, dtype=float)
        self.global_timestamps = deque(maxlen=self.capacity)

        self.stats_dict = {
            'total_raw_peaks': set(),
            'total_raw_valleys': set(),
            'total_super_peaks': set(),
            'total_super_valleys': set(),
            'total_full_peaks': set(),
            'total_full_valleys': set(),
            'total_overlap_peaks': set(),
            'total_overlap_valleys': set()
        }

    def get_sync_state(self):
        with self.queue_lock:
            return self.data_ptr, self.global_idx

    def get_current_avg(self):
        with self.queue_lock:
            if self.current_avg is not None:
                return self.current_avg.copy()
            return None

    def get_timestamp_by_logical_x(self, x_val):
        with self.queue_lock:
            q_len = len(self.global_timestamps)
            if q_len == 0:
                return None
            offset = self.global_idx - q_len
            idx = int(x_val) - offset
            if 0 <= idx < q_len:
                return self.global_timestamps[idx]
            return None

    def clear_memory(self):
        with self.queue_lock:
            self.data_x = None
            self.data_bid = None
            self.data_ask = None
            self.data_ma = None
            self.data_e_val = None
            self.data_e_flag = None
            self.global_timestamps.clear()

    def _sanitize_arrays(self, energy_batch, prices_batch):
        if prices_batch is None:
            return None, None, 0

        try:
            p_arr = np.asarray(prices_batch, dtype=float)
            if p_arr.ndim == 1:
                p_arr = p_arr.reshape(1, -1)
            batch_size = p_arr.shape[0]
            if batch_size == 0:
                return None, None, 0
        except Exception:
            return None, None, 0

        if p_arr.shape[1] < 2:
            pad_cols = 2 - p_arr.shape[1]
            padding = np.full((batch_size, pad_cols), np.nan, dtype=float)
            p_arr = np.concatenate([p_arr, padding], axis=1)

        e_arr = None
        if energy_batch is not None:
            try:
                e_arr = np.asarray(energy_batch, dtype=float)
                if e_arr.shape[0] > 0:
                    e_arr = np.nan_to_num(e_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    e_arr = np.clip(e_arr, -1e7, 1e7)
                else:
                    e_arr = None
            except Exception:
                pass

        p_arr[np.isinf(p_arr)] = np.nan
        p_arr = np.where(np.isfinite(p_arr), np.clip(p_arr, -1e7, 1e7), p_arr)

        return e_arr, p_arr, batch_size

    def _update_latest_avg(self, avg_energy):
        self.current_avg = np.full((self.num_channels, 2), np.nan, dtype=float)
        try:
            if avg_energy is not None:
                arr = np.asarray(avg_energy, dtype=float)
                if arr.size > 0 and not np.all(np.isnan(arr)):
                    if arr.ndim == 1 and len(arr) == 2 and self.num_channels == 1:
                        self.current_avg = arr.reshape(1, 2)
                    elif arr.ndim == 2:
                        if arr.shape == (self.num_channels, 2):
                            self.current_avg = arr.copy()
                        elif arr.shape[1] == 2:
                            if self.num_channels == 1:
                                self.current_avg = arr[-1].reshape(1, 2)
                            else:
                                c_len = min(arr.shape[0], self.num_channels)
                                self.current_avg[:c_len] = arr[:c_len]
                    elif arr.ndim == 3:
                        if arr.shape[1] == self.num_channels and arr.shape[2] == 2:
                            self.current_avg = arr[-1].copy()
                        else:
                            c_len = min(arr.shape[1], self.num_channels)
                            self.current_avg[:c_len] = arr[-1, :c_len]
        except Exception as e:
            test_log(f"Error parsing latest avg: {e}")

    def ingest_data(self, energy_batch, prices_batch, rect_min_width, rect_pad_width, avg_energy=None, times_batch=None):
        try:
            if _is_test:
                test_log(f"""Status: {np.shape(energy_batch)} | {np.shape(prices_batch)}""")
            with self.queue_lock:
                self._ingest_data_internal(energy_batch, prices_batch, rect_min_width, rect_pad_width, avg_energy, times_batch)
        except Exception as e:
            test_log(f"CRITICAL ERROR in ingest_data (dirty data rejected): {e}")

    def _ingest_data_internal(self, energy_batch, prices_batch, rect_min_width, rect_pad_width, avg_energy, times_batch):
        if self.data_x is None:
            return
        e_arr, p_arr, batch_size = self._sanitize_arrays(energy_batch, prices_batch)
        if p_arr is None:
            return

        bids, asks = p_arr[:, 0], p_arr[:, 1]
        mas = p_arr[:, 2] if p_arr.shape[1] > 2 else np.full(batch_size, np.nan)

        self._update_latest_avg(avg_energy)

        if times_batch is not None:
            timestamps_or_strs = times_batch
        else:
            now = time.time()
            timestamps_or_strs = now - (batch_size - 1 - np.arange(batch_size)) * 0.1

        self.global_timestamps.extend(timestamps_or_strs)

        end_logical = self.global_idx + batch_size
        start_logical = self.global_idx

        if e_arr is not None:
            L = e_arr.shape[0]
            H_len = L - batch_size

            if H_len < 0:
                pad_shape = list(e_arr.shape)
                pad_shape[0] = -H_len
                padding = np.zeros(pad_shape, dtype=e_arr.dtype)
                e_arr = np.concatenate([padding, e_arr], axis=0)
                H_len = 0
                L = batch_size

            start_logical = self.global_idx - H_len

            if start_logical < 0:
                clip_len = -start_logical
                start_logical = 0
                e_arr = e_arr[clip_len:]

            batch_x = np.arange(start_logical, end_logical)

            triangles = ChartGeometry.generate_triangles(
                self.stats_dict, self.num_channels, rect_min_width, rect_pad_width, e_arr, batch_x
            )
            self._write_energy_to_ring_buffer(triangles, start_logical, end_logical)

        new_x = np.arange(self.global_idx, end_logical)
        self._write_prices_to_ring_buffer(batch_size, new_x, bids, asks, mas)

        self.global_idx += batch_size

    def _write_energy_to_ring_buffer(self, triangles, start_logical, end_logical):
        capacity = self.capacity
        slices = RingBufferPointerManager.get_write_slices(start_logical, end_logical, capacity)
        if not slices: return

        for c in range(self.num_channels):
            val_up, flag_up, val_dn, flag_dn = triangles[c]
            if len(val_up) > capacity:
                val_up = val_up[-capacity:]
                flag_up = flag_up[-capacity:]
                val_dn = val_dn[-capacity:]
                flag_dn = flag_dn[-capacity:]

            for phys_start, phys_end, data_start, data_end in slices:
                self.data_e_val[c, 0, phys_start:phys_end] = val_up[data_start:data_end]
                self.data_e_flag[c, 0, phys_start:phys_end] = flag_up[data_start:data_end]
                self.data_e_val[c, 1, phys_start:phys_end] = val_dn[data_start:data_end]
                self.data_e_flag[c, 1, phys_start:phys_end] = flag_dn[data_start:data_end]

    def _write_prices_to_ring_buffer(self, batch_size, new_x, bids, asks, mas):
        capacity = self.capacity
        start_logical = self.data_ptr
        end_logical = start_logical + batch_size

        slices = RingBufferPointerManager.get_write_slices(start_logical, end_logical, capacity)
        if not slices: return

        if batch_size > capacity:
            new_x = new_x[-capacity:]
            bids = bids[-capacity:]
            asks = asks[-capacity:]
            mas = mas[-capacity:]

        for phys_start, phys_end, data_start, data_end in slices:
            self.data_x[phys_start:phys_end] = new_x[data_start:data_end]
            self.data_bid[phys_start:phys_end] = bids[data_start:data_end]
            self.data_ask[phys_start:phys_end] = asks[data_start:data_end]
            self.data_ma[phys_start:phys_end] = mas[data_start:data_end]

        self.data_ptr += batch_size

        if _is_test:
            test_log(f"""Status: {batch_size} | {self.data_ptr} | {bids[-1]}""")

    def get_logical_slice(self, arr, logical_start, logical_end):
        with self.queue_lock:
            if arr is None:
                return np.array([], dtype=np.float64)
            capacity = self.capacity
            if capacity == 0: return np.array([], dtype=arr.dtype)

            slices = RingBufferPointerManager.get_read_slices(self.data_ptr, logical_start, logical_end, capacity)
            if not slices:
                shape = list(arr.shape)
                shape[-1] = 0
                return np.empty(shape, dtype=arr.dtype)

            if len(slices) == 1:
                phys_start, phys_end = slices[0]
                return arr[..., phys_start:phys_end]
            else:
                (phys_start1, phys_end1), (phys_start2, phys_end2) = slices
                part1 = arr[..., phys_start1:phys_end1]
                part2 = arr[..., phys_start2:phys_end2]
                return np.concatenate([part1, part2], axis=-1)

    def reset_energy_status(self):
        with self.queue_lock:
            try:
                for c in range(self.num_channels):
                    if hasattr(self, 'data_e_flag') and self.data_e_flag is not None:
                        mask_0 = (self.data_e_flag[c] == 0)
                        mask_2 = (self.data_e_flag[c] == 2)
                        mask_8 = (self.data_e_flag[c] == 8)
                        mask_10 = (self.data_e_flag[c] == 10)

                        self.data_e_flag[c][mask_0] = 1
                        self.data_e_flag[c][mask_2] = 3
                        self.data_e_flag[c][mask_8] = 9
                        self.data_e_flag[c][mask_10] = 11
            except Exception as e:
                test_log(f"CRITICAL ERROR in reset_energy_status: {e}")
