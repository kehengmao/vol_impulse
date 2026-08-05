
"""Live Binance orchestration for analysis and PySide6 visualization."""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .chart import Drawer
from .kernels import EgiDataChannel, PulsaEngine
from .objs_binance import BinanceFuturesDataWS

_is_test = False
LOGGER = logging.getLogger(__name__)


def quick_binance_setup(
    symbol: str = "BTCUSDT",
    *,
    iceberg_ratio: float = 2.0,
    min_window: int = 10,
    max_window: int = 120,
    initial_samples: int = 2000,
    max_plot_samples: int = 100000,
    peak_factor: float = 3.0,
    secondary_peak_factor: float = 0.3,
    sample_interval_ms: int = 100,
    snapshots_per_sample: int = 10,
    update_every: int = 2,
    estimate_shared_parameters: bool = False,
    rectangle_min_width: int = 4,
    rectangle_padding: int = 3,
    render_interval_ms: int = 1000,
) -> "PulsaRunner":
    """Create a configured Binance futures runner.

    Construction connects to Binance REST and WebSocket endpoints immediately.
    Call ``runner.drawer.start(runner.update)`` to start the PySide6 UI.
    """
    symbol = symbol.strip().upper()
    if not symbol or not symbol.isalnum():
        raise ValueError("symbol must contain only letters and digits")
    if initial_samples < 2:
        raise ValueError("initial_samples must be at least 2")
    if min_window <= 0 or max_window < min_window:
        raise ValueError("window bounds must satisfy 0 < min_window <= max_window")

    loader = BinanceFuturesDataWS(
        symbol=symbol,
        total_length=initial_samples,
        interval_ms=sample_interval_ms,
        actual_snapshots_per_snapshot=snapshots_per_sample,
    )
    runner = PulsaRunner(
        binance_loader=loader,
        init_fetch_num=initial_samples,
        ice_ratio=iceberg_ratio,
        min_window=min_window,
        max_window=max_window,
        peak_factor=peak_factor,
        peak_factor2=secondary_peak_factor,
        update_per_num=update_every,
        is_estimated_paras=estimate_shared_parameters,
        max_plot_num=max_plot_samples,
        min_width=rectangle_min_width,
        pad_width=rectangle_padding,
        render_interval_ms=render_interval_ms,
    )
    LOGGER.info("Binance runner ready for %s", symbol)
    return runner

@dataclass
class PulsaRunner:
    """Coordinate Binance ingestion, inference, and chart updates."""
    binance_loader: BinanceFuturesDataWS
    init_fetch_num: int = 10000
    ice_ratio: float = 2.
    min_window: int = 10
    max_window: int = 600
    peak_factor: float = 20
    peak_factor2: float = 2

    update_per_num: int = 2
    is_estimated_paras: bool = False

    max_plot_num: int = 100000
    min_width: int = 10
    pad_width: int = 10
    render_interval_ms: int = 1000


    _inited: bool = field(default=False, init=False)
    _max_energy: float = field(default=0., init=False)


    _accumulated_len: int = field(default=0, init=False)

    _counter: int = field(default=0, init=False)
    _total_push_counter: int = field(default=0, init=False)


    _internal_avg_energy_pos: np.ndarray | None = field(default=None, init=False)
    _avg_count_pos: int = field(default=0, init=False)
    _internal_avg_energy_neg: np.ndarray | None = field(default=None, init=False)
    _avg_count_neg: int = field(default=0, init=False)

    _avg_energy_out: np.ndarray | None = field(default=None, init=False)
    _lock: threading.Lock = field(init=False)

    def __post_init__(self):

        self._lock = threading.Lock()

        self.pulsa = PulsaEngine(
            self.init_fetch_num,
            self.binance_loader.tick_size,
            self.ice_ratio,
            is_estimated_paras=self.is_estimated_paras,
            min_window=self.min_window,
            max_window=self.max_window,
            peak_factor=self.peak_factor,
            peak_factor2=self.peak_factor2
        )

        self.drawer = Drawer(
            max_plot_num=min(self.max_plot_num, self.init_fetch_num),
            min_width=self.min_width,
            pad_width=self.pad_width,
            render_interval_ms=self.render_interval_ms,
        )


        internal_lookback_factor = 100
        self.full_update_per_num = int(max(1, round(self.peak_factor * internal_lookback_factor * self.max_window)))

        if _is_test:
            LOGGER.debug("Runner windows: %d..%d", self.min_window, self.max_window)

    def update(self, drawer: Drawer = None):
        """Continuously process new samples and publish chart batches."""
        try:
            LOGGER.info("Analysis loop started for %s", self.binance_loader.symbol.upper())
            last_local_time = None


            warning_interval = max(5.0, (self.binance_loader.interval_ms * 10) / 1000.0)
            last_runner_activity_time = time.time()

            while True:
                current_time = time.time()


                if current_time - last_runner_activity_time > warning_interval:
                    LOGGER.warning(
                        "No new market data for %.1f seconds (batch=%d/%d, buffered=%d)",
                        warning_interval,
                        self._counter,
                        self.update_per_num,
                        self._accumulated_len,
                    )
                    last_runner_activity_time = current_time


                    if self._inited:
                        LOGGER.info("Requesting Binance reconnection")
                        try:
                            self.binance_loader.reconnect()

                            self._counter = 1
                        except Exception as e:
                            LOGGER.exception("Binance reconnection failed: %s", e)

                try:
                    if self.binance_loader.is_updated:
                        df_all = self.binance_loader.fetch()
                        if df_all is None or df_all.empty:
                            time.sleep(0.01)
                            continue

                        if not self._inited:
                            if len(df_all) >= self.init_fetch_num:
                                last_runner_activity_time = time.time()
                                LOGGER.info(
                                    "Initializing analysis with %d market snapshots",
                                    self.init_fetch_num,
                                )

                                self.drawer.update_tick_size(self.pulsa.bin_size)

                                df_new = df_all.tail(self.init_fetch_num).copy()

                                with self._lock:
                                    df_to_load = df_new
                                    processed_len = int(self.pulsa.load_data(df_to_load) or 0)
                                    if processed_len <= 0:
                                        continue

                                    self._inited = True

                                    self._push_to_drawer(df_all, increment_len=processed_len)

                                last_local_time = df_new['local_time'].iloc[-1]
                                LOGGER.info("Analysis initialized through %s", last_local_time)
                            else:
                                last_runner_activity_time = time.time()
                                LOGGER.debug(
                                    "Waiting for initial history: %d/%d snapshots",
                                    len(df_all),
                                    self.init_fetch_num,
                                )


                        else:
                            try:
                                df_new = self._extract_new_data(df_all, last_local_time)
                                if not df_new.empty:
                                    last_runner_activity_time = time.time()
                                    if _is_test:
                                        LOGGER.debug("Processing incremental batch %s", df_new.shape)

                                    with self._lock:
                                        df_to_load = df_new
                                        processed_len = int(self.pulsa.load_data(df_to_load) or 0)
                                        if processed_len <= 0:
                                            continue

                                        self._push_to_drawer(df_all, increment_len=processed_len)

                                    last_local_time = df_new['local_time'].iloc[-1]
                            except Exception as e_inner:
                                LOGGER.exception("Incremental analysis failed: %s", e_inner)
                                raise e_inner
                except Exception as cycle_e:
                    LOGGER.exception("Runner cycle failed: %s", cycle_e)

                time.sleep(0.01)
        except Exception as e:
            LOGGER.exception("Analysis loop stopped after an error: %s", e)

    def _push_to_drawer(self, df_all: pd.DataFrame, increment_len: int):
        """Accumulate processed samples and publish complete render batches."""

        t_len = increment_len
        if _is_test:
            LOGGER.debug("Processed %d samples", t_len)
        if t_len == 0:
            return


        self._accumulated_len += t_len
        if _is_test:
            LOGGER.debug("Buffered %d samples at batch %d", self._accumulated_len, self._counter)


        if self._counter % self.update_per_num == 0:
            if _is_test:
                LOGGER.debug("Publishing chart batch %d/%d", self._counter, self.update_per_num)
            self._flush_cache_and_push(df_all)
            self._counter = 0
        else:
            if _is_test:
                LOGGER.debug("Waiting for the next chart batch")

        self._counter += 1

    def _extract_new_data(self, df_all: pd.DataFrame, last_local_time) -> pd.DataFrame:
        if last_local_time is not None:
            matches = df_all.index[df_all['local_time'] == last_local_time].tolist()
            if matches:
                last_idx = matches[-1]
                return df_all.iloc[last_idx + 1:]

        return df_all.tail(1)

    def _update_internal_avg(self, target_data, peak_dir):
        if target_data is None or target_data.size == 0:
            return

        T = target_data.shape[0]
        if T > 0:
            mask_pos = peak_dir > 0.5
            mask_neg = peak_dir < -0.5

            batch_sum_pos = np.sum(target_data * mask_pos, axis=0)
            batch_count_pos = np.sum(mask_pos, axis=0)

            batch_sum_neg = np.sum(target_data * mask_neg, axis=0)
            batch_count_neg = np.sum(mask_neg, axis=0)


            if self._internal_avg_energy_pos is None:
                self._internal_avg_energy_pos = np.zeros_like(batch_sum_pos)
                self._avg_count_pos = batch_count_pos
                np.divide(batch_sum_pos, batch_count_pos, out=self._internal_avg_energy_pos, where=batch_count_pos > 0)
            else:
                total_sum_pos = self._internal_avg_energy_pos * self._avg_count_pos + batch_sum_pos
                self._avg_count_pos += batch_count_pos
                np.divide(total_sum_pos, self._avg_count_pos, out=self._internal_avg_energy_pos, where=self._avg_count_pos > 0)


            if self._internal_avg_energy_neg is None:
                self._internal_avg_energy_neg = np.zeros_like(batch_sum_neg)
                self._avg_count_neg = batch_count_neg
                np.divide(batch_sum_neg, batch_count_neg, out=self._internal_avg_energy_neg, where=batch_count_neg > 0)
            else:
                total_sum_neg = self._internal_avg_energy_neg * self._avg_count_neg + batch_sum_neg
                self._avg_count_neg += batch_count_neg
                np.divide(total_sum_neg, self._avg_count_neg, out=self._internal_avg_energy_neg, where=self._avg_count_neg > 0)

    def _normalize(self, energy_array: np.ndarray, actual_increment: int):
        if energy_array is None or energy_array.size == 0:
            return energy_array

        target_idx = EgiDataChannel.egi_value.value
        extremas_idx = EgiDataChannel.extremas.value

        target_data = energy_array[:, :, target_idx : target_idx + 1]
        peak_dir = energy_array[:, :, extremas_idx : extremas_idx + 1]

        target_data[~np.isfinite(target_data)] = 0
        target_data[target_data < 0] = 0


        increment_data = target_data[-actual_increment:] if actual_increment > 0 else np.empty((0, target_data.shape[1], 1))
        increment_dir = peak_dir[-actual_increment:] if actual_increment > 0 else np.empty((0, peak_dir.shape[1], 1))
        self._update_internal_avg(increment_data, increment_dir)


        current_max = target_data.max() if target_data.size > 0 else 0
        old_max = self._max_energy
        new_max = max(float(current_max), float(old_max))
        self._max_energy = new_max


        if self._max_energy > 1e-10:
            energy_array[:, :, target_idx : target_idx + 1] = target_data / self._max_energy


            if self._internal_avg_energy_pos is not None and self._internal_avg_energy_neg is not None:
                norm_pos = self._internal_avg_energy_pos / self._max_energy
                norm_neg = self._internal_avg_energy_neg / self._max_energy
                self._avg_energy_out = np.concatenate([norm_pos, norm_neg], axis=-1)
        else:
            energy_array[:, :, target_idx : target_idx + 1] = 0
            if self._internal_avg_energy_pos is not None and self._internal_avg_energy_neg is not None:
                self._avg_energy_out = np.zeros_like(np.concatenate([self._internal_avg_energy_pos, self._internal_avg_energy_neg], axis=-1))

        return energy_array

    def _prepare_drawer_data(self, df_all: pd.DataFrame, actual_increment: int):


        safe_lookback = 2 * self.max_window


        if self._total_push_counter == 1:
            total_fetch_len = self.init_fetch_num
        else:
            total_fetch_len = actual_increment + safe_lookback

            if total_fetch_len > self.init_fetch_num:
                total_fetch_len = self.init_fetch_num


        energy_array = self.pulsa.to_egi_results(total_fetch_len=total_fetch_len)
        if energy_array is None or energy_array.size == 0:
            return None, None, None, None



        energy_array = self._normalize(energy_array, actual_increment)



        plot_price_len = actual_increment
        if self.max_plot_num > 0 and plot_price_len > self.max_plot_num:
            plot_price_len = self.max_plot_num




            if self._total_push_counter != 1:
                plot_energy_len = plot_price_len + safe_lookback
                if plot_energy_len < len(energy_array):
                    energy_array = energy_array[-plot_energy_len:]

        df_plot = df_all.tail(plot_price_len)
        batch_prices = df_plot[['bid', 'ask']].values
        batch_prices_f = df_plot[['bid', 'ask', 'last']].values
        batch_times = df_plot['local_time'].values

        return energy_array, batch_prices, batch_prices_f, batch_times

    def _enforce_max_plot_limit(self, energy_array, batch_prices, batch_prices_f, batch_times):
        limit = self.max_plot_num
        if limit <= 0:
            return energy_array, batch_prices, batch_prices_f, batch_times

        if energy_array is not None and len(energy_array) > limit:
            energy_array = energy_array[-limit:]

        if batch_prices is not None and len(batch_prices) > limit:
            batch_prices = batch_prices[-limit:]

        if batch_prices_f is not None and len(batch_prices_f) > limit:
            batch_prices_f = batch_prices_f[-limit:]

        if batch_times is not None and len(batch_times) > limit:
            batch_times = batch_times[-limit:]

        return energy_array, batch_prices, batch_prices_f, batch_times

    def _flush_cache_and_push(self, df_all: pd.DataFrame):
        if self._accumulated_len == 0:
            return

        self._total_push_counter += 1
        is_full_update = (self._total_push_counter % self.full_update_per_num == 0)

        actual_increment = self._accumulated_len

        if is_full_update:
            self.drawer.reset_energy_status()


        energy_array_plot, batch_prices, batch_prices_f, batch_times = self._prepare_drawer_data(
            df_all, actual_increment
        )

        if energy_array_plot is None:
            self._accumulated_len = 0
            return


        e_push = energy_array_plot
        p_push = batch_prices
        t_push = batch_times


        e_push, p_push, batch_prices_f, t_push = self._enforce_max_plot_limit(
            e_push, p_push, batch_prices_f, t_push
        )

        safe_avg_energy_out = None
        if self._avg_energy_out is not None:
            safe_avg_energy_out = np.nan_to_num(self._avg_energy_out, nan=0.0, posinf=0.0, neginf=0.0)


        if _is_test:
            print(f'Current shape of energy_array_plot {e_push.shape} and price size {len(p_push)}')

        self._debug(e_push, len(p_push), t_push, batch_prices_f)
        self.drawer.push(e_push, p_push, safe_avg_energy_out, t_push)

        self._accumulated_len = 0

        # if self._total_push_counter == 1:

        #     self.drawer.reset_energy_status()

    def _debug(self, energy_array, batch_increment, batch_times, batch_prices):
        if not _is_test:
            return
        print(f'[Runner Debug] CurrentMax: {self._max_energy}')
        if self._internal_avg_energy_pos is not None and self._internal_avg_energy_neg is not None:
            avg_pos_val = np.nanmean(self._internal_avg_energy_pos)
            avg_neg_val = np.nanmean(self._internal_avg_energy_neg)
            norm_pos_val = avg_pos_val / self._max_energy if self._max_energy > 1e-10 else 0
            norm_neg_val = avg_neg_val / self._max_energy if self._max_energy > 1e-10 else 0
            print(f'[Runner Debug] Internal raw avg - Up: {avg_pos_val:.6f} | Down: {avg_neg_val:.6f}')
            print(f'[Runner Debug] Current normalized avg - Up: {norm_pos_val:.6f} | Down: {norm_neg_val:.6f}')

        force_dir_idx = EgiDataChannel.egi_force_dir.value
        extremas_idx = EgiDataChannel.extremas.value
        extremas2_idx = getattr(EgiDataChannel, 'extremas2', None)
        extremas2_idx = extremas2_idx.value if extremas2_idx is not None else None

        recent_energy = energy_array[-batch_increment:]
        recent_force_dir = recent_energy[:, :, force_dir_idx]

        batch_gt_05 = int(np.sum(recent_force_dir > 0.5))
        batch_lt_05 = int(np.sum(recent_force_dir < 0.5))

        recent_peaks = recent_energy[:, :, extremas_idx]
        peak_mask = np.abs(recent_peaks) > 0.5
        peak_indices = np.where(peak_mask)
        num_peaks = len(peak_indices[0])

        num_peaks2 = 0
        if extremas2_idx is not None:
            recent_peaks2 = recent_energy[:, :, extremas2_idx]
            num_peaks2 = int(np.sum(np.abs(recent_peaks2) > 0.5))

        if num_peaks > 0 or batch_gt_05 > 0 or batch_lt_05 > 0:
            print(f"[Runner Push] Size: {batch_increment} | force_dir > 0.5: {batch_gt_05} | < 0.5: {batch_lt_05}")
            print(f"Peaks found: {num_peaks} | Peaks2: {num_peaks2}")
            if num_peaks > 0:
                t_idx, c_idx = peak_indices[0][0], peak_indices[1][0]
                print(f"  -> First Peak: rel_idx={t_idx}, time={batch_times[t_idx]}, price={batch_prices[t_idx]}, ch={c_idx}, dir={recent_peaks[t_idx, c_idx]}")
