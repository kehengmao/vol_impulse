"""PySide6 window controller for second-level market visualization."""

import sys
import threading
from collections import deque
import traceback

from PySide6 import QtCore, QtWidgets

import numpy as np

from .chart import HighFreqTickChart
from .chart_core_utils import write_crash_log
from .chart_store import RingBufferDataStore

DEFAULT_RENDER_INTERVAL_MS = 1000


class Drawer(QtWidgets.QMainWindow):
    """Buffer worker-thread data and render it safely on the Qt main thread."""

    def __init__(
        self,
        max_plot_num=100000,
        min_width=10,
        pad_width=10,
        render_interval_ms=DEFAULT_RENDER_INTERVAL_MS,
    ):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        super().__init__()
        if render_interval_ms <= 0:
            raise ValueError("render_interval_ms must be greater than zero")
        self.max_plot_num = max_plot_num
        self.tick_size = 0.5
        self.min_width = min_width
        self.pad_width = pad_width
        self.render_interval_ms = int(render_interval_ms)
        self._running = True
        self.chart = None
        self.store = None

        self.prices_queue = deque()
        self.cmd_queue = deque()
        self.latest_energy = None
        self.latest_avg = None
        self.queue_lock = threading.RLock()

        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self._process_queue_and_render)

        self._create_chart()

    def _create_chart(self):
        self.chart = HighFreqTickChart(rect_min_width=self.min_width, rect_pad_width=self.pad_width)
        self.setCentralWidget(self.chart)
        self.resize(1000, 800)

    def update_tick_size(self, tick_size):
        self.tick_size = tick_size

    def push(self, energy_array, prices_array, avg_energy=None, times_array=None):
        """Publish the latest analysis frame to the GUI thread."""
        with self.queue_lock:
            if prices_array is not None:
                self.prices_queue.append((prices_array, times_array))

            if energy_array is not None:
                self.latest_energy = energy_array
            if avg_energy is not None:
                self.latest_avg = avg_energy

    def start(self, logic_func=None):
        """Start optional worker logic and enter the Qt event loop."""
        self.show()
        self.render_timer.start(self.render_interval_ms)

        if logic_func:
            t = threading.Thread(target=logic_func, args=(self,), daemon=True)
            t.start()
        sys.exit(self.app.exec())

    def _process_queue_and_render(self):
        """Merge queued samples and render one frame on the GUI thread."""
        try:
            with self.queue_lock:
                cmds = list(self.cmd_queue)
                self.cmd_queue.clear()

                prices_batches = list(self.prices_queue)
                self.prices_queue.clear()

                energy = self.latest_energy
                avg = self.latest_avg


                self.latest_energy = None
                self.latest_avg = None

            if self.chart is not None:
                for cmd in cmds:
                    if cmd == "RESET_ENERGY":
                        self.chart.reset_energy_status()

            if not prices_batches and energy is None and avg is None:
                return

            if self.chart is not None:
                if len(prices_batches) > 500:
                    write_crash_log(f"WARNING: Drawer prices backlog ({len(prices_batches)}). Dropping old ticks.")
                    prices_batches = prices_batches[-500:]

                all_prices = []
                all_times = []

                for p, t in prices_batches:
                    if p is not None:
                        try:
                            if np.all(np.isnan(p)):
                                continue
                        except Exception:
                            pass
                        all_prices.append(p)
                        if t is not None:
                            if isinstance(t, (list, np.ndarray)):
                                all_times.extend(t)
                            else:
                                all_times.append(t)

                if all_prices:
                    cat_prices = np.vstack(all_prices)
                    cat_times = all_times if len(all_times) == cat_prices.shape[0] else None
                else:
                    cat_prices = None
                    cat_times = None

                if cat_prices is not None or energy is not None or avg is not None:
                    if self.store is None and energy is not None and cat_prices is not None:
                        e_arr = np.asarray(energy)
                        p_arr = np.asarray(cat_prices)
                        if p_arr.ndim == 1:
                            p_arr = p_arr.reshape(1, -1)
                        batch_size = p_arr.shape[0]
                        if batch_size > 0 and e_arr.shape[0] > 0:
                            num_channels = e_arr.shape[1] if e_arr.ndim > 2 else 1
                            init_capacity = max(self.max_plot_num, batch_size)
                            self.store = RingBufferDataStore(num_channels, init_capacity, capacity_multiplier=2)
                            self.chart.set_store(self.store)

                    self.chart.ingest_data(energy, cat_prices, avg, cat_times)

                self.chart.render_curves()

        except Exception as e:
            error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            write_crash_log(f"Error in _process_queue_and_render:\n{error_msg}")
            traceback.print_exc()

    def stop(self):
        self._running = False
        self.render_timer.stop()
        self.app.quit()

    def reset_energy_status(self):
        with self.queue_lock:
            self.cmd_queue.append("RESET_ENERGY")
