'Internal implementation for the volume-impulse demo.'
import os

# os.environ["CHART_QT_LOGGING_RULES"] = "qt.qpa.*=true;qt.widgets.*=true"

import threading
from collections import deque
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg

# ---- Monkey Patch to prevent PySide6 Heap Corruption with fillLevel/downsample ----
_original_generatePath = pg.PlotCurveItem.generatePath

def _safe_generatePath(self, *args, **kwargs):
    import pyqtgraph.functions as fn
    _old_arrayToQPath = fn.arrayToQPath

    kept_arrays = []
    def _hook(x, y, *args, **kwargs):
        kept_arrays.append((x, y))
        try:
            return _old_arrayToQPath(x, y, *args, **kwargs)
        except TypeError as e:
            if 'finiteCheck' in str(e) and 'finiteCheck' in kwargs:
                kwargs.pop('finiteCheck')
                return _old_arrayToQPath(x, y, *args, **kwargs)
            raise

    fn.arrayToQPath = _hook
    try:
        path = _original_generatePath(self, *args, **kwargs)
    finally:
        fn.arrayToQPath = _old_arrayToQPath

    # Keep strong references to the local arrays (x2, y2) generated for fillLevel/downsampling.
    # This prevents PySide6's QPolygonF from pointing to freed memory, fixing 0xc0000374 Heap Corruption!
    self._fill_keepalive = kept_arrays
    return path

pg.PlotCurveItem.generatePath = _safe_generatePath
# ----------------------------------------------------------------------

from .chart_core_utils import _is_test, time_it, test_log
from .chart_builder import ChartBuilder
from .chart_renderer import ChartRenderer
from .chart_events import ChartEvents

pg.setConfigOptions(antialias=False)
pg.setConfigOption('crashWarning', True)

class HighFreqTickChart(pg.GraphicsLayoutWidget):
    sig_init_energy_plots = Signal(int)
    sig_request_render = Signal()

    def __init__(self, rect_min_width=10, rect_pad_width=10):
        super().__init__()
        self._closing = False
        self.view_driven_updates = False
        self.rect_min_width = rect_min_width
        self.rect_pad_width = rect_pad_width
        self.sync_lock = threading.RLock()

        self.sig_init_energy_plots.connect(self._init_energy_plots, Qt.ConnectionType.QueuedConnection)
        self.sig_request_render.connect(self._do_render_curves_gui, Qt.ConnectionType.QueuedConnection)

        self.can_auto = True
        self.chunk_cap = None
        self.num_channels = None
        self.dyn_ptr = 0
        self.global_idx = 0
        self.has_baked = False
        self.need_static_render = False
        self.following_mouse = True

        self.energy_plots = []
        self.energy_curves_static = []
        self.energy_curves_dyn = []
        self.crosshairs_p2 = []
        self.avg_lines = []

        self.time_axis = None

        ChartBuilder.init_main_plot(self)
        self.p1.vb.sigXRangeChanged.connect(self._on_x_range_changed)
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def _init_energy_plots(self, num_channels):
        with self.sync_lock:
            if hasattr(self, 'num_channels') and self.num_channels == num_channels and getattr(self, 'energy_plots', None):
                return
            self.num_channels = num_channels
            ChartBuilder.build_energy_plots(self, num_channels)

    def set_store(self, store):
        with self.sync_lock:
            self.store = store
            self.num_channels = store.num_channels
            self.sig_init_energy_plots.emit(self.num_channels)

    @property
    def data_x(self): return self.store.data_x if getattr(self, 'store', None) else None
    @property
    def data_bid(self): return self.store.data_bid if getattr(self, 'store', None) else None
    @property
    def data_ask(self): return self.store.data_ask if getattr(self, 'store', None) else None
    @property
    def data_ma(self): return self.store.data_ma if getattr(self, 'store', None) else None
    @property
    def data_e_val(self): return self.store.data_e_val if getattr(self, 'store', None) else None
    @property
    def data_e_flag(self): return self.store.data_e_flag if getattr(self, 'store', None) else None
    @property
    def current_avg(self):
        if hasattr(self, 'store') and self.store:
            return self.store.get_current_avg()
        return None

    @property
    def global_timestamps(self):
        if hasattr(self, 'store') and getattr(self, 'store', None) is not None:
            with self.store.queue_lock:
                try:
                    return deque(self.store.global_timestamps)
                except Exception:
                    return deque()
        return deque()

    @property
    def data_ptr(self):
        if hasattr(self, 'store') and self.store:
            with self.store.queue_lock:
                return self.store.data_ptr
        return 0

    @data_ptr.setter
    def data_ptr(self, value):
        if hasattr(self, 'store') and self.store:
            with self.store.queue_lock:
                self.store.data_ptr = value

    @property
    def global_idx(self):
        if hasattr(self, 'store') and self.store:
            with self.store.queue_lock:
                return self.store.global_idx
        return 0

    @global_idx.setter
    def global_idx(self, value):
        if hasattr(self, 'store') and self.store:
            with self.store.queue_lock:
                self.store.global_idx = value

    def _on_x_range_changed(self, vb, xrange):
        with self.sync_lock:
            ChartEvents.on_x_range_changed(self, vb, xrange)

    def _auto_range(self):
        with self.sync_lock:
            ChartEvents.auto_range(self)

    def reset_energy_status(self):
        with self.sync_lock:
            if self.can_auto:
                self._pending_auto_range = True

            if getattr(self, 'num_channels', None) is None:
                return

            try:
                if hasattr(self, 'store') and self.store:
                    self.store.reset_energy_status()
                self.need_static_render = True
            except Exception as e:
                test_log(f"CRITICAL ERROR in reset_energy_status: {e}")

    def _sync_avg_lines_to_gui(self):
        ChartRenderer.sync_avg_lines_to_gui(self)

    @time_it
    def ingest_data(self, energy_batch, prices_batch, avg_energy=None, times_batch=None):
        if getattr(self, 'store', None) is None:
            return

        with self.sync_lock:
            self.store.ingest_data(
                energy_batch,
                prices_batch,
                self.rect_min_width,
                self.rect_pad_width,
                avg_energy,
                times_batch
            )

            self.need_static_render = True

            if self.can_auto:
                self._pending_auto_range = True

            self.render_curves()

    def _get_logical_slice(self, arr, logical_start, logical_end):
        if getattr(self, 'store', None) is None:
            return np.array([], dtype=np.float64)
        return self.store.get_logical_slice(arr, logical_start, logical_end)

    def _get_visible_p1_data(self, v_min, v_max):
        visible_p1 = []
        s_d, e_d = 0, 0

        with self.sync_lock:
            if hasattr(self, 'store') and self.store:
                d_ptr, _ = self.store.get_sync_state()
                if d_ptr > 0:
                    capacity = self.store.capacity

                    s_d = max(0, d_ptr - capacity, int(v_min))
                    e_d = min(d_ptr, int(v_max) + 1)

                    if e_d > s_d:
                        visible_p1.extend([
                            self._get_logical_slice(self.data_bid, s_d, e_d),
                            self._get_logical_slice(self.data_ask, s_d, e_d),
                            self._get_logical_slice(self.data_ma, s_d, e_d)
                        ])

        return visible_p1, (s_d, e_d)

    @time_it
    def _update_y_range(self, v_min, v_max):
        with self.sync_lock:
            ChartRenderer.update_y_range(self, v_min, v_max)

    def render_curves(self):
        with self.sync_lock:
            if getattr(self, '_closing', False):
                return
            if not getattr(self, '_render_queued', False):
                self._render_queued = True
                self.sig_request_render.emit()

    @time_it
    def _do_render_curves_gui(self):
        with self.sync_lock:
            if getattr(self, '_closing', False):
                self._render_queued = False
                return
            self._render_queued = False
            try:
                if getattr(self, 'p1', None) is None:
                    return

                if getattr(self, '_pending_auto_range', False):
                    self._pending_auto_range = False
                    if self.can_auto:
                        self._auto_range()

                self._sync_avg_lines_to_gui()

                view_range = self.p1.vb.viewRange()
                if not view_range: return
                v_min, v_max = view_range[0]

                if not (np.isfinite(v_min) and np.isfinite(v_max)):
                    return

                if not getattr(self, 'view_driven_updates', False):
                    import time
                    now_y = time.time()
                    if (not hasattr(self, '_last_y_update_time')) or (now_y - self._last_y_update_time > 0.2) or getattr(self, '_pending_auto_range', False):
                        self._last_y_update_time = now_y
                        ChartRenderer.update_y_range(self, v_min, v_max)

                self._render_layer_curves(v_min=v_min, v_max=v_max)

            except Exception as e:
                test_log(f"""Status: {e}""")
                import traceback
                test_log(traceback.format_exc())

    def _get_fallback_price(self, data_arr):
        with self.sync_lock:
            d_ptr = self.store.get_sync_state()[0] if hasattr(self, 'store') and self.store else 0
            if d_ptr <= 0: return 0.0
            search_len = min(d_ptr, 2000)
            sliced = self._get_logical_slice(data_arr, d_ptr - search_len, d_ptr)

            if not isinstance(sliced, np.ndarray) or sliced.size == 0:
                return 0.0

            valid_mask = np.isfinite(sliced) & (sliced != 0)
            if np.any(valid_mask):
                return float(sliced[valid_mask][-1])
            return 0.0

    @time_it
    def _render_layer_curves(self, v_min, v_max):
        if not hasattr(self, 'store') or not self.store: return
        d_ptr, _ = self.store.get_sync_state()
        if d_ptr <= 0 or getattr(self, 'data_x', None) is None:
            return

        capacity = self.store.capacity

        s_idx = max(0, d_ptr - capacity, int(v_min) - 1000)
        e_idx = min(d_ptr, int(v_max) + 1000)

        if e_idx <= s_idx:
            return

        x_slice = self._get_logical_slice(self.data_x, s_idx, e_idx)

        curves_data = [
            (self.curve_bid_dyn, self.data_bid),
            (self.curve_ask_dyn, self.data_ask),
            (self.curve_ma_dyn,  self.data_ma)
        ]

        if _is_test:
            test_log(f"""Status: {d_ptr} | {s_idx} | {e_idx}""")

        for curve, data in curves_data:
            slice_data = self._get_logical_slice(data, s_idx, e_idx)

            if len(x_slice) > 0 and len(x_slice) == len(slice_data):
                try:
                    if getattr(self, 'p1', None) and getattr(self.p1, 'vb', None):
                        # Ensure memory is contiguous and kept alive during Qt paintEvent
                        x_view = np.ascontiguousarray(x_slice)
                        y_view = np.ascontiguousarray(slice_data)
                        if not hasattr(curve, '_keepalive_queue'):
                            curve._keepalive_queue = deque(maxlen=3)
                        curve._keepalive_queue.append((x_view, y_view))
                        curve.setData(x=x_view, y=y_view)
                except Exception:
                    pass
            else:
                try:
                    empty_x = np.array([], dtype=np.float64)
                    empty_y = np.array([], dtype=np.float64)
                    if not hasattr(curve, '_keepalive_queue'):
                        curve._keepalive_queue = deque(maxlen=3)
                    curve._keepalive_queue.append((empty_x, empty_y))
                    curve.setData(x=empty_x, y=empty_y)
                except Exception:
                    pass

        if getattr(self, 'num_channels', None) is None: return
        if not getattr(self, 'energy_curves_dyn', None) or len(self.energy_curves_dyn) != self.num_channels: return

        current_avg_copy = self.store.get_current_avg()

        for c in range(self.num_channels):
            e_val = self.data_e_val
            e_flag = self.data_e_flag
            if e_val is None or e_flag is None:
                continue

            val_up = self._get_logical_slice(e_val[c, 0], s_idx, e_idx)
            flag_up = self._get_logical_slice(e_flag[c, 0], s_idx, e_idx)
            val_dn = self._get_logical_slice(e_val[c, 1], s_idx, e_idx)
            flag_dn = self._get_logical_slice(e_flag[c, 1], s_idx, e_idx)

            for i in range(16):
                is_up_dir = (i < 4) or (8 <= i < 12)
                curve = self.energy_curves_dyn[c][i]

                target_val = val_up if is_up_dir else val_dn
                target_flag = flag_up if is_up_dir else flag_dn

                flag_to_check = i if is_up_dir else (i - 4)

                if len(x_slice) > 0 and len(target_val) == len(x_slice) and len(target_flag) == len(x_slice):
                    ey = np.where(target_flag == flag_to_check, target_val, 0.0)
                    if not is_up_dir:
                        ey = -ey
                    try:
                        if getattr(self, 'p1', None) and getattr(self.p1, 'vb', None):
                            # Ensure memory is contiguous and kept alive during Qt paintEvent
                            x_view = np.ascontiguousarray(x_slice)
                            y_view = np.ascontiguousarray(ey)
                            if not hasattr(curve, '_keepalive_queue'):
                                curve._keepalive_queue = deque(maxlen=3)
                            curve._keepalive_queue.append((x_view, y_view))
                            curve.setData(x=x_view, y=y_view)
                    except Exception:
                        pass
                else:
                    try:
                        empty_x = np.array([], dtype=np.float64)
                        empty_y = np.array([], dtype=np.float64)
                        if not hasattr(curve, '_keepalive_queue'):
                            curve._keepalive_queue = deque(maxlen=3)
                        curve._keepalive_queue.append((empty_x, empty_y))
                        curve.setData(x=empty_x, y=empty_y)
                    except Exception:
                        pass

            if current_avg_copy is not None:
                avg_up = current_avg_copy[c, 0]
                avg_dn = current_avg_copy[c, 1]

                if np.isfinite(avg_up) and self.avg_lines[c][0]:
                    self.avg_lines[c][0].setPos(float(avg_up))
                if np.isfinite(avg_dn) and self.avg_lines[c][1]:
                    self.avg_lines[c][1].setPos(float(-np.abs(avg_dn)))

    def mousePressEvent(self, ev):
        try:
            if ev.button() == Qt.MouseButton.LeftButton:
                with self.sync_lock:
                    self.can_auto = False
                    self.following_mouse = not self.following_mouse
                    if not self.following_mouse:
                        if getattr(self, 'h_line1', None) and self.h_line1.isVisible(): self.h_line1.hide()
                        if getattr(self, 'crosshairs_p2', None):
                            for _, h_line in self.crosshairs_p2:
                                if h_line.isVisible(): h_line.hide()
            super().mousePressEvent(ev)
        except Exception as e:
            test_log(f"mousePressEvent error: {e}")

    def mouseDoubleClickEvent(self, ev):
        try:
            if ev.button() == Qt.MouseButton.LeftButton:
                with self.sync_lock:
                    self.can_auto = True
                    self._auto_range()
            super().mouseDoubleClickEvent(ev)
        except Exception as e:
            pass

    def wheelEvent(self, ev):
        try:
            with self.sync_lock:
                self.can_auto = False
            super().wheelEvent(ev)
        except Exception as e:
            pass

    def _mouse_moved(self, evt):
        with self.sync_lock:
            ChartEvents.mouse_moved(self, evt)

    def closeEvent(self, event=None):
        try:
            if hasattr(self, 'proxy') and self.proxy:
                self.proxy.disconnect()
                self.proxy = None

            with self.sync_lock:
                self.can_auto = False
                self._closing = True
                self.following_mouse = False
                self._render_queued = False

                if hasattr(self, 'p1') and self.p1:
                    try:
                        self.p1.vb.sigXRangeChanged.disconnect(self._on_x_range_changed)
                    except Exception:
                        pass

                    try:
                        empty_x = np.array([], dtype=np.float64)
                        empty_y = np.array([], dtype=np.float64)
                        if not hasattr(self.curve_bid_dyn, '_keepalive_queue'): self.curve_bid_dyn._keepalive_queue = deque(maxlen=3)
                        self.curve_bid_dyn._keepalive_queue.append((empty_x, empty_y))
                        self.curve_bid_dyn.setData(x=empty_x, y=empty_y)
                    except Exception:
                        pass
                    try:
                        empty_x = np.array([], dtype=np.float64)
                        empty_y = np.array([], dtype=np.float64)
                        if not hasattr(self.curve_ask_dyn, '_keepalive_queue'): self.curve_ask_dyn._keepalive_queue = deque(maxlen=3)
                        self.curve_ask_dyn._keepalive_queue.append((empty_x, empty_y))
                        self.curve_ask_dyn.setData(x=empty_x, y=empty_y)
                    except Exception:
                        pass
                    try:
                        empty_x = np.array([], dtype=np.float64)
                        empty_y = np.array([], dtype=np.float64)
                        if not hasattr(self.curve_ma_dyn, '_keepalive_queue'): self.curve_ma_dyn._keepalive_queue = deque(maxlen=3)
                        self.curve_ma_dyn._keepalive_queue.append((empty_x, empty_y))
                        self.curve_ma_dyn.setData(x=empty_x, y=empty_y)
                    except Exception:
                        pass

                ChartBuilder.cleanup_energy_plots(self)

                t_axis = getattr(self, 'time_axis', None)
                if t_axis:
                    try:
                        t_axis.chart_ref = None
                    except: pass

                if hasattr(self, 'store') and self.store:
                    if os.environ.get("CHART_CLEAR_STORE_ON_CLOSE") == "1":
                        self.store.clear_memory()

        except Exception as e:
            test_log(f"closeEvent error: {e}")

        if event:
            super().closeEvent(event)
