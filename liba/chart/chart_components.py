import pyqtgraph as pg
import numpy as np
from datetime import datetime
from .chart_core_utils import test_log, QtDebugContext, validate_qt_pointer, safe_qt_operation, QT_DEBUG_MODE

class SafeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tick_count = 100

    def tickValues(self, minVal, maxVal, size):
        with QtDebugContext(f"SafeAxisItem.tickValues [{minVal:.2f}, {maxVal:.2f}]"):
            try:
                if not (np.isfinite(minVal) and np.isfinite(maxVal)):
                    test_log(f"Invalid range in tickValues: min={minVal}, max={maxVal}")
                    return []
                if abs(maxVal - minVal) > 1e15:
                    test_log(f"Range too wide in tickValues: {maxVal - minVal}")
                    return []

                # Validate linked view before operation
                if not validate_qt_pointer(self.linkedView(), "SafeAxisItem.linkedView"):
                    return []

                ticks = safe_qt_operation("super().tickValues", super().tickValues, minVal, maxVal, size)
                if ticks is None:
                    return []


                total_ticks = sum(len(t[1]) for t in ticks)
                if total_ticks > self.max_tick_count:
                    test_log(f"Too many ticks ({total_ticks}), limiting to {self.max_tick_count}")
                    return [(ticks[0][0], ticks[0][1][:self.max_tick_count])]
                return ticks
            except Exception as e:
                test_log(f"Exception in SafeAxisItem.tickValues: {e}")
                if QT_DEBUG_MODE:
                    import traceback
                    test_log(f"Traceback: {traceback.format_exc()}")
                return []

class SimplifiedTimeAxis(SafeAxisItem):
    def __init__(self, chart_ref=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chart_ref = chart_ref
        self.max_tick_count = 1000

    def paint(self, p, opt, widget):
        try:

            vb = self.linkedView()
            if vb is not None:
                vr = vb.viewRange()[0]
                if not (np.isfinite(vr[0]) and np.isfinite(vr[1])):
                    return
            super().paint(p, opt, widget)
        except Exception:
            pass

    def tickStrings(self, values, scale, spacing):
        strs = []
        if not self.chart_ref or not hasattr(self.chart_ref, 'store') or not self.chart_ref.store:
            return [""] * len(values)

        try:
            for v in values:
                if np.isfinite(v):
                    t = self.chart_ref.store.get_timestamp_by_logical_x(v)
                    if t is not None:
                        if isinstance(t, (int, float, np.number)):
                            strs.append(datetime.fromtimestamp(t).strftime('%H:%M:%S.%f')[:-3])
                        else:
                            strs.append(str(t))
                    else:
                        strs.append("")
                else:
                    strs.append("")
        except Exception:
            strs = [""] * len(values)

        return strs
