import time
import numpy as np
from datetime import datetime
from PySide6.QtCore import Qt
from .chart_core_utils import test_log, QtDebugContext, validate_qt_pointer, safe_qt_operation, QT_DEBUG_MODE

class ChartEvents:
    @staticmethod
    def on_x_range_changed(chart, vb, xrange):
        try:
            if getattr(chart, '_closing', False):
                return
            v_min, v_max = xrange

            if not (np.isfinite(v_min) and np.isfinite(v_max)):
                return

            if getattr(chart, 'view_driven_updates', False):
                from .chart_renderer import ChartRenderer
                ChartRenderer.update_y_range(chart, v_min, v_max)

            chart.render_curves()

        except Exception as e:
            test_log(f"_on_x_range_changed error: {e}")

    @staticmethod
    def mouse_moved(chart, evt):
        if getattr(chart, '_closing', False):
            return
        if not chart.following_mouse:
            return

        with QtDebugContext("mouse_moved"):
            try:
                pos = evt[0]

                # Validate chart.p1 before use
                if not validate_qt_pointer(getattr(chart, 'p1', None), "chart.p1"):
                    return

                if not hasattr(chart.p1, 'sceneBoundingRect') or chart.p1.sceneBoundingRect() is None:
                    return

                p1_contains = False
                try:
                    p1_contains = chart.p1.sceneBoundingRect().contains(pos)
                except RuntimeError as e:
                    test_log(f"RuntimeError in p1.sceneBoundingRect check: {e}")
                    return

                if p1_contains:
                    try:
                        mouse_point = chart.p1.vb.mapSceneToView(pos)
                        ChartEvents.update_crosshair(chart, mouse_point.x(), mouse_point.y(), chart.p1)
                    except RuntimeError as e:
                        test_log(f"RuntimeError in p1 mouse handling: {e}")
                        pass
                elif getattr(chart, 'energy_plots', None):
                    for i, p in enumerate(chart.energy_plots):
                        if not validate_qt_pointer(p, f"energy_plots[{i}]"):
                            continue
                        try:
                            if p.sceneBoundingRect().contains(pos):
                                mouse_point = p.vb.mapSceneToView(pos)
                                ChartEvents.update_crosshair(chart, mouse_point.x(), mouse_point.y(), p)
                                break
                        except RuntimeError as e:
                            test_log(f"RuntimeError in energy_plot[{i}] check: {e}")
                            continue
            except Exception as e:
                test_log(f"Unexpected error in mouse_moved: {e}")
                if QT_DEBUG_MODE:
                    import traceback
                    test_log(f"Traceback: {traceback.format_exc()}")

    @staticmethod
    def update_crosshair(chart, x, y, active_plot):
        with QtDebugContext("update_crosshair"):
            try:
                if getattr(chart, '_closing', False):
                    return
                # Validate main chart object
                if not validate_qt_pointer(getattr(chart, 'p1', None), "chart.p1"):
                    return

                if not (np.isfinite(x) and np.isfinite(y)):
                    return

                time_str = ""
                if hasattr(chart, 'store') and chart.store:
                    t = chart.store.get_timestamp_by_logical_x(x)
                    if t is not None:
                        if isinstance(t, (int, float, np.number)):
                            time_str = datetime.fromtimestamp(t).strftime('%H:%M:%S.%f')[:-3]
                        else:
                            time_str = str(t)

                if not time_str:
                    time_str = "--:--:--"

                # Update main vertical line
                if validate_qt_pointer(chart.v_line1, "chart.v_line1"):
                    try:
                        chart.v_line1.setPos(x)
                        if hasattr(chart.v_line1, 'label') and chart.v_line1.label:
                            chart.v_line1.label.setFormat(time_str)
                    except RuntimeError as e:
                        test_log(f"RuntimeError updating v_line1: {e}")

                # Update crosshair lines for energy plots
                if getattr(chart, 'crosshairs_p2', None):
                    for i, (v_line, _) in enumerate(chart.crosshairs_p2):
                        if validate_qt_pointer(v_line, f"crosshairs_p2[{i}].v_line"):
                            try:
                                v_line.setPos(x)
                                if hasattr(v_line, 'label') and v_line.label:
                                    v_line.label.setFormat(time_str)
                                if not v_line.isVisible():
                                    v_line.show()
                            except RuntimeError as e:
                                test_log(f"RuntimeError updating crosshairs_p2[{i}].v_line: {e}")
                                continue

                if active_plot == chart.p1:
                    # Update main horizontal line
                    if validate_qt_pointer(chart.h_line1, "chart.h_line1"):
                        try:
                            chart.h_line1.setPos(y)
                            if not chart.h_line1.isVisible():
                                chart.h_line1.show()
                        except RuntimeError as e:
                            test_log(f"RuntimeError updating h_line1: {e}")

                    # Hide horizontal lines for energy plots
                    if getattr(chart, 'crosshairs_p2', None):
                        for i, (_, h_line) in enumerate(chart.crosshairs_p2):
                            if validate_qt_pointer(h_line, f"crosshairs_p2[{i}].h_line"):
                                try:
                                    if h_line.isVisible():
                                        h_line.hide()
                                except RuntimeError as e:
                                    test_log(f"RuntimeError hiding crosshairs_p2[{i}].h_line: {e}")
                                    continue
                else:
                    # Hide main horizontal line
                    if validate_qt_pointer(chart.h_line1, "chart.h_line1"):
                        try:
                            if chart.h_line1.isVisible():
                                chart.h_line1.hide()
                        except RuntimeError as e:
                            test_log(f"RuntimeError hiding h_line1: {e}")

                    # Update appropriate energy plot horizontal line
                    if getattr(chart, 'crosshairs_p2', None):
                        for i, p in enumerate(chart.energy_plots):
                            if i >= len(chart.crosshairs_p2):
                                continue
                            _, h_line = chart.crosshairs_p2[i]
                            if validate_qt_pointer(h_line, f"crosshairs_p2[{i}].h_line") and validate_qt_pointer(p, f"energy_plots[{i}]"):
                                try:
                                    if p == active_plot:
                                        h_line.setPos(y)
                                        if not h_line.isVisible():
                                            h_line.show()
                                    else:
                                        if h_line.isVisible():
                                            h_line.hide()
                                except RuntimeError as e:
                                    test_log(f"RuntimeError updating crosshairs_p2[{i}].h_line: {e}")
                                    continue
            except Exception as e:
                test_log(f"Unexpected error in update_crosshair: {e}")
                if QT_DEBUG_MODE:
                    import traceback
                    test_log(f"Traceback: {traceback.format_exc()}")

    @staticmethod
    def auto_range(chart):
        try:
            if getattr(chart, '_closing', False):
                return
            if not hasattr(chart, 'store') or not chart.store: return
            d_ptr, g_idx = chart.store.get_sync_state()

            if g_idx == 0 or d_ptr == 0:
                return

            x_latest = float(g_idx)
            view_width = 1000.0
            x_min = max(0.0, x_latest - view_width)
            x_max = x_latest + view_width * 0.05

            if x_min >= x_max:
                x_max = x_min + 1.0

            if validate_qt_pointer(getattr(chart, 'p1', None), "chart.p1") and hasattr(chart.p1, 'vb') and validate_qt_pointer(chart.p1.vb, "chart.p1.vb"):
                safe_qt_operation("auto_range.setXRange", chart.p1.vb.setXRange, x_min, x_max, padding=0)

        except Exception as e:
            test_log(f"""Status: {e}""")
