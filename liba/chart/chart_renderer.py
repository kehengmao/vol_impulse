import numpy as np
from .chart_core_utils import _is_test, test_log, time_it, QtDebugContext, validate_qt_pointer, safe_qt_operation, QT_DEBUG_MODE

class ChartRenderer:
    @staticmethod
    def calculate_safe_bounds(arrays_list, pad_ratio=0.05):
        if not arrays_list:
            return None

        try:
            clean_arrays = [a for a in arrays_list if getattr(a, 'size', 0) > 0]
            if not clean_arrays:
                return None
            all_data = np.concatenate(clean_arrays)
        except Exception:
            return None

        valid_data = all_data[np.isfinite(all_data)]
        if valid_data.size == 0:
            return None

        try:
            c_min, c_max = float(valid_data.min()), float(valid_data.max())
            c_min = float(np.clip(c_min, -1e7, 1e7))
            c_max = float(np.clip(c_max, -1e7, 1e7))

            padding = float(np.clip((c_max - c_min) * pad_ratio, 0.1, 1e8))

            if c_min >= c_max:
                c_min -= 1.0
                c_max += 1.0

            return c_min - padding, c_max + padding
        except Exception:
            return None

    @staticmethod
    def update_main_plot_bounds(chart, v_min, v_max):
        with QtDebugContext("update_main_plot_bounds", [(getattr(chart, 'p1', None), "chart.p1")]):
            visible_p1, _ = chart._get_visible_p1_data(v_min, v_max)
            bounds = ChartRenderer.calculate_safe_bounds(visible_p1)
            if bounds and validate_qt_pointer(getattr(chart, 'p1', None), "chart.p1"):
                if hasattr(chart.p1, 'vb') and validate_qt_pointer(chart.p1.vb, "chart.p1.vb"):
                    safe_qt_operation("setRange", chart.p1.vb.setRange, yRange=list(bounds), padding=0)

    @staticmethod
    def update_energy_plots_bounds(chart, v_min, v_max):
        if getattr(chart, 'num_channels', None) is None:
            return
        if not getattr(chart, 'energy_plots', None) or len(chart.energy_plots) != chart.num_channels:
            return

        _, (s_d, e_d) = chart._get_visible_p1_data(v_min, v_max)

        all_visible_ec = []
        current_avg_copy = chart.store.get_current_avg() if hasattr(chart, 'store') and chart.store else None
        d_ptr = chart.store.get_sync_state()[0] if hasattr(chart, 'store') and chart.store else 0

        for c in range(chart.num_channels):
            if d_ptr > 0 and e_d > s_d:
                e_val = chart.data_e_val
                if e_val is not None:
                    sliced_e = chart._get_logical_slice(e_val[c], s_d, e_d)
                    all_visible_ec.append(sliced_e[0].ravel())
                    all_visible_ec.append(-sliced_e[1].ravel())

            if current_avg_copy is not None:
                v_up = current_avg_copy[c, 0]
                v_dn = current_avg_copy[c, 1]
                if np.isfinite(v_up): all_visible_ec.append(np.array([v_up]))
                if np.isfinite(v_dn): all_visible_ec.append(np.array([-np.abs(v_dn)]))

        bounds = ChartRenderer.calculate_safe_bounds(all_visible_ec)
        if bounds is None:
            bounds = (-1e-9, 1e-9)

        for c in range(chart.num_channels):
            p = chart.energy_plots[c]
            if not validate_qt_pointer(p, f"energy_plots[{c}]"):
                continue
            if not hasattr(p, 'vb') or not validate_qt_pointer(p.vb, f"energy_plots[{c}].vb"):
                continue
            safe_qt_operation(f"energy_plot[{c}].setRange", p.vb.setRange, yRange=list(bounds), padding=0)

    @staticmethod
    @time_it
    def update_y_range(chart, v_min, v_max):
        if getattr(chart, 'data_ptr', 0) == 0 and not getattr(chart, 'has_baked', False):
            return

        if _is_test:
            test_log(f"""Status: {v_min} | {v_max}""")

        try:
            ChartRenderer.update_main_plot_bounds(chart, v_min, v_max)
            ChartRenderer.update_energy_plots_bounds(chart, v_min, v_max)
        except Exception as e:
            test_log(f"CRITICAL ERROR in _update_y_range: {e}")
            import traceback
            test_log(traceback.format_exc())

    @staticmethod
    def sync_avg_lines_to_gui(chart):
        if getattr(chart, 'num_channels', None) is None:
            return
        current_avg_copy = chart.store.get_current_avg() if hasattr(chart, 'store') and chart.store else None
        if current_avg_copy is None:
            return
        if not getattr(chart, 'avg_lines', None) or len(chart.avg_lines) != chart.num_channels:
            return

        with QtDebugContext("sync_avg_lines_to_gui"):
            try:
                for c in range(chart.num_channels):
                    v_up = current_avg_copy[c, 0]
                    v_dn = current_avg_copy[c, 1]

                    line_up, line_down = chart.avg_lines[c]

                    # Validate line objects
                    if not validate_qt_pointer(line_up, f"avg_lines[{c}].up"):
                        continue
                    if not validate_qt_pointer(line_down, f"avg_lines[{c}].down"):
                        continue

                    if np.isfinite(v_up):
                        safe_qt_operation(f"setPos avg_up[{c}]", line_up.setPos, float(v_up))
                        safe_qt_operation(f"setFormat avg_up[{c}]", line_up.label.setFormat, f"{float(v_up):.2f}")
                        safe_qt_operation(f"show avg_up[{c}]", lambda: line_up.show() if not line_up.isVisible() else None)
                    else:
                        safe_qt_operation(f"hide avg_up[{c}]", lambda: line_up.hide() if line_up.isVisible() else None)

                    if np.isfinite(v_dn):
                        val_y = float(-np.abs(v_dn))
                        safe_qt_operation(f"setPos avg_down[{c}]", line_down.setPos, val_y)
                        safe_qt_operation(f"setFormat avg_down[{c}]", line_down.label.setFormat, f"-{float(np.abs(v_dn)):.2f}")
                        safe_qt_operation(f"show avg_down[{c}]", lambda: line_down.show() if not line_down.isVisible() else None)
                    else:
                        safe_qt_operation(f"hide avg_down[{c}]", lambda: line_down.hide() if line_down.isVisible() else None)
            except Exception as e:
                test_log(f"Error syncing avg lines to GUI: {e}")
                if QT_DEBUG_MODE:
                    import traceback
                    test_log(f"Traceback: {traceback.format_exc()}")
