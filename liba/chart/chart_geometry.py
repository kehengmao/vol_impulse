import numpy as np
from ..kernels import EgiDataChannel
from .chart_core_utils import _is_test, test_log, check_extrema_num

class ChartGeometry:
    @staticmethod
    def debug_energy(stats_dict, batch_x, raw_extremas, extremas, full_extremas, max_solidity=1.0):
        if not _is_test:
            return

        stats_dict['total_raw_peaks'].update(batch_x[raw_extremas == 1])
        stats_dict['total_raw_valleys'].update(batch_x[raw_extremas == -1])

        stats_dict['total_super_peaks'].update(batch_x[extremas == 1])
        stats_dict['total_super_valleys'].update(batch_x[extremas == -1])

        stats_dict['total_full_peaks'].update(batch_x[full_extremas == 1])
        stats_dict['total_full_valleys'].update(batch_x[full_extremas == -1])

        overlap_peak_mask = (full_extremas == 1) & (raw_extremas == 1)
        overlap_valley_mask = (full_extremas == -1) & (raw_extremas == -1)
        stats_dict['total_overlap_peaks'].update(batch_x[overlap_peak_mask])
        stats_dict['total_overlap_valleys'].update(batch_x[overlap_valley_mask])

        raw_total = len(stats_dict['total_raw_peaks']) + len(stats_dict['total_raw_valleys'])
        super_total = len(stats_dict['total_super_peaks']) + len(stats_dict['total_super_valleys'])
        hist_total = len(stats_dict['total_full_peaks']) + len(stats_dict['total_full_valleys'])
        overlap_total = len(stats_dict['total_overlap_peaks']) + len(stats_dict['total_overlap_valleys'])

        if _is_test:
            test_log(
                f"""Status: {max_solidity} | {raw_total} | {len(stats_dict['total_raw_peaks'])} | {len(stats_dict['total_raw_valleys'])} | {super_total} | {len(stats_dict['total_super_peaks'])} | {len(stats_dict['total_super_valleys'])} | {hist_total} | {len(stats_dict['total_full_peaks'])} | {len(stats_dict['total_full_valleys'])} | {overlap_total} | {len(stats_dict['total_overlap_peaks'])} | {len(stats_dict['total_overlap_valleys'])}"""
                f"  -> Raw extrema: {raw_total} ({len(stats_dict['total_raw_peaks'])}, {len(stats_dict['total_raw_valleys'])})\n"
                f"  -> Active super extrema: {super_total} ({len(stats_dict['total_super_peaks'])}, {len(stats_dict['total_super_valleys'])})\n"
                f"  -> Full super-extrema history: {hist_total} ({len(stats_dict['total_full_peaks'])}, {len(stats_dict['total_full_valleys'])})\n"
                f"  -> Super/raw overlaps: {overlap_total} ({len(stats_dict['total_overlap_peaks'])}, {len(stats_dict['total_overlap_valleys'])})"
            )

    @staticmethod
    def extract_group_flags(chan_data, extremas_idx, full_extremas_idx, raw_extremas, egi_force_dir, flag_offset):
        extremas      = np.asarray(chan_data[:, extremas_idx], dtype=float)
        full_extremas = np.asarray(chan_data[:, full_extremas_idx], dtype=float)
        egi_dir       = np.asarray(chan_data[:, EgiDataChannel.egi_dir.value], dtype=float)

        is_extrema = np.abs(extremas) > 0.5
        is_semi_overlap = (np.abs(full_extremas) > 0.5) & (np.abs(raw_extremas) > 0.5) & (~is_extrema)
        valid_mask = is_extrema | is_semi_overlap

        is_solid = is_extrema
        is_tao = egi_force_dir > 0.5

        flag_arr = np.full(len(extremas), -1, dtype=np.int8)
        flag_arr[is_tao & is_solid] = 0 + flag_offset
        flag_arr[is_tao & ~is_solid] = 1 + flag_offset
        flag_arr[~is_tao & is_solid] = 2 + flag_offset
        flag_arr[~is_tao & ~is_solid] = 3 + flag_offset

        is_up_mask = egi_dir > 0.5
        is_dn_mask = egi_dir < -0.5

        return valid_mask, flag_arr, is_up_mask, is_dn_mask, extremas, full_extremas

    @staticmethod
    def triangulate_competitive(v_arr, flag_arr, egi_start, egi_end, egi_peak, abs_time_arr, is_up, rect_min_width, rect_pad_width):
        N = len(v_arr)
        out_val = np.zeros(N, dtype=np.float32)
        out_flag = np.full(N, -1, dtype=np.int8)
        out_is_peak = np.zeros(N, dtype=bool)
        out_is_rect = np.zeros(N, dtype=bool)

        valid_idx = np.where(v_arr != 0)[0]
        if len(valid_idx) == 0:
            return out_val, out_flag, out_is_peak

        if _is_test:
            test_log(f"""Status: {'UP' if is_up else 'DOWN'} | {len(valid_idx)}""")

        latest_abs_time = np.max(abs_time_arr)

        for idx in valid_idx:
            peak_val = v_arr[idx]
            peak_flag = flag_arr[idx]

            start_abs = egi_start[idx]
            end_abs = egi_end[idx]
            peak_abs = egi_peak[idx]

            if not (np.isfinite(start_abs) and np.isfinite(end_abs) and np.isfinite(peak_abs) and start_abs > 0 and end_abs > 0 and peak_abs > 0):
                if peak_val > out_val[idx]:
                    out_val[idx] = peak_val
                    out_flag[idx] = peak_flag
                    out_is_peak[idx] = True
                    out_is_rect[idx] = True
                continue

            try:
                local_start = int((N - 1) - (latest_abs_time - start_abs))
                local_end   = int((N - 1) - (latest_abs_time - end_abs))
                local_peak  = int((N - 1) - (latest_abs_time - peak_abs))

                if local_start > local_peak or local_end < local_peak:
                    if peak_val > out_val[idx]:
                        out_val[idx] = peak_val
                        out_flag[idx] = peak_flag
                        out_is_peak[idx] = True
                        out_is_rect[idx] = True
                    continue

            except (ValueError, OverflowError):
                if peak_val > out_val[idx]:
                    out_val[idx] = peak_val
                    out_flag[idx] = peak_flag
                    out_is_peak[idx] = True
                    out_is_rect[idx] = True
                continue

            if local_start > local_peak or local_end < local_peak:
                if peak_val > out_val[idx]:
                    out_val[idx] = peak_val
                    out_flag[idx] = peak_flag
                    out_is_peak[idx] = True
                    out_is_rect[idx] = True
                continue

            is_narrow = (local_end - local_start) < rect_min_width
            is_rect = False

            if is_narrow:
                local_start = local_end - rect_pad_width
                if local_start > local_peak:
                    local_start = local_peak - rect_pad_width
                is_rect = True

            draw_s = max(0, local_start)
            draw_e = min(N, local_end + 1)
            if draw_s >= draw_e:
                continue

            i_arr = np.arange(draw_s, draw_e)
            vals = np.zeros(draw_e - draw_s, dtype=np.float32)

            if is_rect:
                vals[:] = peak_val
            else:
                left_mask = i_arr <= local_peak
                if local_peak > local_start:
                    vals[left_mask] = peak_val * (i_arr[left_mask] - local_start) / (local_peak - local_start)
                elif np.any(left_mask):
                    vals[left_mask] = peak_val

                right_mask = i_arr > local_peak
                if local_end > local_peak:
                    vals[right_mask] = peak_val * (local_end - i_arr[right_mask]) / (local_end - local_peak)
                elif np.any(right_mask):
                    vals[right_mask] = peak_val

            align_len = min(len(vals), len(out_val[draw_s:draw_e]))
            vals = vals[:align_len]
            draw_e = draw_s + align_len

            if draw_s >= draw_e:
                continue

            win_mask = vals > out_val[draw_s:draw_e]

            if not is_rect:
                win_mask = win_mask | ((vals == out_val[draw_s:draw_e]) & out_is_rect[draw_s:draw_e])

            out_val[draw_s:draw_e][win_mask] = vals[win_mask]
            out_flag[draw_s:draw_e][win_mask] = peak_flag
            out_is_rect[draw_s:draw_e][win_mask] = is_rect

            out_is_peak[draw_s:draw_e][win_mask] = False
            peak_idx_in_slice = local_peak - draw_s
            if 0 <= peak_idx_in_slice < len(vals):
                if win_mask[peak_idx_in_slice]:
                    out_is_peak[local_peak] = True

        return out_val, out_flag, out_is_peak

    @staticmethod
    def process_combined_energy_channel(stats_dict, rect_min_width, rect_pad_width, chan_data, egi_start, egi_end, egi_peak, egi_value, egi_force_dir, abs_time_arr):
        raw_extremas = np.asarray(chan_data[:, EgiDataChannel.raw_extremas.value], dtype=float)

        valid1, flag1, up1, dn1, ex1, fex1 = ChartGeometry.extract_group_flags(
            chan_data, EgiDataChannel.extremas.value, EgiDataChannel.full_extremas.value,
            raw_extremas, egi_force_dir, flag_offset=0
        )

        valid2, flag2, up2, dn2, ex2, fex2 = ChartGeometry.extract_group_flags(
            chan_data, EgiDataChannel.extremas2.value, EgiDataChannel.full_extremas2.value,
            raw_extremas, egi_force_dir, flag_offset=8
        )

        idx1 = np.where(valid1)[0]
        for i in idx1:
            s1 = egi_start[i]
            e1 = egi_end[i]
            if not (s1 > 0 and e1 > 0 and s1 < e1):
                valid1[i] = False
                up1[i] = False
                dn1[i] = False

        idx1 = np.where(valid1)[0]
        s1_arr = egi_start[idx1]
        e1_arr = egi_end[idx1]

        valid_intervals1 = (s1_arr > 0) & (e1_arr > 0) & (s1_arr < e1_arr)
        s1_arr = s1_arr[valid_intervals1]
        e1_arr = e1_arr[valid_intervals1]

        idx2 = np.where(valid2)[0]
        for i in idx2:
            s2 = egi_start[i]
            e2 = egi_end[i]

            if s2 > 0 and e2 > 0 and s2 < e2:
                overlap = np.any((s1_arr <= e2) & (e1_arr >= s2))
                if overlap:
                    valid2[i] = False
                    up2[i] = False
                    dn2[i] = False
            else:
                valid2[i] = False
                up2[i] = False
                dn2[i] = False

        if _is_test:
            ChartGeometry.debug_energy(stats_dict, abs_time_arr, raw_extremas, ex1, fex1, max_solidity=1.0)
            ChartGeometry.debug_energy(stats_dict, abs_time_arr, raw_extremas, ex2, fex2, max_solidity=0.8)

        flag_arr = np.where(valid1, flag1, np.where(valid2, flag2, -1))

        valid_mask = valid1 | valid2
        clean_val = np.where(valid_mask, egi_value, 0.0)
        clean_val = np.clip(np.nan_to_num(clean_val), -1e7, 1e7)

        is_up_mask = up1 | up2
        is_dn_mask = dn1 | dn2

        val_up = np.where(is_up_mask, np.abs(clean_val), 0.0)
        val_dn = np.where(is_dn_mask, np.abs(clean_val), 0.0)

        out_val_up, out_flag_up, out_is_peak_up = ChartGeometry.triangulate_competitive(
            val_up, flag_arr, egi_start, egi_end, egi_peak, abs_time_arr, True, rect_min_width, rect_pad_width)
        out_val_dn, out_flag_dn, out_is_peak_dn = ChartGeometry.triangulate_competitive(
            val_dn, flag_arr, egi_start, egi_end, egi_peak, abs_time_arr, False, rect_min_width, rect_pad_width)

        return out_val_up, out_flag_up, out_val_dn, out_flag_dn

    @staticmethod
    def generate_triangles(stats_dict, num_channels, rect_min_width, rect_pad_width, energy_batch, batch_x):
        if energy_batch.ndim == 2:
            energy_batch = energy_batch[:, np.newaxis, :]

        triangles = []
        for c in range(num_channels):
            chan_data     = energy_batch[:, c, :]

            abs_time_arr  = np.asarray(chan_data[:, EgiDataChannel.abs_time.value], dtype=float)
            egi_value     = np.asarray(chan_data[:, EgiDataChannel.egi_value.value], dtype=float)
            egi_force_dir = np.asarray(chan_data[:, EgiDataChannel.egi_force_dir.value], dtype=float)
            egi_start     = np.asarray(chan_data[:, EgiDataChannel.egi_start_abs_time.value], dtype=float)
            egi_end       = np.asarray(chan_data[:, EgiDataChannel.egi_end_abs_time.value], dtype=float)
            egi_peak      = np.asarray(chan_data[:, EgiDataChannel.egi_peak_abs_time.value], dtype=float)

            if c == 0:
                extremas = np.asarray(chan_data[:, EgiDataChannel.extremas.value], dtype=float)
                check_extrema_num(abs_time_arr, extremas)

            val_up, flag_up, val_dn, flag_dn = ChartGeometry.process_combined_energy_channel(
                stats_dict, rect_min_width, rect_pad_width, chan_data, egi_start, egi_end, egi_peak, egi_value, egi_force_dir, abs_time_arr
            )

            triangles.append((val_up, flag_up, val_dn, flag_dn))

        return triangles
