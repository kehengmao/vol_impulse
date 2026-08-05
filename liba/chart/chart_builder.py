import pyqtgraph as pg
from PySide6.QtCore import Qt
from .chart_components import SafeAxisItem, SimplifiedTimeAxis

class ChartBuilder:
    @staticmethod
    def init_main_plot(chart):
        chart.ci.layout.setSpacing(0)
        chart.ci.setContentsMargins(0, 0, 0, 0)

        axis_items = {
            'left': SafeAxisItem(orientation='left'),
            'bottom': SafeAxisItem(orientation='bottom')
        }
        chart.p1 = chart.addPlot(row=0, col=0, title="Tick Price (Bid / Ask)", axisItems=axis_items)
        chart.p1.showGrid(x=False, y=True, alpha=0.3)
        chart.p1.setMouseEnabled(x=True, y=True)

        bottom_axis_p1 = chart.p1.getAxis('bottom')
        bottom_axis_p1.setStyle(showValues=False, tickLength=0)
        bottom_axis_p1.setPen(pg.mkPen(color=(150, 150, 150, 255), width=1, style=Qt.PenStyle.SolidLine))
        bottom_axis_p1.show()

        chart.p1.vb.setLimits(minXRange=5.0, minYRange=1e-5)
        chart.p1.vb.disableAutoRange()

        chart.v_line1 = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine),
            label='--:--:--', labelOpts={'position': 0.95, 'color': (200, 200, 200), 'fill': (50, 50, 50, 200)}
        )
        chart.h_line1 = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine),
            label='{value:0.2f}', labelOpts={'position': 0.95, 'color': (200, 200, 200), 'fill': (50, 50, 50, 200)}
        )
        chart.p1.addItem(chart.v_line1, ignoreBounds=True)
        chart.p1.addItem(chart.h_line1, ignoreBounds=True)

        opts = {
            'autoDownsample': False,
            'clipToView': False,
            'skipFiniteCheck': True
        }

        chart.curve_bid_dyn = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 51, 51, 255), width=1.5), connect='finite', **opts)
        chart.curve_ask_dyn = pg.PlotCurveItem(pen=pg.mkPen(color=(51, 255, 51, 255), width=1.5), connect='finite', **opts)
        chart.curve_ma_dyn = pg.PlotCurveItem(pen=pg.mkPen('w', width=1.5), connect='finite', **opts)

        chart.p1.addItem(chart.curve_bid_dyn)
        chart.p1.addItem(chart.curve_ask_dyn)
        chart.p1.addItem(chart.curve_ma_dyn)

    @staticmethod
    def cleanup_energy_plots(chart):
        if hasattr(chart, 'crosshairs_p2') and chart.crosshairs_p2:
            for v, h in chart.crosshairs_p2:
                try: v.hide(); v.setParentItem(None); v.deleteLater()
                except: pass
                try: h.hide(); h.setParentItem(None); h.deleteLater()
                except: pass
            chart.crosshairs_p2.clear()

        if hasattr(chart, 'avg_lines') and chart.avg_lines:
            for u, d in chart.avg_lines:
                try: u.hide(); u.setParentItem(None); u.deleteLater()
                except: pass
                try: d.hide(); d.setParentItem(None); d.deleteLater()
                except: pass
            chart.avg_lines.clear()

        if hasattr(chart, 'energy_curves_dyn') and chart.energy_curves_dyn:
            for curves in chart.energy_curves_dyn:
                for c in curves:
                    try: c.hide(); c.setParentItem(None); c.deleteLater()
                    except: pass
            chart.energy_curves_dyn.clear()

        if hasattr(chart, 'energy_plots') and chart.energy_plots:
            for p in chart.energy_plots:
                try: p.setXLink(None)
                except Exception: pass
                try: chart.p1.vb.sigStateChanged.disconnect(p.vb.linkedViewChanged)
                except Exception: pass
                try: chart.p1.vb.sigResized.disconnect(p.vb.linkedViewChanged)
                except Exception: pass
                try: p.vb.sigStateChanged.disconnect(chart.p1.vb.linkedViewChanged)
                except Exception: pass
                try: p.vb.sigResized.disconnect(chart.p1.vb.linkedViewChanged)
                except Exception: pass

                chart.ci.removeItem(p)
                try: p.hide(); p.setParentItem(None)
                except Exception: pass
                try:
                    p.deleteLater()
                except Exception:
                    pass

        if hasattr(chart, 'legend') and chart.legend:
            try: chart.legend.hide(); chart.legend.scene().removeItem(chart.legend)
            except Exception: pass
            try: chart.legend.setParentItem(None)
            except Exception: pass
            chart.legend.deleteLater()
            chart.legend = None

        if hasattr(chart, 'time_axis') and chart.time_axis:
            try: chart.time_axis.hide(); chart.time_axis.setParentItem(None); chart.time_axis.deleteLater()
            except Exception: pass

        chart.energy_plots = []
        chart.energy_curves_static = []
        chart.energy_curves_dyn = []
        chart.crosshairs_p2 = []
        chart.avg_lines = []

    @staticmethod
    def build_energy_plots(chart, num_channels):
        ChartBuilder.cleanup_energy_plots(chart)

        chart.time_axis = SimplifiedTimeAxis(chart_ref=chart, orientation='bottom')

        opts = {
            'autoDownsample': False,
            'clipToView': False,
            'skipFiniteCheck': True
        }

        brushes = [
            pg.mkBrush((255, 51, 51, 255)),   # 0
            pg.mkBrush((255, 51, 51, 102)),   # 1
            pg.mkBrush((255, 165, 0, 255)),   # 2
            pg.mkBrush((255, 165, 0, 102)),   # 3
            pg.mkBrush((51, 255, 51, 255)),   # 4
            pg.mkBrush((51, 255, 51, 102)),   # 5
            pg.mkBrush((0, 255, 255, 255)),   # 6
            pg.mkBrush((0, 255, 255, 102))    # 7
        ]

        pens = [
            pg.mkPen(color=(255, 51, 51, 255), width=2.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(255, 51, 51, 102), width=1.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(255, 165, 0, 255), width=2.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(255, 165, 0, 102), width=1.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(51, 255, 51, 255), width=2.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(51, 255, 51, 102), width=1.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(0, 255, 255, 255), width=2.0, style=Qt.PenStyle.SolidLine),
            pg.mkPen(color=(0, 255, 255, 102), width=1.0, style=Qt.PenStyle.SolidLine)
        ]

        for i in range(8):
            b = brushes[i]
            alpha = b.color().alpha()
            brushes.append(pg.mkBrush((150, 150, 150, alpha)))
            p = pens[i]
            pens.append(pg.mkPen(color=(150, 150, 150, alpha), width=p.width(), style=p.style()))

        chart.ci.layout.setRowStretchFactor(0, num_channels)

        for i in range(num_channels):
            axis = {'left': SafeAxisItem(orientation='left')}
            if i == num_channels - 1:
                axis['bottom'] = chart.time_axis
            else:
                axis['bottom'] = SafeAxisItem(orientation='bottom')

            p = chart.addPlot(row=i+1, col=0, axisItems=axis)
            chart.ci.layout.setRowStretchFactor(i+1, 1)

            bottom_axis = p.getAxis('bottom')
            if i < num_channels - 1:
                bottom_axis.setStyle(showValues=False, tickLength=0)
            bottom_axis.setPen(pg.mkPen(color=(150, 150, 150, 255), width=1, style=Qt.PenStyle.SolidLine))
            bottom_axis.show()

            p.showGrid(x=False, y=True, alpha=0.3)
            p.setXLink(chart.p1)
            p.setMouseEnabled(x=True, y=False)

            p.vb.setLimits(minXRange=5.0)
            p.vb.disableAutoRange()

            zero_line = pg.InfiniteLine(angle=0, pos=0, movable=False, pen=pg.mkPen('w', width=1.0, style=Qt.PenStyle.SolidLine))
            p.addItem(zero_line, ignoreBounds=True)

            c_dyn = []
            for j in range(16):
                c_d = pg.PlotCurveItem(pen=pens[j], fillLevel=0, brush=brushes[j], **opts)
                p.addItem(c_d)
                c_dyn.append(c_d)
            chart.energy_curves_dyn.append(c_dyn)

            chart.energy_plots.append(p)

            if i == num_channels - 1:
                chart.legend = pg.LegendItem(brush=(30, 30, 30, 200), pen=(100, 100, 100))
                chart.legend.setParentItem(chart.p1.vb)
                chart.legend.anchor((0, 1), (0, 1), offset=(10, -10))
                chart.legend.setZValue(99999)

                chart.legend.addItem(c_dyn[0], 'Upper envelope')
                chart.legend.addItem(c_dyn[2], 'Upper guide')
                chart.legend.addItem(c_dyn[4], 'Lower envelope')
                chart.legend.addItem(c_dyn[6], 'Lower guide')

            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine),
                                     label='--:--:--', labelOpts={'position': 0.05, 'color': (200,200,200), 'fill': (50,50,50,200)})
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', style=Qt.PenStyle.DashLine),
                                     label='{value:0.2f}', labelOpts={'position': 0.95, 'color': (200,200,200), 'fill': (50,50,50,200)})
            p.addItem(v_line, ignoreBounds=True)
            p.addItem(h_line, ignoreBounds=True)
            v_line.hide(); h_line.hide()
            chart.crosshairs_p2.append((v_line, h_line))

            line_up = pg.InfiniteLine(
                angle=0, movable=False,
                pen=pg.mkPen(color=(255, 51, 51, 200), width=1.5, style=Qt.PenStyle.SolidLine),
                label='{value:0.2f}',
                labelOpts={'position': 0.95, 'color': (255, 51, 51), 'fill': (30, 30, 30, 200)}
            )
            line_down = pg.InfiniteLine(
                angle=0, movable=False,
                pen=pg.mkPen(color=(51, 255, 51, 200), width=1.5, style=Qt.PenStyle.SolidLine),
                label='{value:0.2f}',
                labelOpts={'position': 0.95, 'color': (51, 255, 51), 'fill': (30, 30, 30, 200)}
            )
            line_up.setZValue(99999)
            line_down.setZValue(99999)
            line_up.hide(); line_down.hide()
            p.addItem(line_up); p.addItem(line_down)
            chart.avg_lines.append((line_up, line_down))
