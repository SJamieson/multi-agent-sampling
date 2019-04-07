from typing import Optional, List, Tuple
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.contour import QuadContourSet, ClabelText
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt


def draw_caldera_maxima(axes: plt.Axes):
    maxima = list()
    maxima.extend(axes.plot(20, 46, 'bx'))
    maxima.extend(axes.plot(79, 79, 'bx'))
    return maxima


class RobotPlot:
    def __init__(self, ax, title):
        self.ax: plt.Axes = ax
        self.pos: Optional[Tuple[int, int]] = None
        self.robot_marker: List[plt.Line2D] = list()
        self.path: List[List[plt.Line2D]] = list()
        self.ax.title.set_text(title)

    def draw_robot(self, new_pos, connect=True):
        for marker in self.robot_marker:
            marker.remove()
        if connect and self.pos is not None:
            self.path.append(ax1.plot([pos[0], new_pos[0]], [pos[1], new_pos[1]], color='k'))
        self.pos = new_pos
        self.robot_marker = self.ax.plot(*new_pos, '*m')
        return self.path[-1], self.robot_marker


class ContourPlot(RobotPlot):
    def __init__(self, ax, title):
        super().__init__(ax, title)
        self.contours: Optional[QuadContourSet] = None
        self.contour_labels: List[ClabelText] = list()
        self.cbar: Optional[Colorbar] = None

    def draw_contours(self, X, Y, Z, label=True, colorbar=False, **contour_kw):
        if self.contours is not None:
            for coll in self.contours.collections:
                coll.remove()
            for label in self.contour_labels:
                label.remove()
            if self.cbar is not None:
                self.cbar.remove()
        self.contours = self.ax.contour(X, Y, Z, **contour_kw)
        changed = [self.contours]
        if label:
            self.contour_labels = self.ax.clabel(self.contours, inline=1, fontsize=8, fmt='%.3g')
            changed.append(self.contour_labels)
        if colorbar:
            self.cbar = plt.gcf().colorbar(self.contours, ax=self.ax, fraction=0.046, pad=0.04)
            changed.append(self.cbar)
        return changed


class HeatmapPlot(RobotPlot):
    def __init__(self, ax, title):
        super().__init__(ax, title)
        self.im: Optional[AxesImage] = None
        self.cbar: Optional[Colorbar] = None

    def draw_heatmap(self, map, label=True, colorbar=False, **heatmap_kw):
        if self.cbar is not None:
            self.cbar.remove()
        self.im = self.ax.imshow(map, **heatmap_kw)
        changed = self.im
        if colorbar:
            self.cbar = plt.gcf().colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            changed.append(self.cbar)
        return changed
