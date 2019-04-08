from typing import Optional, List, Tuple, Dict
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.contour import QuadContourSet, ClabelText
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import warnings


def caldera_sim_function(x, y):
    warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
    x, y = x / 10.0, y / 10.0
    z0 = mlab.bivariate_normal(x, y, 10.0, 5.0, 5.0, 0.0)
    z1 = mlab.bivariate_normal(x, y, 1.0, 2.0, 2.0, 5.0)
    z2 = mlab.bivariate_normal(x, y, 1.7, 1.7, 8.0, 8.0)
    return 50000.0 * z0 + 2500.0 * z1 + 5000.0 * z2


def draw_caldera_maxima(axes: plt.Axes):
    maxima = list()
    maxima.extend(axes.plot(20, 46, 'bx'))
    maxima.extend(axes.plot(79, 79, 'bx'))
    return maxima


class RobotPlot:
    def __init__(self, ax, title):
        self.ax: plt.Axes = ax
        self.pos: Dict[int, Optional[Tuple[int, int]]] = dict()
        self.robot_marker: Dict[int, List[plt.Line2D]] = dict()
        self.path: Dict[int, List[List[plt.Line2D]]] = dict()
        self.ax.title.set_text(title)

    def draw_robot(self, new_pos, connect=True, index=0):
        for marker in self.robot_marker.get(index, list()):
            marker.remove()
        changed = list()
        if connect and self.pos.get(index, None) is not None:
            if index not in self.path:
                self.path[index] = list()
            self.path[index].append(self.ax.plot([self.pos[index][0], new_pos[0]],
                                                 [self.pos[index][1], new_pos[1]], color='k'))
            changed.extend(self.path[index][-1])
        self.robot_marker[index] = self.ax.plot(*new_pos, '*m')
        self.pos[index] = new_pos
        changed.extend(self.robot_marker[index])
        return changed


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
                self.cbar = None
        self.contours = self.ax.contour(X, Y, Z, **contour_kw)
        changed = [self.contours]
        if label:
            self.contour_labels = self.ax.clabel(self.contours, inline=1, fontsize=8, fmt='%.3g')
            changed.append(self.contour_labels)
        if colorbar and len(self.contour_labels) > 0:
            self.cbar = plt.gcf().colorbar(self.contours, ax=self.ax, fraction=0.046, pad=0.04)
            changed.append(self.cbar)
        return changed


class HeatmapPlot(RobotPlot):
    def __init__(self, ax, title):
        super().__init__(ax, title)
        self.im: Optional[AxesImage] = None
        self.cbar: Optional[Colorbar] = None

    def draw_heatmap(self, map, colorbar=True, **heatmap_kw):
        if self.cbar is not None:
            self.cbar.remove()
        self.im = self.ax.imshow(map, interpolation='nearest', **heatmap_kw)
        self.ax.invert_yaxis()
        changed = [self.im]
        if colorbar:
            self.cbar = plt.gcf().colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            # changed.append(self.cbar)
        return changed
