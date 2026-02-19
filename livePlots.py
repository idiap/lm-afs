# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Damien Teney damien.teney@idiap.ch
# SPDX-License-Identifier: MIT

'''
Helper classes to plot losses, accuracy, activation magnitudes, and learned activation functions during training.
'''

import os
import time
import math
from abc import ABC, abstractmethod
import numpy as np
import matplotlib
if ("SLURM_JOB_ID" in os.environ): # Non-interactive job: make invisible plots that can be saved as images after training.
    BACKEND = "Agg"
else:
    BACKEND = "Qt5Agg" # or "TkAgg", "QtAgg"; "Qt5" is good on Windows
matplotlib.use(BACKEND)
import matplotlib.pyplot as plt

plt.rcParams.update({
    # ---------------- Fonts ----------------
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],   # bundled with matplotlib
    "mathtext.fontset": "dejavuserif",

    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # ---------------- Lines ----------------
    "lines.linewidth": 1.2,
    "lines.markersize": 4,

    # ---------------- Axes ----------------
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "axes.spines.top": True,
    "axes.spines.right": True,

    # ---------------- Ticks ----------------
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,

    # ---------------- Legend ----------------
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.3,

    # ---------------- Saving ----------------
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

class LivePlot(ABC): # Abstract class
    @abstractmethod
    def createFig(self): pass

    @abstractmethod
    def plot(self, x): pass # Arguments can differ

    def setTitle(self, newTitle):
        self.currentTitle = newTitle
        if BACKEND == "Agg": return # Non-interactive job
        self.hFig.canvas.manager.set_window_title(self.currentTitle)

    def saveFig(self, directory):
        if self.hFig is None: return None
        os.makedirs(directory, exist_ok=True)
        fileName = self.currentTitle # Current title of figure
        fileName = self.makeSafeFileName(fileName)
        fileName = os.path.join(directory, fileName + ".png") # Add path and extension
        self.hFig.savefig(fileName, dpi=150)
        return fileName

    def flushFig(self):
        if BACKEND == "Agg": return # Non-interactive job
        self.hFig.canvas.flush_events()

    def drawFig(self):
        if BACKEND == "Agg": return # Non-interactive job
        window = self.hFig.canvas.manager.window
        #if window.isMinimized(): window.showNormal() # Un-minimize window
        if not window.isMinimized():
            self.hFig.canvas.draw_idle()
            self.hFig.canvas.flush_events()

    def setFigureSize(self, widthPx=1520, heightPx=1100, maximize=False): # Set the figure window size in pixels; works on Windows, Linux, macOS, and headless Agg.
        fig = self.hFig
        if fig is None: return
        if "TkAgg" in BACKEND:
            window = fig.canvas.manager.window
            dpi = 100 # Larger DPI = smaller figure
            fig.set_size_inches(widthPx/dpi, heightPx/dpi, forward=True)
            if maximize: window.state("zoomed")
        elif "Qt" in BACKEND: # PyQt5, PyQt6, PySide2, PySide6
            window = fig.canvas.manager.window
            window.resize(int(widthPx), int(heightPx))
            window.move(int((window.screen().geometry().width() - widthPx) // 2), int((window.screen().geometry().height() - heightPx) // 2)) # Center the figure on the screen
            if maximize: window.showMaximized()
            for child in window.children(): # Hide native Qt toolbar
                if child.__class__.__name__ == "NavigationToolbar2QT": child.hide()
        else: # Fallback: use figsize in inches
            dpi = fig.dpi or 100
            fig.set_size_inches(widthPx / dpi, heightPx / dpi, forward=True)
        mgr = fig.canvas.manager
        if hasattr(mgr, "toolmanager"): mgr.toolmanager = None
        if hasattr(mgr, "toolbar"): mgr.toolbar_visible = False
        #plt.tight_layout()
        #padding = 0.02; fig.subplots_adjust(left=padding, right=(1-padding), top=(1-padding), bottom=padding, hspace=padding, wspace=padding) # Reduce margins/padding
        #fig.set_constrained_layout(False)

    @staticmethod
    def makeSafeFileName(s):
        remove = "\\/ : ,?!#@\"'"
        return "".join(c for c in str(s) if c not in remove)
    
    @staticmethod
    def figCloseCallback(event):
        fig = event.canvas.figure
        if not getattr(fig, "windowCloseRequested", False): # "False" is the default value if the attribute does not exist yet
            fig.windowCloseRequested = True
        fig.show() # Prevent closing the figure

class LivePlotLoss(LivePlot): # Figure with acc/loss curves that can be updated live during training
    def __init__(self, titleStr="tr", minAcc=0.0, maxAcc=1.0, minLoss=0.0, maxLoss=10.0, nModels=1, nStepsTotal=1000):
        self.titleStr = str(titleStr)
        self.minAcc = float(minAcc)
        self.maxAcc = float(maxAcc)
        self.minLoss = float(minLoss)
        self.maxLoss = float(maxLoss)
        self.nModels = int(nModels)
        self.nStepsTotal = int(nStepsTotal)
        self.hFig = None
        self.hAxes = []
        self.hLine = None # shape: [2][3][nModels]
        self.currentTitle = ""
        self.createFig()

    def shouldStopTraining(self):
        if self.hFig is None: return False
        return getattr(self.hFig, "windowCloseRequested", False)

    def createFig(self):
        plt.ion()
        self.hFig, axes = plt.subplots(2, 1, sharex=True, constrained_layout=True)
        self.hAxes = axes[::-1] # Flip: acc on top, loss on bottom
        self.setTitle(self.titleStr)

        # Setup window-closing callback
        self.hFig.windowCloseRequested = False
        self.hFig.canvas.mpl_connect("close_event", self.figCloseCallback)

        self.hAxes[1].set_ylabel("Acc")
        self.hAxes[0].set_ylabel("Loss")
        self.hAxes[0].set_xlabel("Training step")
        for ax in self.hAxes:
            ax.set_xlim(1, max(2, self.nStepsTotal))

        # Create lines: 2 (loss/acc) x 3 (tr/va/te) x nModels
        colors = [
            (0.6, 0.6, 0.6, 0.6),  # Tr: gray
            (0.0, 0.0, 1.0, 0.2),  # Va: blue
            (1.0, 0.0, 0.0, 0.2)   # Te: red
        ]
        widths = [1.0, 1.5, 1.5]
        self.hLine = [[[None for _ in range(self.nModels)] for _ in range(3)] for _ in range(2)]
        for i in range(2):
            for j in range(3):
                for m in range(self.nModels):
                    (line,) = self.hAxes[i].plot([], [], linestyle="-", color=colors[j], linewidth=widths[j])
                    self.hLine[i][j][m] = line

        tickSpacing = (self.maxAcc - self.minAcc) / 10
        self.hAxes[1].set_yticks(np.arange(self.minAcc, self.maxAcc + tickSpacing, tickSpacing)) # Acc Y axis
        self.hAxes[1].set_ylim((self.minAcc, self.maxAcc)) # Acc Y axis
        self.hAxes[0].set_ylim(self.minLoss, self.maxLoss) # Loss Y axis

        self.setFigureSize(widthPx=1520, heightPx=1100, maximize=False)
        #drawFig(self.hFig)

    def plot(self, step, lossTr=None, lossVa=None, lossTe=None, 
                         accTr=None, accVa=None, accTe=None,
                         timeElapsed1=None, timeElapsed2=None):
        if self.hFig is None: return
        inputs = [[lossTr, lossVa, lossTe], [accTr, accVa, accTe]]
        for i in range(2):
            for j in range(3):
                v = inputs[i][j]
                if v is None:
                    continue
                elif isinstance(v, (list, tuple)):
                    v = np.array(v, dtype=float)
                elif np.isscalar(v):
                    v = np.array([v], dtype=float)
                else:
                    v = np.array(v, dtype=float)
                assert v.size <= self.nModels, f"Wrong number of elements in input [{i},{j}]: v.size={v.size}, nModels={self.nModels}"
                for k in range(v.size):
                    if not np.isnan(v[k]):
                        line = self.hLine[i][j][k]
                        xdata = np.append(line.get_xdata(), step)
                        ydata = np.append(line.get_ydata(), v[k])
                        line.set_data(xdata, ydata)

        # Update title
        if step is not None:
            title = f"{self.titleStr}-step{step:06d}"
        for t in [timeElapsed1, timeElapsed2]:
            if t is not None: # Converts seconds to dd:hh:mm:ss and append to title
                title += f"-{int(t)//86400:02d}:{(int(t)%86400)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}"
        self.setTitle(title)

        #drawFig(self.hFig)

class LivePlotAct(LivePlot): # Figure with plots of activation magnitudes that can be updated live during training.
    def __init__(self, titleStr="act", yLabel="Activation magnitude", minVal=0.0, maxVal=np.inf, nCurves=1, nGroups=1, nStepsTotal=1000):
        self.titleStr = str(titleStr)
        self.yLabel = str(yLabel)
        self.minVal = float(minVal)
        self.maxVal = float(maxVal)
        self.nCurves = int(nCurves)
        self.nGroups = int(nGroups)
        self.nStepsTotal = int(nStepsTotal)
        self.hFig = None
        self.hAxes = None
        self.hLines = []
        self.currentTitle = ""
        self.createFig()
    
    def createFig(self):
        plt.ion()
        self.hFig, self.hAxes = plt.subplots(1, 1, sharex=True, constrained_layout=True)
        self.hAxes.set_xlabel("Training step")
        self.hAxes.set_ylabel(self.yLabel)
        self.hAxes.set_xlim(1, max(2, self.nStepsTotal))
        self.hLines = []

        # Colors: first group gray, rest HSV
        colors = [(.4, .4, .4)] + [plt.cm.hsv(i/(self.nGroups-1))[:3] for i in range(1, self.nGroups)]
        lineStyles = ["-"] * self.nGroups
        
        lineId = 0
        for groupId in range(self.nGroups):
            for c in range(self.nCurves):
                if c == 0:
                    lw, alpha = 2.0, 0.3
                elif c < self.nCurves - 1:
                    lw, alpha = 1.0, 0.3
                else:
                    lw, alpha = 1.5, 0.3
                line, = self.hAxes.plot([], [], linestyle=lineStyles[groupId], color=colors[groupId] + (alpha,), linewidth=lw)
                self.hLines.append(line)
                lineId += 1

        self.hAxes.set_ylim(self.minVal, self.maxVal)
        self.setTitle(self.titleStr)
        self.setFigureSize(widthPx=1520, heightPx=1100/2, maximize=False)
        #drawFig(self.hFig)

    def plot(self, groupId, step, vals): # Update the lines for a given group at a training step
        vals = np.array(vals, dtype=float)
        assert vals.size == self.nCurves
        for c in range(self.nCurves):
            lineId = c + (groupId - 1) * self.nCurves
            if not np.isnan(vals[c]):
                line = self.hLines[lineId]
                xdata = np.append(line.get_xdata(), step)
                ydata = np.append(line.get_ydata(), vals[c])
                line.set_data(xdata, ydata)
        if np.any(vals > self.maxVal):
            self.hAxes.set_ylim(self.minVal, np.nanmax([self.maxVal, np.max(vals)]))
        self.setTitle(f"{self.titleStr}-step{step:06d}")
        #drawFig(self.hFig)

class LivePlotAf(LivePlot): # Plot of spline activation functions that can be updated live during training.
    def __init__(self, titleStr="af", nTiles=1, afRange=1.0, afNAnchors=10):
        self.titleStr = str(titleStr)
        self.nTiles = int(nTiles)
        assert nTiles > 0, f"nTiles={nTiles}"
        self.afRange = float(afRange)
        self.afAnchors = np.linspace(-self.afRange, self.afRange, afNAnchors)
        self.hFig = None
        self.hAxes = []
        self.hLine = []
        self.lineColor = (0.1, 0.1, 1.0, 0.5)
        self.lineWidth = 1.5
        self.lineMarker = None # Could use '.' for marker
        self.currentTitle = ""
        self.createFig()
    
    def createFig(self):
        plt.ion()
        self.hFig, axes = plt.subplots(1, self.nTiles, constrained_layout=True)
        if (self.nTiles == 1): axes = [axes]
        self.hAxes = axes
        self.hLine = []
        for ax in self.hAxes:
            ax.set_box_aspect(1) # Square
            line, = ax.plot([], [], linestyle='-', color=self.lineColor, linewidth=self.lineWidth, marker=self.lineMarker)
            self.hLine.append(line)
        self.setTitle(self.titleStr)
        self.setFigureSize(widthPx=180*self.nTiles, heightPx=180, maximize=False)
        #drawFig(self.hFig)
    
    def addLine(self): # Add a new line to each tile
        for i, ax in enumerate(self.hAxes):
            self.hLine[i] = ax.plot([], [], linestyle='-', color=self.lineColor, linewidth=self.lineWidth, marker=self.lineMarker)
    
    def plot(self, step, xs, ysAll): # Update the lines with new activation function values
        if self.hFig is None: return
        xs = np.array(xs, dtype=float)
        for i in range(self.nTiles):
            ys = np.array(ysAll[i], dtype=float)
            assert xs.shape == ys.shape, f"xs.shape={xs.shape}, ys.shape={ys.shape}"
            self.hLine[i].set_data(xs, ys)
            xLim = np.max(np.abs(xs)); self.hAxes[i].set_xlim(-xLim, xLim) # Adjust X-axis
            yLim = np.max(np.abs(ys)); self.hAxes[i].set_ylim(-yLim, yLim) # Adjust Y-axis
            #self.hAxes[i].autoscale(enable=True, axis='y') # Adjust Y-axis
        self.setTitle(f"{self.titleStr}-step{step:06d}")
        #drawFig(self.hFig)
