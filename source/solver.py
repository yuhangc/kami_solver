import numpy as np
import cv2

import matplotlib.pyplot as plt

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D


class PuzzleSolver:
    def __init__(self, num_steps, num_colors=None):
        # original image and abstracted graph of the puzzle
        self.puzzle_img = None
        self.puzzle_graph = None

        # number of steps and colors total
        self.num_steps = num_steps
        self.num_colors = num_colors

        # FIXME: hard-code in the separation in color panel
        self.color_panel_separate = 300

        # FIXME: hard-code in the minimum count for detection in saturation histogram
        self.min_count_sat = 5

        # FIXME: hard-code in the hue value width for color detection
        self.h_val_width = 1

        # FIXME: hard-code in the saturation limit for classification
        self.sat_limit = 50

        # list storing the colors hsv value
        self.colors = []

        # solution
        self.sol = []

        # geometry quantities
        self.height = None
        self.width = None

        # puzzle image is decomposed into triangular cells
        self.n_cells_h = 28
        self.n_cells_w = 20
        self.cell_height = None
        self.cell_width = None
        self.cell_color = -np.ones((self.n_cells_h, self.n_cells_w))

        # increment in (h, w) for connectivity
        self.inc = np.array([[-1, 0],  [1, 0], [0, -1], [0, 1]])
        self.inc_mask = np.array([3, 2, 2, 3])

        # fast determination of cell type
        self.cell_lookup = np.array([[0, 1, 2, 3], [2, 3, 0, 1]])

        # fast lookup for cell vertex locations
        self.cell_vertices = None

    def load_puzzle(self, puzzle_id, search_dir=None, down_sample=True, show=False, enhance=True):
        if search_dir is None:
            puzzle_file = "../puzzles/{0}.PNG".format(puzzle_id)
        else:
            puzzle_file = "{0}/{1}.PNG".format(search_dir, puzzle_id)

        self.puzzle_img = cv2.imread(puzzle_file)

        # optionally down_sample the image
        if down_sample:
            self.puzzle_img = cv2.resize(self.puzzle_img, None, fx=0.5, fy=0.5)
            self.color_panel_separate /= 2

        # optionally increase contrast
        # if enhance:
        #     self.puzzle_img = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2HSV)
        #     self.puzzle_img[:, :, 1] = cv2.multiply(self.puzzle_img[:, :, 1], np.array([1.0]))
        #     self.puzzle_img = cv2.cvtColor(self.puzzle_img, cv2.COLOR_HSV2RGB)

        # optionally show the puzzle image
        if show:
            cv2.imshow("puzzle", self.puzzle_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # count and store colors
        self.pre_process_colors(show)

        # update geometry parameters
        self.height, self.width, _ = self.puzzle_img.shape
        self.cell_height = float(self.height) / self.n_cells_h
        self.cell_width = float(self.width * 2.0) / self.n_cells_w

        h, w = (int(self.cell_height), int(self.cell_width))
        tol = 5
        self.cell_vertices = np.array([[[[0, 0]], [[0, h-tol]], [[w-tol, 0]]],
                                       [[[tol, h]], [[w, tol]], [[w, h]]],
                                       [[[0, tol]], [[0, h]], [[w-tol, h]]],
                                       [[[tol, 0]], [[w, h-tol]], [[w, 0]]]])

        # construct the puzzle graph
        self.construct_puzzle_graph(show)

    def cell_type(self, xh, xw):
        xh &= 0x01
        xw &= 0x03
        return self.cell_lookup[xh, xw]

    def classify_cell(self, xh, xw):
        xw_low = int(xw / 2 * self.cell_width)
        xw_high = int((xw / 2 + 1) * self.cell_width)
        xh_low = int(xh * self.cell_height)
        xh_high = int((xh + 1) * self.cell_height)

        # only consider a local rectangular area
        puzzle_local = self.puzzle_img[xh_low:xh_high, xw_low:xw_high, :]

        # create a mask based on cell type
        cell_type = self.cell_type(xh, xw)

        mask = np.zeros(puzzle_local.shape[:2], np.uint8)

        gray = cv2.cvtColor(puzzle_local, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        _, cnt, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnt = self.cell_vertices[cell_type]
        cv2.drawContours(mask, [cnt], 0, 255, -1)

        # obtain the region of interest
        roi = cv2.bitwise_and(puzzle_local, puzzle_local, mask=mask)
        # cv2.imshow("image", roi)
        # cv2.waitKey(0)

        # classify the region
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hsv_mean = cv2.mean(roi, mask=mask)
        # hsv_mean = np.mean(roi, axis=(0, 1))

        # if saturation is low classify by saturation, otherwise classify by hue
        colors = np.array(self.colors)
        if hsv_mean[1] < self.sat_limit:
            s_diff = np.abs(colors[:, 1] - hsv_mean[1]) / colors[:, 3]
            self.cell_color[xh, xw] = np.argmin(s_diff)
        else:
            h_diff = np.abs(colors[:, 0] - hsv_mean[0]) / colors[:, 2]
            self.cell_color[xh, xw] = np.argmin(h_diff)

        # for k in range(self.num_colors):
        #     dist = (hsv_mean[:2] - self.colors[k][:2]) / self.colors[k][2:]
        #     if np.linalg.norm(dist) < dist_min:
        #         dist_min = np.linalg.norm(dist)
        #         self.cell_color[xh, xw] = k

    def construct_puzzle_graph(self, show=False):
        # decompose the image into triangle cells
        for xh in range(self.n_cells_h):
            for xw in range(self.n_cells_w):
                self.classify_cell(xh, xw)

        plt.clf()
        plt.imshow(self.cell_color)
        plt.show()

        # convert the puzzle image to hsv color space
        # puzzle_hsv = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2HSV)
        #
        # for k in range(self.num_colors):
        #     h_low, h_high, s_low, s_high = self.colors[k]
        #
        #     # find segments of each color
        #     range_low = np.array([h_low-1, s_low-10, 0])
        #     range_high = np.array([h_high+1, s_high+10, 255])
        #     mask = cv2.inRange(puzzle_hsv, range_low, range_high)
        #
        #     # display result
        #     if show:
        #         res = cv2.bitwise_and(self.puzzle_img, self.puzzle_img, mask=mask)
        #         cv2.imshow("image", res)
        #         cv2.waitKey(0)

    def pre_process_colors(self, show=False):
        gray = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2GRAY)
        # if show:
        #     cv2.imshow("image", gray)
        #     cv2.waitKey(0)

        # find the horizontal line
        edges = cv2.Canny(gray, 30, 150, apertureSize=3)
        # if show:
        #     cv2.imshow("image", edges)
        #     cv2.waitKey(0)

        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 100, min_theta=np.pi / 2.1, max_theta=np.pi / 1.9)

        # if show:
        #     img = deepcopy(self.puzzle_img)
        #     for line in lines:
        #         rho, theta = line[0]
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         x1 = int(x0 + 1000 * (-b))
        #         y1 = int(y0 + 1000 * a)
        #         x2 = int(x0 - 1000 * (-b))
        #         y2 = int(y0 - 1000 * a)
        #         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #
        #     cv2.imshow("image", img)
        #     cv2.waitKey(0)

        if len(lines) > 1:
            # TODO: find best rho when multiple lines detected
            bottom = int(lines[0, 0, 0])
        else:
            bottom = int(lines[0, 0, 0])

        # separate the color panel with the puzzle
        color_panel = deepcopy(self.puzzle_img[bottom + 2:, :, :])

        self.puzzle_img = self.puzzle_img[:bottom - 2, :, :]
        if show:
            cv2.imshow("image", self.puzzle_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # extract colors from the color panel
        color_panel = color_panel[:, self.color_panel_separate:, :]
        # cv2.imshow("image", color_panel)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # directly using the puzzle image to do clustering and count colors
        color_panel = cv2.cvtColor(color_panel, cv2.COLOR_RGB2HSV)
        hist1 = cv2.calcHist([color_panel], [0], None, [180], [0, 180])
        hist2 = cv2.calcHist([color_panel], [0, 1], None, [180, 256], [0, 180, 0, 256])

        # show 1d and 2d histogram for inspection
        # if show:
        #     plt.subplot(2, 1, 1)
        #     plt.plot(hist1)
        #     plt.subplot(2, 1, 2)
        #     plt.imshow(hist2, interpolation='nearest')
        #     plt.show()

        # manually find the largest peak values and indices
        hist1 = hist1.reshape((len(hist1,)))
        dhist = np.diff(hist1, 2)
        idx = np.where(dhist < -1000)
        idx_plot = np.asarray(idx) + 1
        idx = (idx_plot, )

        # plt.clf()
        # plt.plot(hist1)
        # plt.plot(idx_plot, hist1[idx], 'rx')
        # plt.show()

        if self.num_colors is None:
            self.num_colors = len(idx_plot[0])
        elif self.num_colors != len(idx_plot[0]):
            raise Exception("incorrect number of colors detected!")

        # record the HS values of the peaks
        for k in range(self.num_colors):
            h_val = idx_plot[0, k]
            hist_strip = hist2[h_val-1:h_val+1, :]
            idx_sat = np.where(hist_strip > self.min_count_sat)
            s_val_low = min(idx_sat[1])
            s_val_high = max(idx_sat[1])

            # record the color in format (hue_low, hue_high, sat_low, sat_high)
            # self.colors.append((h_val - self.h_val_width, h_val + self.h_val_width,
            #                     s_val_low, s_val_high))
            # record the color in format (h, s, h_var, s_var)
            self.colors.append(np.array([h_val, 0.5 * (s_val_low + s_val_high),
                                         1.0, 0.5 * (s_val_high - s_val_low)]))

    def solve_puzzle(self):
        pass

    def visualize_solution(self):
        pass

    def visualize_graph(self):
        pass
