import numpy as np
import cv2

import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.signal import find_peaks_cwt


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

        # list storing the colors hsv value
        self.colors = []

        # solution
        self.sol = []

    def load_puzzle(self, puzzle_id, search_dir=None, down_sample=False, show=False):
        if search_dir is None:
            puzzle_file = "../puzzles/{0}.PNG".format(puzzle_id)
        else:
            puzzle_file = "{0}/{1}.PNG".format(search_dir, puzzle_id)

        self.puzzle_img = cv2.imread(puzzle_file)

        # optionally down_sample the image
        if down_sample:
            self.puzzle_img = cv2.resize(self.puzzle_img, None, fx=0.5, fy=0.5)
            self.color_panel_separate /= 2

        # optionally show the puzzle image
        if show:
            cv2.imshow("puzzle", self.puzzle_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # count and store colors
        self.pre_process_colors(show)

        # construct the puzzle graph
        self.construct_puzzle_graph(show)

    def construct_puzzle_graph(self, show=False):
        pass

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
        cv2.imshow("image", color_panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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

        # manually find the largest peak values
        hist1 = hist1.reshape((len(hist1,)))
        dhist = np.diff(hist1, 2)
        idx = np.where(dhist < -1000)
        idx_plot = np.asarray(idx) + 1
        idx = (idx_plot, )

        plt.clf()
        plt.plot(hist1)
        plt.plot(idx_plot, hist1[idx], 'rx')
        plt.show()

        if self.num_colors is None:
            self.num_colors = len(idx_plot[0])
        elif self.num_colors != len(idx_plot[0]):
            raise Exception("incorrect number of colors detected!")

        # record the H values of the peaks

    def solve_puzzle(self):
        pass

    def visualize_solution(self):
        pass

    def visualize_graph(self):
        pass
