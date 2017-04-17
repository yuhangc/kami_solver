import numpy as np
import cv2

import matplotlib.pyplot as plt

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D


class PuzzleSolver:
    def __init__(self, num_steps, num_colors=None):
        # original image and abstracted graph of the puzzle
        self.puzzle_id = None
        self.puzzle_img = None
        self.puzzle_graph = None

        # number of steps and colors total
        self.num_steps = num_steps
        self.num_colors = num_colors
        self.num_colors_left = num_colors

        # FIXME: hard-code in the separation in color panel
        self.color_panel_separate = 300

        # FIXME: hard-code in the minimum count for detection in saturation histogram
        self.min_count_sat = 5

        # FIXME: hard-code in the hue value width for color detection
        self.h_val_width = 1

        # FIXME: hard-code in the saturation limit for classification
        self.sat_limit = 50
        self.val_limit = 50

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

        # variables that store cell color and patch number
        self.cell_color = -np.ones((self.n_cells_h, self.n_cells_w), dtype=int)
        self.cell_patch = -np.ones((self.n_cells_h, self.n_cells_w), dtype=int)

        # increment in (h, w) for connectivity
        self.inc = np.array([[-1, 0],  [1, 0], [0, -1], [0, 1]])
        self.inc_mask = np.array([3, 2, 2, 3])

        # fast determination of cell type
        self.cell_lookup = np.array([[0, 1, 2, 3], [2, 3, 0, 1]])

        # fast lookup for cell vertex locations
        self.cell_vertices = None

        # number of patches and patch properties
        self.num_patches = 0
        self.patch_connect = None
        self.patch_adj = []
        self.patch_size = None
        self.patch_color = None

        # number of patches in certain color
        self.num_patches_color = None

    def load_puzzle(self, puzzle_id, puzzle_suf="", search_dir=None, down_sample=True, show=False):
        self.puzzle_id = puzzle_id

        if search_dir is None:
            puzzle_file = "../puzzles/{0}{1}.PNG".format(puzzle_id, puzzle_suf)
        else:
            puzzle_file = "{0}/{1}{2}.PNG".format(search_dir, puzzle_id, puzzle_suf)

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

        # classify the region
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hsv_mean = cv2.mean(roi, mask=mask)
        hsv_mean = hsv_mean[:3]

        # if saturation is low classify by saturation, otherwise classify by hue
        colors = np.array(self.colors)

        # if saturation and brightness is not too low classify using hue
        if hsv_mean[1] > self.sat_limit and hsv_mean[2] > self.sat_limit:
            hsv_diff = np.abs(colors - hsv_mean)
            hsv_diff = hsv_diff[:, 0]**2 + hsv_diff[:, 1]**2 + hsv_diff[:, 2]**2
            self.cell_color[xh, xw] = np.argmin(hsv_diff)
        else:
            # classify using saturation and value
            sv_diff = np.abs(colors[:, 1:] - hsv_mean[1:])
            sv_diff = sv_diff[:, 0]**2 + sv_diff[:, 1]**2
            self.cell_color[xh, xw] = np.argmin(sv_diff)

    def flood_fill(self, xh, xw, patch_number, patch_color):
        if xh < 0 or xh >= self.n_cells_h or xw < 0 or xw >= self.n_cells_w:
            return

        if self.cell_patch[xh, xw] != -1 or self.cell_color[xh, xw] != patch_color:
            return

        # assign patch number to current cell
        self.cell_patch[xh, xw] = patch_number

        # look for neighbors
        cell_type = self.cell_type(xh, xw)
        for dir_inc in range(4):
            if dir_inc == self.inc_mask[cell_type]:
                continue
            xh_new = xh + self.inc[dir_inc, 1]
            xw_new = xw + self.inc[dir_inc, 0]
            self.flood_fill(xh_new, xw_new, patch_number, patch_color)

    def draw_patches(self):
        visited = np.zeros((self.num_patches, ))
        img_labeled = deepcopy(self.puzzle_img)

        for xh in range(self.n_cells_h):
            for xw in range(self.n_cells_w):
                patch_id = self.cell_patch[xh, xw]
                if not visited[patch_id]:
                    cell_type = self.cell_type(xh, xw)
                    if cell_type == 0 or cell_type == 2:
                        xw_mid = int((xw / 2) * self.cell_width)
                    else:
                        xw_mid = int((xw / 2 + 0.5) * self.cell_width)
                    if cell_type == 0 or cell_type == 3:
                        xh_mid = int((xh + 0.5) * self.cell_height)
                    else:
                        xh_mid = int((xh + 1) * self.cell_height)
                    cv2.putText(img_labeled, "{0}".format(patch_id), (xw_mid, xh_mid),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                    visited[patch_id] = 1

        color_panel_height = 50
        img_color_panel = np.zeros((color_panel_height, img_labeled.shape[1], img_labeled.shape[2]), dtype=np.uint8)
        for k in range(self.num_colors):
            pt1 = (self.width * k / self.num_colors, 0)
            pt2 = (self.width * (k + 1) / self.num_colors, color_panel_height - 1)
            # color = (self.colors[k][0], self.colors[k][1], 127)
            cv2.rectangle(img_color_panel, pt1, pt2, self.colors[k], -1)

        img_color_panel = cv2.cvtColor(img_color_panel, cv2.COLOR_HSV2RGB)
        img_labeled = np.vstack((img_labeled, img_color_panel))
        cv2.imshow("patches", img_labeled)
        cv2.waitKey(0)

        # save to solution folder
        save_dir = "../solutions/{0}.png".format(self.puzzle_id)
        cv2.imwrite(save_dir, img_labeled)

    def count_and_connect_patches(self):
        self.patch_connect = np.zeros((self.num_patches, self.num_patches), dtype=int)
        self.patch_size = np.zeros((self.num_patches, ), dtype=int)
        self.patch_color = np.zeros((self.num_patches,), dtype=int)

        for xh in range(self.n_cells_h):
            for xw in range(self.n_cells_w):
                # update patch size and color
                self.patch_size[self.cell_patch[xh, xw]] += 1
                self.patch_color[self.cell_patch[xh, xw]] = self.cell_color[xh, xw]
                cell_type = self.cell_type(xh, xw)
                for dir_inc in range(4):
                    if dir_inc == self.inc_mask[cell_type]:
                        continue
                    xh_new = xh + self.inc[dir_inc, 1]
                    xw_new = xw + self.inc[dir_inc, 0]
                    if 0 <= xh_new < self.n_cells_h and 0 <= xw_new < self.n_cells_w:
                        # update patch connectivity
                        self.patch_connect[self.cell_patch[xh, xw],
                                           self.cell_patch[xh_new, xw_new]] = 1

        # set self connection to false
        for patch_id in range(self.num_patches):
            self.patch_connect[patch_id, patch_id] = 0

        # construct the adjacency list
        for patch_id in range(self.num_patches):
            neighbors = []
            for patch_next in range(self.num_patches):
                if self.patch_connect[patch_id, patch_next]:
                    neighbors.append(patch_next)
            self.patch_adj.append(neighbors)

        # plt.clf()
        # plt.imshow(self.patch_connect)
        # plt.show()

    def construct_puzzle_graph(self, show=False):
        # decompose the image into triangle cells
        for xh in range(self.n_cells_h):
            for xw in range(self.n_cells_w):
                self.classify_cell(xh, xw)

        plt.clf()
        plt.imshow(self.cell_color)
        plt.show()

        # find all patches
        self.num_patches_color = np.zeros((self.num_colors, ))

        for xh in range(self.n_cells_h):
            for xw in range(self.n_cells_w):
                if self.cell_patch[xh, xw] < 0:
                    self.flood_fill(xh, xw, self.num_patches, self.cell_color[xh, xw])
                    self.num_patches_color[self.cell_color[xh, xw]] += 1
                    self.num_patches += 1

        self.draw_patches()

        # count patch size and find connectivity
        self.count_and_connect_patches()

    def pre_process_colors(self, show=False):
        gray = cv2.cvtColor(self.puzzle_img, cv2.COLOR_RGB2GRAY)

        # find the horizontal line
        edges = cv2.Canny(gray, 30, 150, apertureSize=3)

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

        # record hsv values of the peaks
        for k in range(self.num_colors):
            # extract each color region
            lb = np.array([idx_plot[0, k] - 1, 0, 0])
            up = np.array([idx_plot[0, k] + 1, 255, 255])
            mask = cv2.inRange(color_panel, lb, up)

            # filter the mask to remove noises
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # cv2.imshow("image", mask)
            # cv2.waitKey(0)
            hsv_mean = cv2.mean(color_panel, mask=mask)
            print hsv_mean

            # record the means
            self.colors.append(hsv_mean[:3])

    def simple_solver(self, steps_left, connect, adj, sol, valid_patch, num_patches_color):
        # simple prunes
        num_colors_unconnected = 0
        num_colors_left = 0
        for n_patches in num_patches_color:
            if n_patches > 1:
                num_colors_unconnected += 1
            if n_patches > 0:
                num_colors_left += 1

        # no solution if number of unconnected colors are more than steps left
        if num_colors_left > steps_left + 1 or num_colors_unconnected > steps_left:
            return False

        # if have solution
        if num_colors_left == 1:
            return True

        # current step for labeling connectivity, starting from 2
        step_curr = self.num_steps - steps_left

        # try all patches and all colors
        for patch_id in range(self.num_patches):
            if not valid_patch[patch_id]:
                continue
            for color_id in range(self.num_colors):
                if num_patches_color[color_id] == 0 or color_id == self.patch_color[patch_id]:
                    continue
                # if step_curr < 4:
                #     print step_curr, patch_id, color_id
                # make current patch current color
                color_old = self.patch_color[patch_id]
                self.patch_color[patch_id] = color_id

                # update solution
                sol.append((patch_id, color_id))

                # update number of patches of
                # each color
                num_patches_color_new = deepcopy(num_patches_color)
                num_patches_color_new[color_old] -= 1
                num_patches_color_new[color_id] += 1

                # update connectivity
                adj_new = deepcopy(adj)
                valid_patch_new = deepcopy(valid_patch)
                for patch_next in adj[patch_id]:
                    if valid_patch[patch_next] and self.patch_color[patch_next] == color_id:
                        # update number of patches for each color
                        num_patches_color_new[color_id] -= 1
                        # invalidate the patch
                        valid_patch_new[patch_next] = 0

                        # connect all neighbors of patch_next to current patch
                        # temporarily invalidate current patch
                        valid_patch_new[patch_id] = 0
                        for patch_next_next in adj[patch_next]:
                            if valid_patch_new[patch_next_next] and adj_new[patch_id].count(patch_next_next) == 0:
                                adj_new[patch_id].append(patch_next_next)
                                adj_new[patch_next_next].remove(patch_next)
                                adj_new[patch_next_next].append(patch_id)
                        valid_patch_new[patch_id] = 1

                # only recurse down if adjacent patches have the same color
                if num_patches_color_new[color_id] <= num_patches_color[color_id]:
                    have_sol = self.simple_solver(steps_left - 1, connect, adj_new,
                                                  sol, valid_patch_new, num_patches_color_new)
                    if have_sol:
                        return True

                # restore everything
                self.patch_color[patch_id] = color_old
                sol.pop()

        # couldn't find solution if reach here
        return False

    def solve_puzzle(self, solver_id=0):
        self.sol = []
        if solver_id == 0:
            valid_patch = np.ones((self.num_patches, ), dtype=int)
            have_sol = self.simple_solver(self.num_steps, self.patch_connect, self.patch_adj,
                                          self.sol, valid_patch, self.num_patches_color)
            if not have_sol:
                raise Exception("Couldn't find solution!")

        print self.sol

    def visualize_solution(self):
        pass

    def visualize_graph(self):
        pass
