import numpy as np
from fo.fo_factory import FOFactory
from fo.ue import UE
from fo.lh import LH
from queue import Queue


class HH(object):
    def __init__(self):
        self.D = 0
        self.sample_size = 0
        self.branch = 0
        self.l = 0
        # use array to store a B-adic tree
        self.tree = np.zeros(0)
        # build node->range mapping for each node in the tree
        self.node_ranges = None
        self.hh_tree = np.empty(0)
        self.max_iter = 500
        self.rho = 1
        self.improvement = 1e-4
        self.abs_error = 1e-4

    def init_method(self, b, domain_size, epsilon):
        pass

    def init_hh(self, args):
        self.D = args.d
        self.sample_size = args.samples
        self.branch = args.branch
        self.l = int(np.log(self.D) / np.log(self.branch) + 1)
        # use array to store a B-adic tree
        self.tree = np.zeros( int((self.D * self.branch - 1)/(self.branch - 1)) )
        self.tree[0] = 1
        # build node->range mapping for each node in the tree
        self.node_ranges = self.init_node_ranges()
        self.hh_tree = np.empty(0)

    def init_node_ranges(self):
        assert(np.round(np.log(self.D) / np.log(self.branch), 2).is_integer())
        # print("branch:", self.branch)
        # exit()
        node_range = [(0, self.D-1)]
        q = Queue()
        q.put((0, self.D-1))
        while not q.empty():
            (left, right) = q.get()
            if left == right:
                continue
            r = int((right - left + 1) / self.branch)
            # print(r)
            while left <= right:
                node_range.append((left, left + r - 1))
                if r != 1:
                    q.put((left, left + r - 1))
                left = left + r

        return node_range

    # get raw HH tree, may have negative values in the nodes
    def estimate(self, real_samples, epsilon):
        print("sample size:", self.sample_size)
        layer_sample_sizes = np.random.multinomial(self.sample_size,
                                                   np.ones(self.l - 1)/(self.l-1),
                                                   size= 1)[0]
        sample_pointer = 0
        for idx, layer_count in enumerate(layer_sample_sizes):
            layer = idx + 1
            # print("layer:", layer, layer_count)
            layer_nodes = int(np.round(np.exp(layer * np.log(self.branch))))
            subset_samples = np.copy(real_samples[sample_pointer : sample_pointer + layer_count])
            sample_pointer += layer_count
            real_dist, _ = np.histogram(subset_samples, layer_nodes, range=(0, self.D))
            # print("real_dist:", real_dist)
            # specify the frequency oracle for the HH, use OUE here
            fo = LH()
            fo.init_e(epsilon, layer_nodes)
            # print("layer nodes:", layer_nodes)
            sub_estimated_dist = fo.estimate(real_dist)
            # print('estimation: ', layer_nodes, layer_count, sum(sub_estimated_dist))
            # normalize each layer, may get negative value here
            sub_estimated_dist = sub_estimated_dist/ layer_count


            # assign values to the tree layer
            tree_start = int(np.around((np.exp(layer * np.log(self.branch)) - 1) / (self.branch-1) ) )
            tree_end = int(np.around((np.exp((layer+1) * np.log(self.branch)) - 1) / (self.branch-1) ) )
            self.tree[tree_start:tree_end] = sub_estimated_dist

        self.hh_tree = np.copy(self.tree)

        return self.tree

    def get_tree_layer_nodes(self, tree, layer):
        start = int(np.around((np.exp(layer * np.log(self.branch)) - 1) / (self.branch-1)))
        end = int(np.around((np.exp((layer + 1) * np.log(self.branch)) - 1) / ( self.branch-1)))
        # print(start, end)
        # if tree == 'origin':
        #     return self.tree[start: end+1]
        # else:
        #     return self.CI_tree[start: end+1]
        return tree[start: end+1]

    def get_parent_idx(self, idx):
        return int((idx-1)/self.branch)

    def get_children_idxs(self, idx):
        left = self.branch * idx + 1
        right = self.branch * (idx + 1)
        if left >= len(self.tree) or right >= len(self.tree):
            left = np.nan
            right = np.nan
        return left, right

    #  leaves have height 0, root has height self.l
    def get_idx_tree_height(self, idx):
        layer = int(np.log(idx * (self.branch - 1) + 1) / np.log(self.branch) )
        height = self.l - layer - 1
        return height

    def HH_CI(self):
        # weighted averaging
        print("HH_CI start...")
        averaged_tree = np.copy(self.tree)
        for i in reversed(range(len(self.tree)-self.D)):
            left_child, right_child = self.get_children_idxs(i)
            height = self.get_idx_tree_height(i)
            # print(i, "height:", height)
            child_sum = sum(self.tree[left_child: right_child+1])
            w2 = (np.exp((height -1)* np.log(self.branch)) - 1)/(np.exp(height * np.log(self.branch)) - 1)
            w1 = 1 - w2
            averaged_tree[i] = w1 * self.tree[i] + w2 * child_sum

        self.CI_tree = np.copy(averaged_tree)
        # print(">>> min", min(self.CI_tree), " len:", len(self.CI_tree))
        for i in range(1, len(self.tree)):
            parent = self.get_parent_idx(i)
            left_child, right_child = self.get_children_idxs(parent)
            if np.isnan(left_child) or np.isnan(right_child):
                if self.CI_tree[i] < 0:
                    print(i, "convert to 0")
                    self.CI_tree[i] = 0
                continue
            # if self.CI_tree[parent] <= 0:
            #     # if the parent has non-positive value, all nodes in the subtree have 0 value
            #     self.CI_tree[parent] = 0
            #     self.CI_tree[i] = 0
            # else:
            #     self.CI_tree[i] = averaged_tree[i] + (self.CI_tree[parent] -
            #                                           sum(averaged_tree[left_child: right_child+1])) / self.branch
            self.CI_tree[i] = averaged_tree[i] + (self.CI_tree[parent] -
                                                  sum(averaged_tree[left_child: right_child + 1])) / self.branch

            # if self.CI_tree[i] < 0:
            #     self.CI_tree[i] = 0
        # turn the negative value in leaves to 0
        # self.CI_tree[self.CI_tree < 0] = 0
        # print(self.CI_tree)
        # print(sum(self.get_tree_layer_nodes(self.CI_tree, 2)), sum(self.get_tree_layer_nodes(self.tree, 2)))
        # print(sum(self.get_tree_layer_nodes(self.CI_tree, 1)),  sum(self.get_tree_layer_nodes(self.tree, 1)))
        # print(sum(self.get_tree_layer_nodes(self.CI_tree, 0)), sum(self.get_tree_layer_nodes(self.tree, 0)))
        # print(">> imporve:", sum(abs(self.CI_tree - self.hh_tree)))
        # print(">> min:", min(self.CI_tree))
        self.hh_tree = np.copy(self.CI_tree)
        print("HH_CI end.")
        return

    def range_query(self, start, end):
        node_idx = []
        # node_q1 and node_q2 are for BFS search
        node_q1 = Queue()
        node_q2 = Queue()
        # ranges and ranges2 are for tracking ranges
        ranges = Queue()
        ranges2 = Queue()
        node_q1.put(0)
        ranges.put((start, end))
        while not ranges.empty():
            (range_start, range_end) = ranges.get()
            while not node_q1.empty():
                node = node_q1.get()
                (l, r) = self.node_ranges[node]
                if l > range_end and not ranges.empty():
                    (range_start, range_end) = ranges.get()

                if l >= range_start and r <= range_end:
                    node_idx.append(node)
                    # print("add node:", node)
                    if not ranges.empty() and  r == range_end:
                        (range_start, range_end) = ranges.get()
                elif l < range_start and r <= range_end and r >= range_start:
                    left_child, right_child = self.get_children_idxs(node)
                    if np.isnan(left_child) or np.isnan(right_child):
                        continue
                    for i in range(left_child, right_child + 1):
                        node_q2.put(i)
                    ranges2.put((range_start, r))
                    # print("insert range 1:", range_start, r)
                elif l >= range_start and r > range_end and l <= range_end:
                    left_child, right_child = self.get_children_idxs(node)
                    if np.isnan(left_child) or np.isnan(right_child):
                        continue
                    for i in range(left_child, right_child + 1):
                        node_q2.put(i)
                    ranges2.put((l, range_end))
                    # print("insert range 2:", l, range_end, left_child, right_child)
                elif l < range_start and r > range_end:
                    left_child, right_child = self.get_children_idxs(node)
                    if np.isnan(left_child) or np.isnan(right_child):
                        continue
                    for i in range(left_child, right_child + 1):
                        node_q2.put(i)
                    ranges2.put((range_start, range_end))


            node_q1 = node_q2
            node_q2 = Queue()
            ranges = ranges2
            ranges2 = Queue()

        frequency_sum = 0
        for idx in node_idx:
            frequency_sum += self.hh_tree[idx]
        if frequency_sum < 0:
            frequency_sum = 0
        return frequency_sum

    def get_cdf(self):
        cdf = np.zeros(self.D)
        for i in range(self.D):
            cdf[i] = self.range_query(0, i)

        return cdf

    def rang_query_test(self, sample_prob_hist, query_range, norm, query_times):
        start = np.random.uniform(0, 1 - query_range / float(self.D), query_times) * self.D
        start = start.astype(int)
        end = start + query_range
        diff = []
        for i in range(query_times):
            hh_result = self.range_query2(start[i], end[i])
            true_result = sum(sample_prob_hist[start[i]: end[i] + 1])
            print("range query:", hh_result, true_result)
            if norm == 1:
                diff.append(abs(true_result - hh_result))
            else:
                diff.append(np.square(true_result - hh_result))

        return sum(diff)/len(diff)

    def get_leaves(self):
        leaves = np.copy(self.get_tree_layer_nodes(self.hh_tree, self.l-1))
        # print('number of leaves: ', len(leaves), self.l)
        return leaves

    def HTree(self, tree):
        # weighted averaging
        # print("HH_CI start...")
        averaged_tree = np.copy(tree)
        for i in reversed(range(len(tree)-self.D)):
            left_child, right_child = self.get_children_idxs(i)
            height = self.get_idx_tree_height(i)
            # print(i, "height:", height)
            child_sum = sum(tree[left_child: right_child+1])
            w2 = (np.exp((height -1)* np.log(self.branch)) - 1)/(np.exp(height * np.log(self.branch)) - 1)
            w1 = 1 - w2
            averaged_tree[i] = w1 * tree[i] + w2 * child_sum

        return_tree = np.copy(averaged_tree)
        # print(">>> min", min(self.CI_tree), " len:", len(self.CI_tree))
        for i in range(1, len(tree)):
            parent = self.get_parent_idx(i)
            left_child, right_child = self.get_children_idxs(parent)
            if np.isnan(left_child) or np.isnan(right_child):
                if return_tree[i] < 0:
                    # print(i, "convert to 0")
                    return_tree[i] = 0
                continue
             return_tree[i] = averaged_tree[i] + (return_tree[parent] -
                                                  sum(averaged_tree[left_child: right_child + 1])) / self.branch
        # print("HTree end.")
        return return_tree

    def ADMM_post_processiong(self):
        # print("self.D", self.D)
        x = np.copy(self.tree)
        # print(self.get_tree_layer_nodes(x, 1))
        # print(self.get_tree_layer_nodes(x, 2))
        y = np.zeros(len(self.tree))
        z = np.zeros(len(self.tree))
        w = np.zeros(len(self.tree))
        mu = np.zeros(len(self.tree))
        nu = np.zeros(len(self.tree))
        eta = np.zeros(len(self.tree))

        iter = 0
        while iter < self.max_iter:
            old_x = np.copy(x)
            y = self.rho / (self.rho + 1) * (x - self.tree + mu)
            z = self.HTree(x + nu)
            w = x + eta
            w[w < 0] = 0
            x = ((y + self.tree - mu) + (z - nu) + (w - eta)) / 3

            mu = mu + x - self.tree - y
            nu = nu + x - z
            eta = eta + x - w

            improve = self.rho * np.sqrt(np.linalg.norm(x - old_x))
            diff1 = np.sqrt(np.linalg.norm(x - y - self.tree))
            diff2 = np.sqrt(np.linalg.norm(x - z))
            diff3 = np.sqrt(np.linalg.norm(x - w))
            # print(iter, "iter:", improve, diff1, diff2, diff3)
            if  improve < self.improvement:
                if diff1 < self.abs_error:
                    if diff2 < self.abs_error:
                        if diff3 < self.abs_error:
                            print("break with iteration:", iter)
                            break

            iter += 1

        self.hh_tree = x
        return x