import time
from queue import PriorityQueue
from typing import Callable

import networkx as nx
import numpy as np

from mesh2brick.data.brick_library import brick_library, dimensions_to_brick_id
from mesh2brick.data.brick_structure import Brick, BrickStructure, ConnectivityBrickStructure
from mesh2brick.planning import plan_robotic_operation


def first_zero_idx(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Finds the index of the first occurrence of 0 along axis
    Returns the length of the last dimension if no zero occurs.
    """
    arr_eq_zero = arr == 0
    return np.where(arr_eq_zero.any(axis=axis), np.argmax(arr_eq_zero, axis=axis), arr.shape[axis])


def first_nonzero_idx(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    return first_zero_idx(arr == 0, axis)


def k_ring_neighbors(node, k: int, graph: nx.Graph) -> list:
    shortest_paths = nx.single_source_shortest_path(graph, node, cutoff=k)
    return list(shortest_paths.keys())


def valid_brick(h, w) -> bool:
    try:
        dimensions_to_brick_id(h, w)
        return True
    except ValueError:
        return False


def get_merged_brick(b1: Brick, b2: Brick) -> Brick | None:
    assert b1.z == b2.z

    if b1.x == b2.x and b1.h == b2.h and (b1.y + b1.w == b2.y or b2.y + b2.w == b1.y):
        new_h, new_w = b1.h, b1.w + b2.w
        if valid_brick(new_h, new_w):
            new_x, new_y = b1.x, min(b1.y, b2.y)
            return Brick(h=new_h, w=new_w, x=new_x, y=new_y, z=b1.z)

    elif b1.y == b2.y and b1.w == b2.w and (b1.x + b1.h == b2.x or b2.x + b2.h == b1.x):
        new_h, new_w = b1.h + b2.h, b1.w
        if valid_brick(new_h, new_w):
            new_x, new_y = min(b1.x, b2.x), b1.y
            return Brick(h=new_h, w=new_w, x=new_x, y=new_y, z=b1.z)

    return None


class Voxel2Brick:
    def __init__(self, voxels: np.ndarray, max_failures: int = 10, seed: int = 42):
        self.voxels = voxels.astype(bool)
        self.bricks = ConnectivityBrickStructure(voxels.shape)

        self.n_failures = 0
        self.max_failures = max_failures

        self.rng = np.random.default_rng(seed)

    @property
    def max_x(self) -> int:
        return self.voxels.shape[0]

    @property
    def max_y(self) -> int:
        return self.voxels.shape[1]

    @property
    def max_z(self) -> int:
        return self.voxels.shape[2]

    def __call__(self) -> list[Brick]:
        t_start = time.time()

        # Initialize structure greedily
        self._brickify_voxels_greedy(self.voxels, self._greedy_priority)
        min_components_possible = nx.number_connected_components(self.bricks.neighbor_graph)

        # Split and re-merge critical connectivity areas
        n_components = self.bricks.n_components()
        self.n_failures = 0
        while self.n_failures < self.max_failures:
            if n_components == min_components_possible:
                break
            critical_voxels = self._find_critical_voxels_connectivity()
            removed_bricks = self.bricks.remove_voxel_subset(critical_voxels)
            reverse_layer_order = (self.rng.uniform() > 0.5)
            self._brickify_voxels_greedy(critical_voxels, self._component_priority,
                                         reverse_layer_order=reverse_layer_order)

            # Are the results better?
            new_n_components = self.bricks.n_components()
            if new_n_components < n_components:
                n_components = new_n_components
                self.n_failures = 0
            else:  # No improvement; revert merge
                self.bricks.remove_voxel_subset(critical_voxels)
                self.bricks.add_bricks(removed_bricks)
                self.n_failures += 1

        # Split and re-merge critical stability areas
        stability = self.bricks.stability_score()
        n_components = self.bricks.n_components()
        self.n_failures = 0
        while self.n_failures < self.max_failures:
            if stability.max() < 1.0:
                break
            critical_voxels = self._find_critical_voxels_stability(stability)
            removed_bricks = self.bricks.remove_voxel_subset(critical_voxels)
            self._brickify_voxels_merge(critical_voxels)

            # Are the results better?
            new_stability = self.bricks.stability_score()
            new_n_components = self.bricks.n_components()
            if new_stability.mean() < stability.mean() and new_n_components <= n_components:
                stability = new_stability
                n_components = new_n_components
                self.n_failures = 0
            else:  # No improvement; revert merge
                self.bricks.remove_voxel_subset(critical_voxels)
                self.bricks.add_bricks(removed_bricks)
                self.n_failures += 1

        mesh2brick_time = time.time() - t_start
        print(f'Finished in time: {mesh2brick_time:.4f} s | '
              f'# bricks: {len(self.bricks.bricks)} | '
              f'# connected components: {n_components} | '
              f'# min connected components possible: {min_components_possible} | '
              f'Stability: {stability.max()}')

        return list(self.bricks.bricks.values())

    def _brickify_voxels_greedy(
            self,
            voxel_subset: np.ndarray,
            priority: Callable,
            reverse_layer_order: bool = False,
    ) -> None:
        self._brickify_voxels(voxel_subset, lambda v, z: self._brickify_layer_greedy(v, z, priority),
                              reverse_layer_order=reverse_layer_order)

    def _brickify_voxels_merge(self, voxel_subset: np.ndarray, reverse_layer_order: bool = False) -> None:
        self._brickify_voxels(voxel_subset, self._brickify_layer_merge, reverse_layer_order=reverse_layer_order)

    def _brickify_voxels(
            self,
            voxel_subset: np.ndarray,
            layer_brickify_fn: Callable,
            reverse_layer_order: bool = False,
    ) -> None:
        min_z = first_nonzero_idx(voxel_subset.sum(axis=(0, 1)))
        max_z = self.max_z - first_nonzero_idx(voxel_subset.sum(axis=(0, 1))[::-1])
        if reverse_layer_order:
            for z in reversed(range(min_z, max_z)):
                layer_brickify_fn(voxel_subset, z)
        else:
            for z in range(min_z, max_z):
                layer_brickify_fn(voxel_subset, z)
        assert ((self.bricks.voxel_bricks != 0) == (self.voxels != 0)).all()

    def _brickify_layer_greedy(self, voxel_subset: np.ndarray, z: int, priority: Callable) -> None:
        brick_dimensions = ([(v['height'], v['width']) for v in brick_library.values()] +
                            [(v['width'], v['height']) for v in brick_library.values()
                             if v['height'] != v['width']])

        # Enumerate possible brick placements
        min_x = first_nonzero_idx(voxel_subset[..., z].sum(axis=1))
        max_x = self.max_x - first_nonzero_idx(voxel_subset[..., z].sum(axis=1)[::-1])
        min_y = first_nonzero_idx(voxel_subset[..., z].sum(axis=0))
        max_y = self.max_y - first_nonzero_idx(voxel_subset[..., z].sum(axis=0)[::-1])
        all_brick_placements = [Brick(h=h, w=w, x=x, y=y, z=z)
                                for h, w in brick_dimensions
                                for x in range(min_x, max_x - h + 1) for y in range(min_y, max_y - w + 1)]

        # Filter out bricks that are not completely contained within the voxels
        valid_brick_placements = list(filter(lambda b: voxel_subset[b.slice].all(), all_brick_placements))

        # Place bricks in order of priority
        for brick in sorted(valid_brick_placements, key=priority):
            try:
                self.bricks.add_brick(brick)
            except ValueError:
                pass

    def _greedy_priority(self, brick: Brick):
        dangles = 1 if 0 < self._calc_support_ratio(brick) < 1 else 0
        shorter_side = min(brick.h, brick.w)
        ori_priority = (-1 if brick.ori == 0 else 1) * (-1) ** brick.z
        return (-dangles, -self._count_gaps(brick), -shorter_side, -brick.area, ori_priority,
                brick.x, brick.y, brick.z)

    def _component_priority(self, brick: Brick):
        return -self._count_connecting_components(brick), -brick.area, self.rng.uniform()

    def _calc_support_ratio(self, brick: Brick) -> float:
        if brick.z == 0:
            return 1.0
        total_area = brick.h * brick.w
        supported_area = self.voxels[*brick.slice_2d, brick.z - 1].sum()
        return supported_area / total_area

    def _count_gaps(self, brick: Brick) -> int:
        """
        A "gap" is a pair of voxels beneath the brick that belong to two different bricks
        (and hence those two bricks will be connected by placing the brick).
        This function returns the sum of the depths of gaps beneath the brick.
        """
        if brick.z == 0:
            return 0

        structure_under_brick = self.bricks.voxel_bricks[*brick.slice_2d, :brick.z]
        # Equals 1 at [x,y,z] if voxels [x,y,z] and [x+1,y,z] are in different bricks
        horz_gaps = structure_under_brick[:-1, :, :] != structure_under_brick[1:, :, :]
        # Equals 1 at [x,y,z] if voxels [x,y,z] and [x,y+1,z] are in different bricks
        vert_gaps = structure_under_brick[:, :-1, :] != structure_under_brick[:, 1:, :]

        # [x,y,z] = d, where d is the largest integer such that [x,y,z-i] != [x+1,y,z-i] for all i < d
        horz_gap_depths = first_zero_idx(horz_gaps[..., ::-1])
        vert_gap_depths = first_zero_idx(vert_gaps[..., ::-1])

        return horz_gap_depths.sum() + vert_gap_depths.sum()

    def _count_connecting_components(self, brick: Brick) -> int:
        """
        Returns the number of components that will be connected if brick is added to the structure.
        """
        components = set()
        if brick.z > 0:
            components |= set(np.unique(self.bricks.component_labels()[*brick.slice_2d, brick.z - 1])) - {0}
        if brick.z < self.max_z - 1:
            components |= set(np.unique(self.bricks.component_labels()[*brick.slice_2d, brick.z + 1])) - {0}
        return len(components)

    def _brickify_layer_merge(self, voxel_subset: np.ndarray, z: int) -> None:
        # Fill with 1x1 bricks
        voxel_idxs = list(zip(*np.nonzero(voxel_subset[..., z])))
        brick_1x1s = [Brick(h=1, w=1, x=x, y=y, z=z) for x, y in voxel_idxs]
        node_ids = [self.bricks.add_brick(brick) for brick in brick_1x1s]

        # Add all mergeable brick pairs to queue
        pq = PriorityQueue()
        for b1 in node_ids:
            self._add_mergeable_pairs_to_queue(b1, pq, voxel_subset)

        # Merge pairs until queue is empty
        while not pq.empty():
            _, b1, b2, merged_brick = pq.get()
            if not self.bricks.node_exists(b1) or not self.bricks.node_exists(b2):
                continue
            self.bricks.remove_brick(b1)
            self.bricks.remove_brick(b2)
            b3 = self.bricks.add_brick(merged_brick)
            self._add_mergeable_pairs_to_queue(b3, pq, voxel_subset)

    def _add_mergeable_pairs_to_queue(self, b1: int, pq: PriorityQueue, voxel_subset: np.ndarray) -> None:
        for b2 in self.bricks.neighbor_graph.neighbors(b1):
            brick1, brick2 = self.bricks.bricks[b1], self.bricks.bricks[b2]
            if brick2.z != brick1.z or not voxel_subset[brick2.slice].all():
                continue
            merged_brick = get_merged_brick(brick1, brick2)
            if merged_brick:
                pq.put((self.rng.uniform(0, 1), b1, b2, merged_brick))

    def _find_critical_voxels_connectivity(self) -> np.ndarray:
        """
        From the Legolization paper
        """
        nodes = list(self.bricks.bricks.keys())
        pvals = np.array([self._num_neighboring_components(node) - 1
                          for node in nodes], dtype=float)
        pvals /= pvals.sum()

        selected_node_idx = np.argmax(self.rng.multinomial(1, pvals))
        weakest_node = nodes[selected_node_idx]
        return self._get_critical_voxels(weakest_node)

    def _num_neighboring_components(self, node: int) -> int:
        components = (
                    {self.bricks.node2component()[neighbor] for neighbor in self.bricks.neighbor_graph.neighbors(node)}
                    | {self.bricks.node2component()[node]})
        return len(components)

    def _find_critical_voxels_stability(self, stability: np.ndarray) -> np.ndarray:
        """
        From the Legolization paper. Returns bricks surrounding the weakest brick in the structure
        """
        # Find weakest node in structure
        if stability.max() < 1.0:
            return np.zeros_like(self.voxels)
        weakest_node_idx = np.unravel_index(np.argmax(stability), stability.shape)
        weakest_node = self.bricks.voxel_bricks[weakest_node_idx]
        return self._get_critical_voxels(weakest_node)

    def _get_critical_voxels(self, critical_node) -> np.ndarray:
        critical_nodes = k_ring_neighbors(critical_node, self._k_ring_size(), self.bricks.neighbor_graph)
        critical_bricks = [self.bricks.bricks[n] for n in critical_nodes]
        critical_voxels = np.zeros_like(self.voxels)
        for brick in critical_bricks:
            critical_voxels[brick.slice] = 1
        return critical_voxels

    def _k_ring_size(self) -> int:
        return self.n_failures // 10 + 1


def voxel2brick(voxels: np.ndarray, **kwargs) -> BrickStructure:
    v2l = Voxel2Brick(voxels, **kwargs)
    bricks = v2l()

    bricks_by_layer = {z: [] for z in range(v2l.max_z)}
    for brick in bricks:
        bricks_by_layer[brick.z].append(brick)

    directed_brick_graph = plan_robotic_operation(bricks_by_layer)
    return BrickStructure.from_json(directed_brick_graph, world_dim=voxels.shape[0])
