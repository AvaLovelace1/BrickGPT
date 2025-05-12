import re
import warnings
from dataclasses import dataclass

import numpy as np

from .lego_library import (lego_library,
                           dimensions_to_brick_id, brick_id_to_dimensions,
                           brick_id_to_part_id, part_id_to_brick_id)


@dataclass(frozen=True, order=True, kw_only=True)
class LegoBrick:
    """
    Represents a 1-unit-tall rectangular LEGO brick.
    """
    h: int
    w: int
    x: int
    y: int
    z: int

    @property
    def brick_id(self) -> int:
        return dimensions_to_brick_id(self.h, self.w)

    @property
    def part_id(self) -> str:
        return brick_id_to_part_id(self.brick_id)

    @property
    def ori(self) -> int:
        return 1 if self.h > self.w else 0

    @property
    def area(self) -> int:
        return self.h * self.w

    @property
    def slice_2d(self) -> (slice, slice):
        return slice(self.x, self.x + self.h), slice(self.y, self.y + self.w)

    @property
    def slice(self) -> (slice, slice, int):
        return *self.slice_2d, self.z

    def __repr__(self):
        return self.to_txt()[:-1]

    def to_json(self) -> dict:
        return {
            'brick_id': self.brick_id,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'ori': self.ori,
        }

    def to_txt(self) -> str:
        return f'{self.h}x{self.w} ({self.x},{self.y},{self.z})\n'

    def to_ldr(self, base_height: float = 0) -> str:
        x = (self.x + self.h * 0.5) * 20
        z = (self.y + self.w * 0.5) * 20
        y = (self.z + base_height) * -24
        matrix = '0 0 1 0 1 0 -1 0 0' if self.ori == 0 else '-1 0 0 0 1 0 0 0 -1'
        line = f'1 115 {x} {y} {z} {matrix} {self.part_id}\n'
        step_line = '0 STEP\n'
        return line + step_line

    @classmethod
    def from_json(cls, brick_json: dict):
        h, w = brick_id_to_dimensions(brick_json['brick_id'])
        if brick_json['ori'] == 1:
            h, w = w, h
        x, y, z = brick_json['x'], brick_json['y'], brick_json['z']
        return cls(h=h, w=w, x=x, y=y, z=z)

    @classmethod
    def from_txt(cls, brick_txt: str):
        brick_txt = brick_txt.strip()
        match = re.fullmatch(r'(\d+)x(\d+) \((\d+),(\d+),(\d+)\)', brick_txt)
        if match is None:
            raise ValueError(f'Text Format brick is ill-formatted: {brick_txt}')

        h, w, x, y, z = map(int, match.group(1, 2, 3, 4, 5))
        return cls(h=h, w=w, x=x, y=y, z=z)

    @classmethod
    def from_ldr(cls, brick_ldr: str):
        ldr_components = brick_ldr.strip().split()
        match ldr_components:
            case ['1', _, x0, y0, z0, *matrix, part_id]:
                x0, y0, z0 = map(float, (x0, y0, z0))
                matrix_str = ' '.join(matrix)
                if matrix_str == '0 0 1 0 1 0 -1 0 0':
                    ori = 0
                elif matrix_str == '-1 0 0 0 1 0 0 0 -1':
                    ori = 1
                else:
                    raise ValueError(f'Invalid transformation matrix: {matrix_str}')

                h, w = brick_id_to_dimensions(part_id_to_brick_id(part_id))
                if ori == 1:
                    h, w = w, h

                x = int(x0 / 20 - h * 0.5)
                y = int(z0 / 20 - w * 0.5)
                z = int(-y0 / 24)

                return cls(h=h, w=w, x=x, y=y, z=z)
            case _:
                raise ValueError(f"LDR format is ill-formatted: {brick_ldr}")


class LegoStructure:
    """
    Represents a LEGO structure in the form of a list of LEGO bricks.
    """

    def __init__(self, bricks: list[LegoBrick], world_dim: int = 20):
        self.world_dim = world_dim

        # Check if structure starts at ground level
        z0 = min((brick.z for brick in bricks), default=0)
        if z0 != 0:
            warnings.warn('LEGO structure does not start at ground level z=0.')

        # Build structure from bricks
        self.bricks = []
        self.voxel_occupancy = np.zeros((world_dim, world_dim, world_dim), dtype=int)
        for brick in bricks:
            self.add_brick(brick)

    def __len__(self):
        return len(self.bricks)

    def __repr__(self):
        return self.to_txt()

    def to_json(self) -> dict:
        return {str(i + 1): brick.to_json() for i, brick in enumerate(self.bricks)}

    def to_txt(self) -> str:
        return ''.join([brick.to_txt() for brick in self.bricks])

    def to_ldr(self) -> str:
        return ''.join([brick.to_ldr() for brick in self.bricks])

    def add_brick(self, brick: LegoBrick) -> None:
        self.bricks.append(brick)
        self.voxel_occupancy[brick.slice] += 1

    def undo_add_brick(self) -> None:
        brick = self.bricks[-1]
        self.voxel_occupancy[brick.slice] -= 1
        self.bricks.pop()

    def has_out_of_bounds_bricks(self) -> bool:
        return any(not self.brick_in_bounds(brick) for brick in self.bricks)

    def brick_in_bounds(self, brick: LegoBrick) -> bool:
        return (all(slice_.start >= 0 and slice_.stop <= self.world_dim for slice_ in brick.slice_2d)
                and 0 <= brick.z < self.world_dim)

    def has_collisions(self) -> bool:
        return np.any(self.voxel_occupancy > 1)

    def brick_collides(self, brick: LegoBrick) -> bool:
        return np.any(self.voxel_occupancy[brick.slice])

    def has_floating_bricks(self) -> bool:
        return any(self.brick_floats(brick) for brick in self.bricks)

    def brick_floats(self, brick: LegoBrick) -> bool:
        if brick.z == 0:
            return False  # Supported by ground
        if np.any(self.voxel_occupancy[*brick.slice_2d, brick.z - 1]):
            return False  # Supported from below
        if brick.z != self.world_dim - 1 and np.any(self.voxel_occupancy[*brick.slice_2d, brick.z + 1]):
            return False  # Supported from above
        return True

    def is_stable(self) -> bool:
        """
        Check if the structure is stable using basic structural rules:
        1. No floating bricks
        2. No collisions
        3. Each brick must be supported from below or be at ground level
        4. Each brick must have at least one connection point with adjacent bricks
        """
        if self.has_floating_bricks() or self.has_collisions():
            return False

        # Check that each brick is either at ground level or has support from below
        for brick in self.bricks:
            if brick.z == 0:
                continue  # Ground level is always stable

            # Check if brick has support from below
            has_support = False
            for x in range(brick.x, brick.x + brick.h):
                for y in range(brick.y, brick.y + brick.w):
                    if self.voxel_occupancy[x, y, brick.z - 1]:
                        has_support = True
                        break
                if has_support:
                    break

            if not has_support:
                return False

            # Check if brick has at least one connection point with adjacent bricks
            has_connection = False
            # Check connections in x direction
            if brick.x > 0:
                for y in range(brick.y, brick.y + brick.w):
                    if self.voxel_occupancy[brick.x - 1, y, brick.z]:
                        has_connection = True
                        break
            if not has_connection and brick.x + brick.h < self.world_dim:
                for y in range(brick.y, brick.y + brick.w):
                    if self.voxel_occupancy[brick.x + brick.h, y, brick.z]:
                        has_connection = True
                        break

            # Check connections in y direction
            if not has_connection and brick.y > 0:
                for x in range(brick.x, brick.x + brick.h):
                    if self.voxel_occupancy[x, brick.y - 1, brick.z]:
                        has_connection = True
                        break
            if not has_connection and brick.y + brick.w < self.world_dim:
                for x in range(brick.x, brick.x + brick.h):
                    if self.voxel_occupancy[x, brick.y + brick.w, brick.z]:
                        has_connection = True
                        break

            if not has_connection:
                return False

        return True

    def stability_scores(self) -> np.ndarray:
        """
        This method is deprecated. Use is_stable() instead for stability checking.
        Returns a dummy array of zeros to maintain compatibility.
        """
        warnings.warn("stability_scores() is deprecated. Use is_stable() instead for stability checking.",
                     DeprecationWarning, stacklevel=2)
        return np.zeros((self.world_dim, self.world_dim, self.world_dim))

    @classmethod
    def from_json(cls, lego_json: dict):
        bricks = [LegoBrick.from_json(v) for k, v in lego_json.items() if k.isdigit()]
        return cls(bricks)

    @classmethod
    def from_txt(cls, lego_txt: str):
        bricks_txt = lego_txt.split('\n')
        bricks_txt = [b for b in bricks_txt if b.strip()]  # Remove blank lines
        bricks = [LegoBrick.from_txt(brick) for brick in bricks_txt]
        return cls(bricks)

    @classmethod
    def from_ldr(cls, lego_ldr: str):
        bricks_ldr = lego_ldr.split('0 STEP')  # Split on step lines
        bricks_ldr = [b for b in bricks_ldr if b.strip()]  # Remove blank or whitespace-only lines
        bricks = [LegoBrick.from_ldr(brick) for brick in bricks_ldr]
        return cls(bricks)
