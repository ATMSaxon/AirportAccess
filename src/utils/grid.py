"""ENU voxel-grid construction for OLS SDFs and envelopes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class VoxelGrid:
    """Regular ENU voxel grid centred on the airport ARP."""
    x_min: float; x_max: float; dx: float
    y_min: float; y_max: float; dy: float
    z_min: float; z_max: float; dz: float

    @property
    def shape(self) -> tuple[int, int, int]:
        nx = int(round((self.x_max - self.x_min) / self.dx))
        ny = int(round((self.y_max - self.y_min) / self.dy))
        nz = int(round((self.z_max - self.z_min) / self.dz))
        return nx, ny, nz

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    def coords(self):
        nx, ny, nz = self.shape
        xs = self.x_min + (np.arange(nx) + 0.5) * self.dx
        ys = self.y_min + (np.arange(ny) + 0.5) * self.dy
        zs = self.z_min + (np.arange(nz) + 0.5) * self.dz
        return xs, ys, zs

    def meshgrid(self):
        xs, ys, zs = self.coords()
        return np.meshgrid(xs, ys, zs, indexing="ij")

    @classmethod
    def from_airport_cfg(cls, cfg: dict) -> "VoxelGrid":
        box = cfg["extract_box_m"]
        res = cfg["grid_resolution_m"]
        return cls(
            x_min=-float(box["half_x"]), x_max=float(box["half_x"]), dx=float(res["xy"]),
            y_min=-float(box["half_y"]), y_max=float(box["half_y"]), dy=float(res["xy"]),
            z_min=float(box["z_min"]),   z_max=float(box["z_max"]),  dz=float(res["z"]),
        )

    def world_to_index(self, x, y, z):
        ix = np.clip(np.floor((np.asarray(x) - self.x_min) / self.dx).astype(int), 0, self.shape[0] - 1)
        iy = np.clip(np.floor((np.asarray(y) - self.y_min) / self.dy).astype(int), 0, self.shape[1] - 1)
        iz = np.clip(np.floor((np.asarray(z) - self.z_min) / self.dz).astype(int), 0, self.shape[2] - 1)
        return ix, iy, iz
