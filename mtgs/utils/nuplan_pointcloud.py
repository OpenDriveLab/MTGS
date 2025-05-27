from __future__ import annotations

from io import BytesIO
from typing import IO, Any, List, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

def load_lidar(lidar_path, remove_close=False):

    points = PointCloud.parse_from_file(lidar_path).to_pcd_bin2().T
    points = points[:, :3]  # (n, 3)

    if remove_close:
        x_radius = 3.0
        y_radius = 3.0
        z_radius = 2.0
        x_filt = np.abs(points[:, 0]) < x_radius
        y_filt = np.abs(points[:, 1]) < y_radius
        z_filt = np.abs(points[:, 2]) < z_radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt, z_filt))
        return points[not_close]

    return points


def get_rgb_point_cloud(points, lidar2imgs, images):
    """ get rdb point cloud from multi-view images 
    """
    image_shape = np.array([[image.shape[1], image.shape[0]] for image in images])[:, None, :] # [N, 1, 2]

    xyz1 = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=-1)
    pts_2d = lidar2imgs @ xyz1.T
    pts_2d = pts_2d.transpose(0, 2, 1)
    fov_inds = pts_2d[..., 2] > 0.1
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=99999)
    pts_2d[..., :2] /= pts_2d[..., 2:3]
    pts_2d = pts_2d[..., :2]

    pts_2d /= image_shape # normalize to [0, 1]
    fov_inds = (fov_inds & (pts_2d[..., 0] < 1)
                & (pts_2d[..., 0] >= 0)
                & (pts_2d[..., 1] < 1)
                & (pts_2d[..., 1] >= 0))

    pts_2d = torch.from_numpy(pts_2d)
    pts_2d = pts_2d * 2 - 1  # [-1, +1]

    images = np.array(images, dtype=np.float64)[..., ::-1] / 255.0  # BGR to RGB [0, 1]
    images = torch.from_numpy(images).permute(0, 3, 1, 2)  # to NCHW

    RGB = torch.nn.functional.grid_sample(
        images, pts_2d.unsqueeze(2), align_corners=True, padding_mode='zeros').squeeze(3).permute(0, 2, 1)
    RGB = RGB.cpu().numpy()
    RGB = np.where(fov_inds[..., None], RGB, 0).sum(0)
    fov_num = fov_inds.sum(0)
    fov_mask = fov_num>0
    RGB[fov_mask] /= fov_num[fov_mask][:, None]
    return RGB, fov_mask



class PointCloudHeader(NamedTuple):
    """Class for Point Cloud header."""

    version: str
    fields: List[str]
    size: List[int]
    type: List[str]
    count: List[int]  # type: ignore
    width: int
    height: int
    viewpoint: List[int]
    points: int
    data: str


class PointCloud:
    """
    Class for raw .pcd file.
    """

    def __init__(self, header: PointCloudHeader, points: npt.NDArray[np.float64]) -> None:
        """
        PointCloud.
        :param header: Pointcloud header.
        :param points: <np.ndarray, X, N>. X columns, N points.
        """
        self._header = header
        self._points = points

    @property
    def header(self) -> PointCloudHeader:
        """
        Returns pointcloud header.
        :return: A PointCloudHeader instance.
        """
        return self._header

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """
        Returns points.
        :return: <np.ndarray, X, N>. X columns, N points.
        """
        return self._points

    def save(self, file_path: str) -> None:
        """
        Saves to .pcd file.
        :param file_path: The path to the .pcd file.
        """
        with open(file_path, 'wb') as fp:
            fp.write('# .PCD v{} - Point Cloud Data file format\n'.format(self._header.version).encode('utf8'))
            for field in self._header._fields:
                value = getattr(self._header, field)
                if isinstance(value, list):
                    text = ' '.join(map(str, value))
                else:
                    text = str(value)
                fp.write('{} {}\n'.format(field.upper(), text).encode('utf8'))
            fp.write(self._points.tobytes())

    @classmethod
    def parse(cls, pcd_content: bytes) -> PointCloud:
        """
        Parses the pointcloud from byte stream.
        :param pcd_content: The byte stream that holds the pcd content.
        :return: A PointCloud object.
        """
        with BytesIO(pcd_content) as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @classmethod
    def parse_from_file(cls, pcd_file: str) -> PointCloud:
        """
        Parses the pointcloud from .pcd file on disk.
        :param pcd_file: The path to the .pcd file.
        :return: A PointCloud instance.
        """
        with open(pcd_file, 'rb') as stream:
            header = cls.parse_header(stream)
            points = cls.parse_points(stream, header)
            return cls(header, points)

    @staticmethod
    def parse_header(stream: IO[Any]) -> PointCloudHeader:
        """
        Parses the header of a pointcloud from byte IO stream.
        :param stream: Binary stream.
        :return: A PointCloudHeader instance.
        """
        headers_list = []
        while True:
            line = stream.readline().decode('utf8').strip()
            if line.startswith('#'):
                continue
            columns = line.split()
            key = columns[0].lower()
            val = columns[1:] if len(columns) > 2 else columns[1]
            headers_list.append((key, val))

            if key == 'data':
                break

        headers = dict(headers_list)
        headers['size'] = list(map(int, headers['size']))
        headers['count'] = list(map(int, headers['count']))
        headers['width'] = int(headers['width'])
        headers['height'] = int(headers['height'])
        headers['viewpoint'] = list(map(int, headers['viewpoint']))
        headers['points'] = int(headers['points'])
        header = PointCloudHeader(**headers)

        if any([c != 1 for c in header.count]):
            raise RuntimeError('"count" has to be 1')

        if not len(header.fields) == len(header.size) == len(header.type) == len(header.count):
            raise RuntimeError('fields/size/type/count field number are inconsistent')

        return header

    @staticmethod
    def parse_points(stream: IO[Any], header: PointCloudHeader) -> npt.NDArray[np.float64]:
        """
        Parses points from byte IO stream.
        :param stream: Byte stream that holds the points.
        :param header: <np.ndarray, X, N>. A numpy array that has X columns(features), N points.
        :return: Points of Point Cloud.
        """
        if header.data != 'binary':
            raise RuntimeError('Un-supported data foramt: {}. "binary" is expected.'.format(header.data))

        # There is garbage data at the end of the stream, usually all b'\x00'.
        row_type = PointCloud.np_type(header)
        length = row_type.itemsize * header.points
        buff = stream.read(length)
        if len(buff) != length:
            raise RuntimeError('Incomplete pointcloud stream: {} bytes expected, {} got'.format(length, len(buff)))

        points = np.frombuffer(buff, row_type)

        return points

    @staticmethod
    def np_type(header: PointCloudHeader) -> np.dtype:  # type: ignore
        """
        Helper function that translate column types in pointcloud to np types.
        :param header: A PointCloudHeader object.
        :return: np.dtype that holds the X features.
        """
        type_mapping = {'I': 'int', 'U': 'uint', 'F': 'float'}
        np_types = [type_mapping[t] + str(int(s) * 8) for t, s in zip(header.type, header.size)]

        return np.dtype([(f, getattr(np, nt)) for f, nt in zip(header.fields, np_types)])

    def to_pcd_bin(self) -> npt.NDArray[np.float32]:
        """
        Converts pointcloud to .pcd.bin format.
        :return: <np.float32, 5, N>, the point cloud in .pcd.bin format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])

    def to_pcd_bin2(self) -> npt.NDArray[np.float32]:
        """
        Converts pointcloud to .pcd.bin2 format.
        :return: <np.float32, 6, N>, the point cloud in .pcd.bin2 format.
        """
        lidar_fields = ['x', 'y', 'z', 'intensity', 'ring', 'lidar_info']
        return np.array([np.array(self.points[f], dtype=np.float32) for f in lidar_fields])
