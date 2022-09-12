# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# Altered by: Joseph Rowell (ucabcrr@ucl.ac.uk)

# USAGE 
# # Navigate to code file 
# !python database.py  --database_path $DATABASE_PATH

from fileinput import filename
import sys
import sqlite3 
import numpy as np
import collections
from visualize_model import parse_args
from plane_intersection import Plane, Line
from ply_manipulation import ply_manipulation

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

## const LABEL_NAMES for desired DeepLab model labels.
#Pascal VOC
PASCALVOC_LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

#Cityscapes
CITYSCAPES_LABEL_NAMES = np.asarray([
    'unlabeled', 'ego vehicle', 'out of roi', 'static', 'dynamic', 'ground', 'road',
    'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
    'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 
    'motorcycle', 'bicycle', 'license plate'
    ])

#ADE20K
ADE20K_LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation',  'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle'
    ])

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])



CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))

class readCOLMAPOutput:
    # functions taken from COLMAP's read_write_model.py and adapted to be OOP
    def __init__(self) -> None:
        pass
    def read_images_text(self,path):
    
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesText(const std::string& path)
            void Reconstruction::WriteImagesText(const std::string& path)
        """
        images = {}
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    image_id = int(elems[0])
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    camera_id = int(elems[8])
                    image_name = elems[9]
                    elems = fid.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                           tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    images[image_id] = Image(
                        id=image_id, qvec=qvec, tvec=tvec,
                        camera_id=camera_id, name=image_name,
                        xys=xys, point3D_ids=point3D_ids)
        return images

    def read_points3D_text(self, path):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DText(const std::string& path)
            void Reconstruction::WritePoints3DText(const std::string& path)
        """
        points3D = {}
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    point3D_id = int(elems[0])
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    rgb = np.array(tuple(map(int, elems[4:7])))
                    error = float(elems[7])
                    image_ids = np.array(tuple(map(int, elems[8::2])))
                    point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                    points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                                   error=error, image_ids=image_ids,
                                                   point2D_idxs=point2D_idxs)
        return points3D

    def read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character="<"):
        import struct
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_points3D_binary(self, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                binary_point_line_properties = self.read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                point3D_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = self.read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = self.read_next_bytes(
                    fid, num_bytes=8*track_length,
                    format_char_sequence="ii"*track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb,
                    error=error, image_ids=image_ids,
                    point2D_idxs=point2D_idxs)
        return points3D

    def read_images_binary(self, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesBinary(const std::string& path)
            void Reconstruction::WriteImagesBinary(const std::string& path)
        """
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_reg_images):
                binary_image_properties = self.read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = self.read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":   # look for the ASCII 0 entry
                    image_name += current_char.decode("utf-8")
                    current_char = self.read_next_bytes(fid, 1, "c")[0]
                num_points2D = self.read_next_bytes(fid, num_bytes=8,
                                               format_char_sequence="Q")[0]
                x_y_id_s = self.read_next_bytes(fid, num_bytes=24*num_points2D,
                                           format_char_sequence="ddq"*num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
        return images

    def read_cameras_text(self,path):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasText(const std::string& path)
            void Reconstruction::ReadCamerasText(const std::string& path)
        """
        cameras = {}
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    camera_id = int(elems[0])
                    model = elems[1]
                    width = int(elems[2])
                    height = int(elems[3])
                    params = np.array(tuple(map(float, elems[4:])))
                    cameras[camera_id] = Camera(id=camera_id, model=model,
                                                width=width, height=height,
                                                params=params)
        return cameras

    def read_cameras_binary(self, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasBinary(const std::string& path)
            void Reconstruction::ReadCamerasBinary(const std::string& path)
        """
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_properties = self.read_next_bytes(
                    fid, num_bytes=24, format_char_sequence="iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                params = self.read_next_bytes(fid, num_bytes=8*num_params,
                                         format_char_sequence="d"*num_params)
                cameras[camera_id] = Camera(id=camera_id,
                                            model=model_name,
                                            width=width,
                                            height=height,
                                            params=np.array(params))
            assert len(cameras) == num_cameras
        return cameras
    
    def read_model(self, path, ext=""):
        import os

        # try to detect the extension automatically
        if ext == "":
            if self.detect_model_format(path, ".bin"):
                ext = ".bin"
            elif self.detect_model_format(path, ".txt"):
                ext = ".txt"
            else:
                print("Provide model format: '.bin' or '.txt'")
                return

        if ext == ".txt":
            cameras = self.read_cameras_text(os.path.join(path, "cameras" + ext))
            images = self.read_images_text(os.path.join(path, "images" + ext))
            points3D = self.read_points3D_text(os.path.join(path, "points3D") + ext)
        else:
            cameras = self.read_cameras_binary(os.path.join(path, "cameras" + ext))
            images = self.read_images_binary(os.path.join(path, "images" + ext))
            points3D = self.read_points3D_binary(os.path.join(path, "points3D") + ext)
        return cameras, images, points3D
    
    def detect_model_format(self, path, ext):

        import os
        if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
            os.path.isfile(os.path.join(path, "images"   + ext)) and \
            os.path.isfile(os.path.join(path, "points3D" + ext)):
            print("Detected model format: '" + ext + "'")
            return True
        return False

    def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
        import struct
        """pack and write to a binary file.
        :param fid:
        :param data: data to send, if multiple elements are sent at the same time,
        they should be encapsuled either in a list or a tuple
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        should be the same length as the data list or tuple
        :param endian_character: Any of {@, =, <, >, !}
        """
        if isinstance(data, (list, tuple)):
            bytes = struct.pack(endian_character + format_char_sequence, *data)
        else:
            bytes = struct.pack(endian_character + format_char_sequence, data)
        fid.write(bytes)

    def write_points3D_text(self, points3D, path):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DText(const std::string& path)
            void Reconstruction::WritePoints3DText(const std::string& path)
        """
        if len(points3D) == 0:
            mean_track_length = 0
        else:
            mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)

        HEADER = "# 3D point list with one line of data per point:\n" + \
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" + \
                "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)

        with open(path, "w") as fid:
            fid.write(HEADER)
            for _, pt in points3D.items(): 
                point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error] #tuple?
                fid.write(" ".join(map(str, point_header)) + " ")
                track_strings = []
                for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                    track_strings.append(" ".join(map(str, [image_id, point2D])))
                fid.write(" ".join(track_strings) + "\n")

    def write_images_text(self, images, path):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesText(const std::string& path)
            void Reconstruction::WriteImagesText(const std::string& path)
        """
        if len(images) == 0:
            mean_observations = 0
        else:
            mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
        HEADER = "# Image list with two lines of data per image:\n" + \
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
                "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)

        with open(path, "w") as fid:
            fid.write(HEADER)
            for _, img in images.items():
                image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
                first_line = " ".join(map(str, image_header))
                fid.write(first_line + "\n")

                points_strings = []
                for xy, point3D_id in zip(img.xys, img.point3D_ids):
                    points_strings.append(" ".join(map(str, [*xy, point3D_id])))
                fid.write(" ".join(points_strings) + "\n")

    def write_cameras_text(self, cameras, path):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasText(const std::string& path)
            void Reconstruction::ReadCamerasText(const std::string& path)
        """
        HEADER = "# Camera list with one line of data per camera:\n" + \
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
                "# Number of cameras: {}\n".format(len(cameras))
        with open(path, "w") as fid:
            fid.write(HEADER)
            for _, cam in cameras.items():
                to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
                line = " ".join([str(elem) for elem in to_write])
                fid.write(line + "\n")

    def write_model(self, cameras, images, points3D, path, ext=".bin"):
        import os
        if ext == ".txt":
            self.write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
            self.write_images_text(images, os.path.join(path, "images" + ext))
            self.write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
        else:
            self.write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
            self.write_images_binary(images, os.path.join(path, "images" + ext))
            self.write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
        return cameras, images, points3D

    def write_cameras_binary(self, cameras, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasBinary(const std::string& path)
            void Reconstruction::ReadCamerasBinary(const std::string& path)
        """
        with open(path_to_model_file, "wb") as fid:
            self.write_next_bytes(fid, len(cameras), "Q")
            for _, cam in cameras.items():
                model_id = CAMERA_MODEL_NAMES[cam.model].model_id
                camera_properties = [cam.id,
                                    model_id,
                                    cam.width,
                                    cam.height]
                self.write_next_bytes(fid, camera_properties, "iiQQ")
                for p in cam.params:
                    self.write_next_bytes(fid, float(p), "d")
        return cameras

    def write_points3D_binary(self, points3D, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """
        with open(path_to_model_file, "wb") as fid:
            self.write_next_bytes(fid, len(points3D), "Q")
            for _, pt in points3D.items():
                self.write_next_bytes(fid, pt.id, "Q")
                self.write_next_bytes(fid, pt.xyz.tolist(), "ddd")
                self.write_next_bytes(fid, pt.rgb.tolist(), "BBB")
                self.write_next_bytes(fid, pt.error, "d")
                track_length = pt.image_ids.shape[0]
                self.write_next_bytes(fid, track_length, "Q")
                for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                    self.write_next_bytes(fid, [image_id, point2D_id], "ii")

    def write_images_binary(self, images, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesBinary(const std::string& path)
            void Reconstruction::WriteImagesBinary(const std::string& path)
        """
        with open(path_to_model_file, "wb") as fid:
            self.write_next_bytes(fid, len(images), "Q")
            for _, img in images.items():
                self.write_next_bytes(fid, img.id, "i")
                self.write_next_bytes(fid, img.qvec.tolist(), "dddd")
                self.write_next_bytes(fid, img.tvec.tolist(), "ddd")
                self.write_next_bytes(fid, img.camera_id, "i")
                for char in img.name:
                    self.write_next_bytes(fid, char.encode("utf-8"), "c")
                self.write_next_bytes(fid, b"\x00", "c")
                self.write_next_bytes(fid, len(img.point3D_ids), "Q")
                for xy, p3d_id in zip(img.xys, img.point3D_ids):
                    self.write_next_bytes(fid, [*xy, p3d_id], "ddq")

    def read_ply_file(filename):
        '''
        '''
        from plyfile import PlyData, PlyElement
        import numpy as np
        import pandas as pd
        plydata = PlyData.read(filename)
        #print(plydata.elements[0].name)
        #plydata.elements[0].data[0]
        return plydata
        

class validation(readCOLMAPOutput):
    def __init__(self) -> None:
        super().__init__()

    def get_camera_position(self, qvec, tvec):
        '''
        Function to determine camera position in 3D world space
        from quaternion and translation vectors
        INPUT: qvec quaternion, tvec Tanslation
        OUTPUT: 3D camera position 
        '''
        rotmat = qvec2rotmat(qvec)
        camera_position = - np.transpose(rotmat) * tvec
        return camera_position

    def ray_tracing(self, imagesbin_path, points3dbin_path):
        '''
        Function to get cartesian equation of 3D line given point 3D coordinates,
        and translation vector tvec, quaternion vector qvec
        INPUT: path to images.txt colmap output
               path to points3D.txt colmap output
               path to new csv 
        OUTPUT: list of dicts with desired values e.g. camera pose, ray traced vector
                csv output for visual verification - ( problematic at the moment with csv arrays TODO)
        '''
        import pandas as pd
        # Avoid np array truncation, comment this out if debugging
        np.set_printoptions(threshold=sys.maxsize) 

        reader = readCOLMAPOutput()
        geometry = geometry()
        # parse in CLI when not debugging
                 
        images = reader.read_images_binary(imagesbin_path) 
        points3D = reader.read_points3D_binary(points3dbin_path)
        ray_traced_points_lod = [] # initialise a list of dicts (lod)
        for key, value in points3D.items():
            imageid = points3D[key].image_ids[0] #take only one image it appears in hence index 0
            # in the list of dicts images, we are accessing the one with 3d point, id's xys where 
            # the pointidx is the  index of the keypoint. 
            X3D = points3D[key].xyz[0]
            Y3D = points3D[key].xyz[1]
            Z3D = points3D[key].xyz[2]
            X2D = images[points3D[key].image_ids[0]].xys[points3D[key].point2D_idxs[0], 0]
            Y2D = images[points3D[key].image_ids[0]].xys[points3D[key].point2D_idxs[0], 1]
            xy = [X2D, Y2D]
            qvec = images[imageid].qvec
            tvec = images[imageid].tvec 
            intensity = get_pixel_intensity(imageid, X2D, Y2D)
            semantic_label = get_label_from_intensity(intensity, ADE20K_LABEL_NAMES)
            camera_position = self.get_camera_position(qvec, tvec)
            #get direction vectoir between camera and 3d point 
            dvec = geometry.get_line_vector(camera_position[0], camera_position[1],camera_position[2], X3D, Y3D, Z3D) 
            line = {'imageid':imageid, 'xy':xy, 'xyz':points3D[key].xyz,'intensity':intensity,'semantic_label':semantic_label,'camera_position':camera_position,'qvec':qvec, 'tvec':tvec, 'dvec':dvec}
            ray_traced_points_lod.append(line)
                
        return ray_traced_points_lod

    def check_if_plane_intersects(self, line: Line, line_vector, point_on_line, normal_vector, point_on_plane):
        '''
        Function to check if a plane intersects ray traced line 
        between points and camera
        '''
        # TODO maybe change inheritance
        if self.normal_vector.ravel().dot(line.vector.ravel()) !=0:
            return True

    def isect_line_plane_v3(self, p0, p1, p_co, p_no, epsilon=1e-6):
        """
        p0, p1: Define the line.
        p_co, p_no: define the plane:
            p_co Is a point on the plane (plane coordinate).
            p_no Is a normal vector defining the plane direction;
                (does not need to be normalized).

        Return a true  or None (when the intersection can't be found).
        Sourced from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection

        """

        u = self.sub_v3v3(p1, p0)
        dot = self.dot_v3v3(p_no, u)

        if abs(dot) > epsilon:
            # The factor of the point between p0 -> p1 (0 - 1)
            # if 'fac' is between (0 - 1) the point intersects with the segment.
            # Otherwise:
            #  < 0.0: behind p0.
            #  > 1.0: infront of p1.
            w = self.sub_v3v3(p0, p_co)
            fac = -self.dot_v3v3(p_no, w) / dot
            u = self.mul_v3_fl(u, fac)
            return True #self.add_v3v3(p0, u)

        # The segment is parallel to plane.
        return None

    # ----------------------
    # generic maths functions

    def add_v3v3(self,v0, v1):
        return (
            v0[0] + v1[0],
            v0[1] + v1[1],
            v0[2] + v1[2],
        )


    def sub_v3v3(self,v0, v1):
        return (
            v0[0] - v1[0],
            v0[1] - v1[1],
            v0[2] - v1[2],
        )


    def dot_v3v3(self, v0, v1):
        return (
            (v0[0] * v1[0]) +
            (v0[1] * v1[1]) +
            (v0[2] * v1[2])
        )


    def len_squared_v3(self, v0):
        return self.dot_v3v3(v0, v0)


    def mul_v3_fl(self, v0, f):
        return (
            v0[0] * f,
            v0[1] * f,
            v0[2] * f,
        )

    def get_matches_confidence(self):
        '''
        Function to determine the confidence level of the two view
        geometries SIFT matching alg. between two points
        '''
        # TODO
        return None



def ray_tracing_constraint(filename, imagesbin_path, point3dbin_path):
    #import ply
    #get ply normals and vertices
    #get rays from camera to point
    #determine if intersect
    # make dict with flags saying itersect 
    # list erroneous points
    ply_manip = ply_manipulation()
    ply = ply_manip.read_ply(filename) #pc array 
    ply_xyz = ply_manip.read_ply_xyz(filename) #vertices 
    #ply_xyzrgb_normal =  ply_manip.read_ply_xyzrgbnormal(filename) #vertices
    #plane_normal = 
    #point_on_plane = 
    
    validator = validation()
    dict_of_rays = validator.ray_tracing(imagesbin_path, point3dbin_path)
    #{'imageid':imageid, 'xy':xy, 'xyz':points3D[key].xyz,'intensity':intensity,'semantic_label':semantic_label,'camera_position':camera_position,'qvec':qvec, 'tvec':tvec, 'dvec':dvec}
    #check if intersects with plane and plane ==  opaque 
    #normals = ply_manip.compute_normal(ply_xyzrgb_normal, faces)


class utils:
    def get_idx_most_frequent_element(self, List):
        counter = 0
        element = List[0]
        
        for i in List:
            curr_frequency = List.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                element = i
        index = List.index(element)
        return element, index 
        #TODO
        # get index of outlier semantic label
        # remove point from dict dict.list.remove(index)

    def add_column_in_csv(self, input_file, output_file, transform_row):
        '''
        Append a column in existing csv using csv.reader / csv.writer class
        '''
        from csv import writer
        from csv import reader

        # Open the input_file in read mode and output_file in write mode
        with open(input_file, 'r') as read_obj, \
                open(output_file, 'w') as write_obj:

            # Create a csv.reader object from the input file object
            csv_reader = reader(read_obj)

            # Create a csv.writer object from the output file object
            csv_writer = writer(write_obj)

            # Read each row of the input csv file as list
            for row in csv_reader:

                # Pass the list / row in the transform function to add column text for this row
                transform_row(row, csv_reader.line_num)

                # Write the updated row / list to the output file
                csv_writer.writerow(row)

class geometry:
    def get_line_vector(self, x1,y1,z1,x2,y2,z2):
        return [x2-x1, y2-y1, z2-z1]

    def get_intersect(self, line_vector, point_on_line, normal_vector, point_on_plane):
        '''
        Function to get intersect of a 3D line and a plane
        INPUT: line vector, point on line, normal vector of plane, point on plane.
        OUTPUT: Intersect coordinate
        '''
        l1 = Line(line_vector, point_on_line)
        p1 = Plane(normal_vector, point_on_plane)
        return p1.intersect(l1)

    # def get_eq_of_plane(self, p1, p2, p3):
    #     '''
    #     Function to get equation of a plane form 3 points on a plane 
    #     INPUT: p1,p2,p3 3D numpy arrays e.g. p1 = np.array([1, 2, 3])
    #     OUTPUT: equation of plane and vectr normal to plane
    #     '''
    #     import numpy as np
    #     # These two vectors are in the  plane
    #     v1 = p3 - p1
    #     v2 = p2 - p1
    #     # the cross product is a vector normal to the plane
    #     cp = np.cross(v1, v2)
    #     a, b, c = cp
    #     # This evaluates a * x3 + b * y3 + c * z3 which equals d
    #     d = np.dot(cp, p3)
    #     print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    #     return a,b,c,d,cp

    # def compute_normal(xyz, face):
    #     normal = 
    #     return normal
def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = \
        0, 1024, 768, np.array((1024., 512., 384.))
    model2, width2, height2, params2 = \
        2, 1024, 768, np.array((1024., 512., 384., 0.1))

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.
 
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)
    
def extract_keypoints(filename, LABEL_NAMES):
    '''
    Function to extract SIFT keypoints from colmap database sparse reconstruction
    INPUT: Full path to CSV
    OUTPUT: CSV file with imageId, Keypoints X, Keypoints Y, Semantic Label (blank)
    
    '''
    import os
    import argparse
    import csv
    import pandas as pd

    # Avoid np array truncation, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("Database path exists.")
        

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # Read and check keypoints.
 
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 6)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))
    i =1
    # write to csv
    with open(filename, 'w') as f:
        for key in keypoints.keys():
            for i in range(len(keypoints[key])):
                # imageId, X, Y, Semantic LabeL
                INTENSITY = get_pixel_intensity(key,keypoints[key][i,:][0], keypoints[key][i,:][1])
                SEMANTIC_LABEL = get_label_from_intensity(INTENSITY, LABEL_NAMES)
                f.write("%s, %s, %s ,%s, %s\n" % (key, keypoints[key][i,:][0], keypoints[key][i,:][1], INTENSITY, SEMANTIC_LABEL))
                #print("%s, %s, %s ,%s, %s\n" % (key, keypoints[key][i,:][0], keypoints[key][i,:][1], intensity, semanticLabel))
                #f.write("%s, %s, %s, %s\n" % (key, keypoints[key][i,:][0], keypoints[key][i,:][1], None))
                while (i ==1):
                    print("PROCESSING...")
                    i += 1

    assert os.path.exists(filename)

    if os.path.exists(filename):
        print("Written keypoints to .csv successfully.")
    #np.savetxt('keypoints.csv', keypoints, delimiter=',')

    db.close()

    file = pd.read_csv(filename)
    # adding headers
    headerlist = ["imageId", "keypoints X", "keypoints Y", "pixel intensity", "semantic label"]

    # converting data frame to csv
    file.to_csv(filename, header=headerlist, index=False)

def get_pixel_intensity(imageId, X, Y):
    '''
    Function to determine semantic label of colmap identified keypoints using DeepLab 
    semantic segmentation as a reference
    INPUT: csvPath path to csv file that holds keypoint location data
    OUTPUT: append a column of semantic label on to csv file
    '''
    from PIL import Image
    
    im = Image.open(r"/home/joerowelll/COMP0132/brighton_workspace/segmentation/out" + str(imageId) +".png")
    px = im.load()
    #Note: centre of top right pixel is indexed (0.5,0.5), like row,column numpy array.
    
    coordinate = (int(float(X)/1.055), int(float(Y)/1.055)) #convert from string to float, then round as SIFT features have subpixel accuracy.
    #scaled down 5.5% diagonally as colmap scales images.


    pixelIntensity = im.getpixel(coordinate)

    return pixelIntensity

def get_label_from_intensity(intensity, LABEL_NAMES):
    return LABEL_NAMES[intensity]

def extract_two_view_geometries(filename, LABEL_NAMES):
    import os
    import argparse
    import csv
    import pandas as pd

    # Avoid np array truncation in csv writing, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("Database path exists.")
         

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # Read and check matches in two view geometries
    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )
    # write to csv
    with open(filename, 'w') as f:
        for key in matches.keys():
            for i in range(len(matches[key])):
                # imageId, X, Y, Semantic LabeL
                intensity = get_pixel_intensity(key, matches[key][i,:][0], matches[key][i,:][1])
                semanticLabel = get_label_from_intensity(intensity, LABEL_NAMES)
                f.write("%s, %s, %s ,%s, %s\n" % (key, matches[key][i,:][0], matches[key][i,:][1], intensity, semanticLabel))
                
    db.close()
    file = pd.read_csv(filename)
    # adding headers
    headerlist = ["imageId", "matches 1", "matches 2", "pixel intensity", "semantic label"]

    # converting data frame to csv
    file.to_csv(filename, header=headerlist, index=False)

def label_3d_points_text(labelscsv_path, imagestxt_path, points3dtxt_path, LABEL_NAMES):
    import pandas as pd
    import csv
    import os
    import argparse
    '''
    Function to add semantic label to 3d points 
    INPUT: -"filename" Path and filename for output .csv file
           -"points3dpath" Path and filename for POINTS3D.txt file output from colmap. Note .txt not .bin
             3D point list with one line of data per point:
             POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
           - "imagestxtpath" Path and Filename for  Images.txt output from colmap
           - LABEL_NAMES semantic labels 
    OUTPUT: Textfile with appended semantic labelling for each 3d point. Space delimited 
              Note: points are not contiguous
    
    '''
    # Avoid np array truncation, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 
    #Any variable name ending with *_idx should be considered as an ordered, contiguous 
    # zero-based index. In general, any variable name ending with *_id should be 
    # considered as an unordered, non-contiguous identifier.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--imagestxt_path", default="images.txt")
    # parser.add_argument("--points3dtxt_path", default="points3D.txt")

    # if os.path.exists(args.imagestxt_path):
    #     print("Image path exists.")
    # if os.path.exists(args.points3dtxt_path):
    #     print("Points 3D path exists.")
    
    # args = parser.parse_args()
    points_labelled_lod = []
    reader = readCOLMAPOutput()
    images = reader.read_images_text(imagestxt_path)
    points3D = reader.read_points3D_text(points3dtxt_path)

    #print(images[points3D[330130].image_ids[0]].xys)
    #print(points3D[330130].xyz[1])
    with open(labelscsv_path, 'w') as f:
        for key, value in points3D.items():
            imageid = points3D[key].image_ids[0] #take only one image it appears in hence index 0
            # in the list of dicts images, we are accessing the one with 3d point, id's xys where the pointidx is the index of the keypoint.
            X2D = images[points3D[key].image_ids[0]].xys[points3D[key].point2D_idxs[0], 0]
            Y2D = images[points3D[key].image_ids[0]].xys[points3D[key].point2D_idxs[0], 1]
            #print(X2d)
            X3D = points3D[key].xyz[0]
            Y3D = points3D[key].xyz[1]
            Z3D = points3D[key].xyz[0]
            INTENSITY = get_pixel_intensity(imageid, X2D, Y2D)
            SEMANTIC_LABEL = get_label_from_intensity(INTENSITY, LABEL_NAMES)
            points_labelled = {'imageid':imageid, 'X2D':X2D, 'Y2D':Y2D, 'X3D':X3D, 'Y3D':Y3D,'Z3D':Z3D, 'intensity':INTENSITY,'semantic_label':SEMANTIC_LABEL}
            points_labelled_lod.append(points_labelled)
            # Write a row to the csv file
            f.write("%s, %s, %s ,%s, %s, %s, %s, %s\n" % (imageid, X2D, Y2D, X3D, Y3D, Z3D, INTENSITY, SEMANTIC_LABEL))
    file = pd.read_csv(labelscsv_path)
    # adding headers to csv file
    headerlist = ["imageid","X2D","Y2D","X3D","Y3D", "Z3D", "INTENSITY", "SEMANTIC_LABEL"]

    # converting data frame to csv
    file.to_csv(labelscsv_path, header=headerlist, index=False)
    print("Successfully semantically labelled 3D points")

def label_3d_points_binary(labelscsv_path, imagesbin_path, points3dbin_path, LABEL_NAMES):
    import pandas as pd
    import csv
    import os
    import argparse
    '''
    Function to add semantic label to 3d points 
    INPUT: -"filename" Path and filename for output .csv file
           -"points3dpath" Path and filename for POINTS3D.txt file output from colmap. Note .txt not .bin
             3D point list with one line of data per point:
             POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
           - "imagestxtpath" Path and Filename for  Images.txt output from colmap
           - LABEL_NAMES semantic labels 
    OUTPUT: Textfile with appended semantic labelling for each 3d point. Space delimited 
              Note: points are not contiguous
    
    '''
    # Avoid np array truncation, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 
    #Any variable name ending with *_idx should be considered as an ordered, contiguous 
    # zero-based index. In general, any variable name ending with *_id should be 
    # considered as an unordered, non-contiguous identifier.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--imagesbin_path", default="images.bin")
    # parser.add_argument("--points3dbin_path", default="points3D.bin")

    # if os.path.exists(args.imagesbin_path):
    #     print("Image binary path exists.")
    # if os.path.exists(args.points3dbin_path):
    #      print("Points 3D binary path exists.") 
    # args = parser.parse_args()

    points_labelled_lod = [] #initisliase list of dicts (lod)
    reader = readCOLMAPOutput()
    images = reader.read_images_binary(imagesbin_path)
    points3D = reader.read_points3D_binary(points3dbin_path)

    with open(labelscsv_path, 'w') as f:
        for key, value in points3D.items():
            imageid = points3D[key].image_ids[0] #take only one image it appears in hence index 0
            # in the list of dicts images, we are accessing the one with 3d point, id's xys where the pointidx is the index of the keypoint.
            X2D = images[points3D[key].image_ids[0]].xys[points3D[key].point2D_idxs[0], 0]
            Y2D = images[points3D[key].image_ids[0]].xys[points3D[key].point2D_idxs[0], 1]
            X3D, Y3D, Z3D = points3D[key].xyz[0], points3D[key].xyz[1], points3D[key].xyz[0]
            
            INTENSITY = get_pixel_intensity(imageid, X2D, Y2D)
            SEMANTIC_LABEL = get_label_from_intensity(INTENSITY, LABEL_NAMES)
            points_labelled = {'imageid':imageid, 'X2D':X2D, 'Y2D':Y2D, 'X3D':X3D, 'Y3D':Y3D,'Z3D':Z3D, 'intensity':INTENSITY,'semantic_label':SEMANTIC_LABEL}
            points_labelled_lod.append(points_labelled)
            # Write a row to the csv file
            f.write("%s, %s, %s ,%s, %s, %s, %s, %s\n" % (imageid, X2D, Y2D, X3D, Y3D, Z3D, INTENSITY, SEMANTIC_LABEL))
    file = pd.read_csv(labelscsv_path)
    # adding headers to csv file
    headerlist = ["imageid","X2D","Y2D","X3D","Y3D", "Y3D", "INTENSITY", "SEMANTIC_LABEL"]

    # converting data frame to csv
    file.to_csv(labelscsv_path, header=headerlist, index=False)
    print("Successfully semantically labelled 3D points")

def semantic_consistency_check(imagestxt_path, points3dtxt_path, images_out_path, points3D_out_path):
    '''
    Function to compare semantic labels of 3d points output from colmap sparse reocnstruction
    and delete observations and/or points that do not have consistent semantic labels.
    INPUT: path to images.txt colmap output
           path to points3D.txt colmap output
           path to new points3D output, without file suffix
               
    OUTPUT: new images.txt output
            new corrected points3D.txt, .bin output
    '''
    print("Semantic consistency check in progress...")
    import warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    # Avoid np array truncation, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 

    reader = readCOLMAPOutput()
    validator = validation()
    utility = utils()
    # parse in CLI when not debugging
                 
    #images = reader.read_images_text(imagestxt_path) 
    images = reader.read_images_binary(imagestxt_path)
    points3D = reader.read_points3D_binary(points3dtxt_path)
    points_lod = [] # initialise a list of dicts (lod)
    X2D = []
    Y2D = []
    qvec = []
    tvec = []
    
    semantic_label = []
    images_out = []
    points3D_out = {}
    pt1 = []
    total_observations = 0
    
    for key, value in points3D.items():
        imageid = points3D[key].image_ids #take only one image it appears in hence index 0
        # in the list of dicts images, we are accessing the one with 3d point, id's xys where 
        # the pointidx is the index of the keypoint.
        #print(len(imageid))
        intensity = []
        for i in range(len(imageid)):
            X2D = images[points3D[key].image_ids[i]].xys[points3D[key].point2D_idxs[i], 0] #0?
            Y2D = images[points3D[key].image_ids[i]].xys[points3D[key].point2D_idxs[i], 1]
            xy2d = images[points3D[key].image_ids[i]].xys[points3D[key].point2D_idxs[i],:]
            qvec = images[imageid[i]].qvec
            tvec = images[imageid[i]].tvec 
            camera_id = images[imageid[i]].camera_id ##
            name = images[imageid[i]].name           ##
            intensity_ = get_pixel_intensity(imageid[i],X2D,Y2D) 
            intensity.append(intensity_)
            most_common_intensity, idx = utility.get_idx_most_frequent_element(intensity)

            semantic_label_ = get_label_from_intensity(intensity_, ADE20K_LABEL_NAMES)
            semantic_label.append(semantic_label_)
            
            #camera_position = validation.get_camera_position(qvec, tvec)
            # append to some lod, and compare all imageid intensity matches
            X3D = points3D[key].xyz[0]
            Y3D = points3D[key].xyz[1]
            Z3D = points3D[key].xyz[2]
            xyz3d = points3D[key].xyz
            row = {'imageid':imageid, 'xy2d':xy2d , 'xyz3d':xyz3d , 'intensity':intensity, 'semantic labels':semantic_label} #list of dicts of arrays  
            #print(row['imageid'])   
            points_lod.append(row)
            #print(row)
            pt1 = Point3D(id=points3D[key].id, xyz=xyz3d, rgb=points3D[key].rgb,
                    error=points3D[key].error, image_ids=points3D[key].image_ids,
                    point2D_idxs=points3D[key].point2D_idxs)
            
            img = {'id':imageid, 'qvec':qvec, 'tvec':tvec, 'camera_id':camera_id, 'name':name }
            pt = {'id':points3D[key].id, 'xyz':xyz3d, 'rgb':points3D[key].rgb, 'error':points3D[key].error, 'image_ids':points3D[key].image_ids,
                    'point2D_idxs':points3D[key].point2D_idxs}
            
            for j in range(len(points_lod[i]['intensity'])):
                if (points_lod[i]['intensity'][j] != most_common_intensity):
            
                    indx = np.where(pt1.image_ids == imageid)
        

                    image_ids_corrected = np.delete(points3D[key].image_ids, indx)
                    point2D_idxs_corrected = np.delete(points3D[key].point2D_idxs, indx)
                    pt1 = Point3D(id=points3D[key].id, xyz=xyz3d, rgb=points3D[key].rgb,
                    error=points3D[key].error, image_ids=image_ids_corrected,
                    point2D_idxs=points3D[key].point2D_idxs)
                    #print(points3D[key].point2D_idxs) # should i change  and delete these ?
    
                if (len(pt1.image_ids) > 2):
                    total_observations +=len(pt1.image_ids)
                    pt1 = Point3D(id=points3D[key].id, xyz=xyz3d, rgb=points3D[key].rgb,
                    error=points3D[key].error, image_ids=pt1.image_ids,
                    point2D_idxs=points3D[key].point2D_idxs)
                    points3D_out[key] = pt1 
    print("Total Observations: ", total_observations, "\n" )
    print("Mean Observations per image: ", total_observations /1102)# this is hardcoded! change this 
    reader.write_points3D_text(points3D_out, points3D_out_path + '.txt')  
    #reader.write_points3D_binary(points3D_out, points3D_out_path + '.bin')
    # dict of collections (accessible like an object), where key is point number ( key doesnt matter)
                    
            # if intensity of any point is not the same between images, (is not the average if there are multiple observations
            # ), then remove that imageid corresponding to matched point, and remove point entirely if point observed in two images only. 
    #reader.write_points3D_text(points3D_out, points3D_out_path)
    
    return points_lod

def remove_dynamic_points(imagestxt_path, points3dtxt_path, points3D_out_path, LABEL_NAMES):
    '''
    Function to compare semantic labels of 3d points output from colmap sparse reocnstruction
    and delete observations and/or points that do not have consistent semantic labels.
    INPUT: path to images.txt colmap output
           path to points3D.txt colmap output
           path to new points3D output, without file suffix
               
    OUTPUT: new images.txt output
            new corrected points3D.txt, .bin output
    '''
    print("Removing dynamic feature points in progress...")
    import warnings
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    # Avoid np array truncation, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 

    reader = readCOLMAPOutput()
    validator = validation()
    utility = utils()
    # parse in CLI when not debugging
                 
    #images = reader.read_images_text(imagestxt_path) 
    images = reader.read_images_binary(imagestxt_path)
    points3D = reader.read_points3D_binary(points3dtxt_path)
    points_lod = [] # initialise a list of dicts (lod)
    X2D = []
    Y2D = []
    qvec = []
    tvec = []
    
    semantic_label = []
    points3D_out = {}
    pt1 = []
    total_observations = 0
    total_points = 0

    sky_indx = 10#np.where(LABEL_NAMES == 'sky')
    car_indx = 13#np.where(LABEL_NAMES == 'car')
    person_indx = 11#np.where(LABEL_NAMES == 'person')
    rider_indx = 12#np.where(LABEL_NAMES=='rider')
    truck_indx = 14#np.where(LABEL_NAMES== 'truck')
    bus_indx = 15#np.where(LABEL_NAMES =='bus') 
    train_indx = 16#np.where(LABEL_NAMES == 'train')
    motorcycle_indx = 17#np.where(LABEL_NAMES=='motorcycle')
    bicycle_indx = 18#np.where(LABEL_NAMES== 'bicycle')

    number_of_images = 1102 #len(images.IMAGE_ID)
    for key, value in points3D.items():
        imageid = points3D[key].image_ids #take only one image it appears in hence index 0
        # in the list of dicts images, we are accessing the one with 3d point, id's xys where 
        # the pointidx is the index of the keypoint.
        intensity = []
        for i in range(len(imageid)):
            X2D = images[points3D[key].image_ids[i]].xys[points3D[key].point2D_idxs[i], 0] #0?
            Y2D = images[points3D[key].image_ids[i]].xys[points3D[key].point2D_idxs[i], 1]
            xy2d = images[points3D[key].image_ids[i]].xys[points3D[key].point2D_idxs[i],:]
            qvec = images[imageid[i]].qvec
            tvec = images[imageid[i]].tvec 
            camera_id = images[imageid[i]].camera_id ##
            name = images[imageid[i]].name           ##
            intensity_ = get_pixel_intensity(imageid[i],X2D,Y2D) 
            intensity.append(intensity_)
            most_common_intensity, idx = utility.get_idx_most_frequent_element(intensity)

            semantic_label_ = get_label_from_intensity(intensity, ADE20K_LABEL_NAMES)
            semantic_label.append(semantic_label_)
                        
            X3D = points3D[key].xyz[0]
            Y3D = points3D[key].xyz[1]
            Z3D = points3D[key].xyz[2]
            xyz3d = points3D[key].xyz
            row = {'imageid':imageid, 'xy2d':xy2d , 'xyz3d':xyz3d , 'intensity':intensity, 'semantic labels':semantic_label} #list of dicts of arrays  
            
            points_lod.append(row)
            
            img = {'id':imageid, 'qvec':qvec, 'tvec':tvec, 'camera_id':camera_id, 'name':name }
            pt = {'id':points3D[key].id, 'xyz':xyz3d, 'rgb':points3D[key].rgb, 'error':points3D[key].error, 'image_ids':points3D[key].image_ids,
                    'point2D_idxs':points3D[key].point2D_idxs}
            
            #for j in range(len(pt1.image_ids)):
                # if (points_lod[i]['intensity'] == sky_indx or points_lod[i]['intensity'] == car_indx or 
                # points_lod[i]['intensity'] == person_indx or points_lod[i]['intensity'] == rider_indx or 
                # points_lod[i]['intensity'] == bus_indx or points_lod[i]['intensity'] == truck_indx or 
                # points_lod[i]['intensity'] == train_indx or points_lod[i]['intensity'] == motorcycle_indx or 
                # points_lod[i]['intensity'] == bicycle_indx):
        #print(sky_indx)
            #print(most_common_intensity)
        
        # if (most_common_intensity != sky_indx or most_common_intensity !=  car_indx or 
        # most_common_intensity !=  person_indx or most_common_intensity != rider_indx or 
        # most_common_intensity !=  bus_indx or most_common_intensity !=  truck_indx or 
        # most_common_intensity !=  train_indx or most_common_intensity != motorcycle_indx or 
        # most_common_intensity !=  bicycle_indx):
            if most_common_intensity<=10:
            
                    #indx = np.where(pt1.image_ids == points_lod[i]["imageid"])
                    
            #image_ids_corrected = np.delete(points3D[key].image_ids, j) # j or indx?
                    #if (len(image_ids_corrected) > 2):
                        #point2D_idxs_corrected = np.delete(points3D[key].point2D_idxs, j)

                        # pt1 = Point3D(id=points3D[key].id, xyz=xyz3d, rgb=points3D[key].rgb,
                        # error=points3D[key].error, image_ids=image_ids_corrected,
                        # point2D_idxs=point2D_idxs_corrected)

                
                #total_observations +=len(pt1.image_ids)
                total_points += 1
                pt1 = Point3D(id=points3D[key].id, xyz=xyz3d, rgb=points3D[key].rgb,
                    error=points3D[key].error, image_ids=points3D[key].image_ids,
                    point2D_idxs=points3D[key].point2D_idxs)
                points3D_out[key] = pt1 
                
                # else:
                #     pt1 = Point3D(id=points3D[key].id, xyz=xyz3d, rgb=points3D[key].rgb,
                #         error=points3D[key].error, image_ids=pt1.image_ids,
                #         point2D_idxs=points3D[key].point2D_idxs)
                #     points3D_out[key] = pt1 

                
    print("Total number of observations: ", total_points, "\n")
    #print("Total Observations: ", total_observations, "\n" )
    #print("Mean Observations per image: ", total_observations / number_of_images)# this is hardcoded! change this 
    reader.write_points3D_text(points3D_out, points3D_out_path + '.txt') 
    return points_lod

def run_colmap(DATASET_PATH):

    import subprocess
  
  
    # If your shell script has shebang, 
    # you can omit shell=True argument.
    subprocess.run(["~/COMP0132/COMP0132/structureFromMotion/do_it_once.sh", 
                    DATASET_PATH], shell=True)


if __name__ == "__main__":
    import argparse
    from plane_intersection import Plane, Line

    parser = argparse.ArgumentParser(description="Semantically Label 3D Points")
    parser.add_argument("--images_path", help="path to colmap output Image file", default = "images")
    parser.add_argument("--images_format", choices=[".bin", ".txt"],
                        help="points3D model format (binary .bin or text .txt)", default=".txt")
    parser.add_argument("--points3d_path",
                        help="path to colmap output points3d file", default = "points3D")
    parser.add_argument("--points3d_format", choices=[".bin", ".txt"],
                        help="points3D model format (binary or text)", default=".txt")
    parser.add_argument("--outputcsv_path",
                        help="path to labelled 3D points .csv file", default="labelled3DPoints.csv")
    parser.add_argument("--database_path",
                        help="path to colmap output database.db", default="database.db")
    args = parser.parse_args()
    images_path = args.images_path + args.images_format
    points3d_path = args.points3d_path + args.points3d_format
    database_path = args.database_path
    outputCSV_path = args.outputcsv_path

    #extract_keypoints("/home/joerowelll/COMP0132/brighton_workspace/brightonKeypoints.csv", LABEL_NAMES)

    #label_3d_points_text("/home/joerowelll/COMP0132/brighton_text/Model_configuration/labelled3Dpoints2.csv",
    #                     "/home/joerowelll/COMP0132/brighton_text/Model_configuration/images.txt", 
    #                     "/home/joerowelll/COMP0132/brighton_text/Model_configuration/points3D.txt",     
    #                     ADE20K_LABEL_NAMES)
    #######################################################################################################
    
    # validator = validation()
    # ray_traced_points_lod = validator.ray_tracing("/home/joerowelll/COMP0132/brighton_text/Model_configuration/images.txt",
    #                                                "/home/joerowelll/COMP0132/brighton_text/Model_configuration/points3D.txt", 
    #                                                "/home/joerowelll/COMP0132/brighton_text/Model_configuration/rayTracing.csv")

    # points_lod2 = semantic_consistency_check("/home/joerowelll/COMP0132/brighton_text/Model_configuration/images.bin",
    #                                         "/home/joerowelll/COMP0132/brighton_text/Model_configuration/points3D.bin",
    #                                         "/home/joerowelll/COMP0132/brighton_text/corrected_model/images",
    #                                         "/home/joerowelll/COMP0132/brighton_text/corrected_model_2/points3D")
    

    # points_lod3 = semantic_consistency_check("/home/joerowelll/COMP0132/colmap_projects/20220813_092312/original/sparse/images.bin",
    #                                          "/home/joerowelll/COMP0132/colmap_projects/20220813_092312/original/sparse/points3D.bin",
    #                                          "/home/joerowelll/COMP0132/colmap_projects/20220813_092312/corrected/test/images",
    #                                          "/home/joerowelll/COMP0132/colmap_projects/20220813_092312/corrected/test/points3D")


    #points_lod4 = semantic_consistency_check("/home/joerowelll/COMP0132/colmap_projects/20220823_211641/original/sparse/images.bin",
                                            #  "/home/joerowelll/COMP0132/colmap_projects/20220823_211641/original/sparse/points3D.bin",
                                            #  "/home/joerowelll/COMP0132/colmap_projects/20220823_211641/corrected/sparse/images",
                                            #  "/home/joerowelll/COMP0132/colmap_projects/20220823_211641/corrected/sparse/points3D")



    #points_lod5 = semantic_consistency_check("/home/joerowelll/COMP0132/colmap_projects/20220828_163731/original/sparse/images.bin",
                                            #  "/home/joerowelll/COMP0132/colmap_projects/20220828_163731/original/sparse/points3D.bin",
                                            #  "/home/joerowelll/COMP0132/colmap_projects/20220828_163731/corrected/sparse/images",
                                            #  "/home/joerowelll/COMP0132/colmap_projects/20220828_163731/corrected/sparse/points3D")

    # points_lod_no_dynamic = remove_dynamic_points("/home/joerowelll/COMP0132/colmap_projects/20220828_163731/original/sparse/images.bin",
    #                           "/home/joerowelll/COMP0132/colmap_projects/20220828_163731/original/sparse/points3D.bin", 
    #                           "/home/joerowelll/COMP0132/colmap_projects/20220828_163731/no_dynamic/sparse/points3D", 
    #                           ADE20K_LABEL_NAMES)

    points_lod_no_dynamic2 = remove_dynamic_points("/home/joerowelll/COMP0132/colmap_projects/20220813_092312/original/sparse/images.bin",
                              "/home/joerowelll/COMP0132/colmap_projects/20220813_092312/original/sparse/points3D.bin", 
                              "/home/joerowelll/COMP0132/colmap_projects/20220813_092312/no_dynamic/sparse/points3D", 
                              ADE20K_LABEL_NAMES)

    points_lod_no_dynamic3 = remove_dynamic_points("/home/joerowelll/COMP0132/colmap_projects/20220823_211641/original/sparse/images.bin",
                              "/home/joerowelll/COMP0132/colmap_projects/20220823_211641/original/sparse/points3D.bin", 
                              "/home/joerowelll/COMP0132/colmap_projects/20220823_211641/no_dynamic/sparse/points3D", 
                              ADE20K_LABEL_NAMES)

    points_lod_no_dynamic4 = remove_dynamic_points("/home/joerowelll/COMP0132/brighton_text/Model_configuration/images.bin",
                                             "/home/joerowelll/COMP0132/brighton_text/Model_configuration/points3D.bin",
                                             "/home/joerowelll/COMP0132/brighton_text/no_dynamic/points3D",
                                             ADE20K_LABEL_NAMES)
    

    # reader = readCOLMAPOutput()
    # validator = validation()
    # plydata = reader.read_ply_file("/home/joerowelll/Downloads/fused.ply")
    # cameras = reader.read_cameras_binary()
    # points = reader.read_points3D_binary()
    
    # for i in range(len(cameras)):
    #     for j in range(len(plydata['face'].data['vertex_indices'])):
    #         face = plydata['face'].data['vertex_indices'][j]
    #         point = points[i]
    #         camera = cameras[i]
    #         point_xyz = point.xyz
    #         camera_xyz = camera.xyz
    #         line_vector = get_line_vector(camera_xyz, point_xyz)
            #get label of plane 
            # if plane label == wall
                #intersects = validator.check_if_plane_intersects(line: Line, line_vector, cameraxyz, normal_vector, point_on_plane)
                #if intersects:
                #   remove point in points...


    
    #       Get camera positions DONE 
    #       Get equation of line between labelled 3D points and Camera Positions DONE
    #  TODO Import dense reocnstruction mesh DONE
    #       Perform planar reocnstutcion Delaunay- DONE
    #       Import plane                           DONE Ish
    #       Determine if plane with label "wall" or "building" 
    #       If plane intersects ray tracing, flag as problematic 
    #       Get cost of problematic points
    #       Try model of points with next smallest cost /highest confidence 

#######################################################################################################



# EXAMPLE USAGE for labelling 3D points from text colmap output (if using argsparse):
#$ python database.py  --database_path ~/COMP0132/brighton_workspace/database.db,
#    --imagestxt_path "/home/joerowelll/COMP0132/brighton_text/Model_configuration/images.txt", 
#    --points3dtxt_path "/home/joerowelll/COMP0132/brighton_text/Model_configuration/points3D.txt" 


# EXAMPLE USAGE for labelling 3D points from binary colmap output:
#$ python database.py  --database_path ~/COMP0132/brighton_workspace/database.db,
#    --imagesbin_path "/home/joerowelll/COMP0132/brighton_text/Model_configuration/images.bin", 
#    --points3dbin_path "/home/joerowelll/COMP0132/brighton_text/Model_configuration/points3D.bin" 


