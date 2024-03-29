import torch.utils.data as data
import torch
import numpy as np
import os
import sys
from PIL import Image
from plyfile import PlyData, PlyElement
from typing import NamedTuple
from lib.utils.colmap import read_write_model
from lib.utils import graphics_utils, camera_utils
from lib.config import cfg
from lib.networks.gs.gs import GaussianModel
import imageio
import json
import cv2
import random


import matplotlib.pyplot as plt
plt.switch_backend('agg')


def trans_t(t):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=np.float32)

def rot_phi(phi):
    return np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1],
    ], dtype=np.float32)

def rot_theta(th) :
    return np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1],
    ], dtype=np.float32)

def pose_spherical(theta, phi, radius):
    """
    Input:
        @theta: [-180, +180]，间隔为 9
        @phi: 固定值 -30
        @radius: 固定值 4
    Output:
        @c2w: 从相机坐标系到世界坐标系的变换矩阵
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: graphics_utils.BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(read_write_model.qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = graphics_utils.focal2fov(focal_length_x, height)
            FovX = graphics_utils.focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = graphics_utils.focal2fov(focal_length_y, height)
            FovX = graphics_utils.focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = graphics_utils.getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return graphics_utils.BasicPointCloud(points=positions, colors=colors, normals=normals)

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        """
        Description:
            __init__ 函数负责从磁盘中 load 指定格式的文件，计算并存储为特定形式

        Input:
            @kwargs: 读取的参数
        Output:
            None
        """
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.data_root = os.path.join(data_root, scene)
        self.input_ratio = kwargs['input_ratio']
        self.llffhold = kwargs['llffhold']
        self.load_iteration = kwargs['load_iteration']
        self.shuffle = kwargs['shuffle']
        self.split = split # train or test
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.num_iter_train = 0
        if split == 'train':
            self.eval = kwargs['eval']
        else:
            self.eval = True
        # cams = kwargs['cams']
        self.resolution_scales = kwargs['resolution_scales']
        self.loaded_iter = None
        
        if self.load_iteration != 0:
            if self.load_iteration == -1:
                # load the max iteration
                pass
                # self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else :
                self.loaded_iter = self.load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        scene_info = self.read_colmap_cameras()

        if not self.loaded_iter:
            if not os.path.exists(cfg.trained_model_dir):
                os.makedirs(cfg.trained_model_dir)
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(cfg.trained_model_dir, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_utils.camera_to_JSON(id, cam))
            with open(os.path.join(cfg.trained_model_dir, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = camera_utils.cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, kwargs)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = camera_utils.cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, kwargs)

        # if self.loaded_iter:
        #     # TODO: create gs
        #     pass
        #     # self.gaussians.load_ply(os.path.join(self.model_path,
        #     #                                                "point_cloud",
        #     #                                                "iteration_" + str(self.loaded_iter),
        #     #                                                "point_cloud.ply"))
        # else:
        self.gaussians = GaussianModel()
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        self.gaussians.save_ply(os.path.join(cfg.trained_model_dir, "point_cloud.ply"))
        del self.gaussians

    def __getitem__(self, index):
        """
        Description:
            __getitem__ 函数负责在运行时提供给网络一次训练需要的输入，以及 ground truth 的输出
        对 NeRF 来说，分别是 1024 条光线以及 1024 个 RGB值
        2dgs: camera,img,point_cloud

        Input:
            @index: 图像下标, 范围为 [0, len-1]
        Output:
            @ret: 包含所需数据的字典(添加 'meta' 用于 evaluate)
        """

        batch={}
        if self.split == 'train':
            batch['FoVx'] = self.train_cameras[self.resolution_scales[0]][index].FoVx
            batch['FoVy'] = self.train_cameras[self.resolution_scales[0]][index].FoVy
            batch['image_height'] = self.train_cameras[self.resolution_scales[0]][index].image_height
            batch['image_width'] = self.train_cameras[self.resolution_scales[0]][index].image_width
            batch['world_view_transform'] = self.train_cameras[self.resolution_scales[0]][index].world_view_transform
            batch['full_proj_transform'] = self.train_cameras[self.resolution_scales[0]][index].full_proj_transform
            batch['camera_center'] = self.train_cameras[self.resolution_scales[0]][index].camera_center
            batch['original_image'] = self.train_cameras[self.resolution_scales[0]][index].original_image
            return batch
        else:
            batch['FoVx'] = self.test_cameras[self.resolution_scales[0]][index].FoVx
            batch['FoVy'] = self.test_cameras[self.resolution_scales[0]][index].FoVy
            batch['image_height'] = self.test_cameras[self.resolution_scales[0]][index].image_height
            batch['image_width'] = self.test_cameras[self.resolution_scales[0]][index].image_width
            batch['world_view_transform'] = self.test_cameras[self.resolution_scales[0]][index].world_view_transform
            batch['full_proj_transform'] = self.test_cameras[self.resolution_scales[0]][index].full_proj_transform
            batch['camera_center'] = self.test_cameras[self.resolution_scales[0]][index].camera_center
            batch['original_image'] = self.test_cameras[self.resolution_scales[0]][index].original_image
            return batch



    def __len__(self):
        """
        Description:
            __len__ 函数返回训练或者测试的数量

        Input:
            None
        Output:
            @len: 训练或者测试的数量
        """
        if self.split == 'train':
            return len(self.train_cameras[self.resolution_scales[0]])
        else:
            return len(self.test_cameras[self.resolution_scales[0]])


    def read_colmap_cameras(self):
        cameras_extrinsic_file = os.path.join(self.data_root, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(self.data_root, "sparse/0", "cameras.bin")
        cam_extrinsics = read_write_model.read_images_binary(cameras_extrinsic_file)
        cam_intrinsics = read_write_model.read_cameras_binary(cameras_intrinsic_file)

        # reading_dir = "images" if images == None else images
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(self.data_root, "images"))
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

        if self.eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % self.llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % self.llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

        nerf_normalization = getNerfppNorm(train_cam_infos)

        ply_path = os.path.join(self.data_root, "sparse/0/points3D.ply")
        bin_path = os.path.join(self.data_root, "sparse/0/points3D.bin")
        txt_path = os.path.join(self.data_root, "sparse/0/points3D.txt")
        xyzs, rgbs = [], []
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyzs, rgbs, _ = read_write_model.read_points3D_binary(bin_path)
            except:
                points3D = read_write_model.read_points3D_text(txt_path)
            #TODO: to be removed or modified to array
            # for _,value in points3D.items():
            #     xyzs.append(value.xyz)
            #     rgbs.append(value.rgb)
            storePly(ply_path, xyzs, rgbs)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

        scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
        return scene_info

    def get_rays(self, H, W, K, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
        i, j = i.t(), j.t()
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d


    def get_render_rays(self):
        self.render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
        self.render_poses = torch.from_numpy(self.render_poses)
        render_rays_o, render_rays_d = [], []

        for i in range(self.render_poses.shape[0]):
            render_ray_o, render_ray_d = self.get_rays(self.H, self.W, self.K, self.render_poses[i, :3, :4])
            render_rays_o.append(render_ray_o)   # (H, W, 3)
            render_rays_d.append(render_ray_d)   # (H, W, 3)

        render_rays_o = torch.stack(render_rays_o)                    # (40, H, W, 3)
        render_rays_d = torch.stack(render_rays_d)                    # (40, H, W, 3)
        return render_rays_o, render_rays_d
