from dataclasses import dataclass
import numpy as np
import cv2
import os

import tqdm

from datasets.dataset import *
from icecream import ic



@dataclass
class Intrinsic:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: np.ndarray

class USBCameraDataset(Dataset):

    def __init__(self, args, device):
        super().__init__("USBCamera", args, device)
        self.parse_metadata()

    def parse_metadata(self):
        # Configure depth and color streams
        self.cap = cv2.VideoCapture(0)

        # Get device product line for setting a supporting resolution
        if not self.cap.isOpened():
            raise NotImplementedError("No RGB camera found")

        self.rate_hz = 30
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Set timestamp
        self.timestamp = 0

        # Start streaming
        ic("Start streaming")
        intrinsics = Intrinsic(
            width= 640,
            height= 480,
            fx= 1.87956286e+03,
            fy= 1.85002154e+03,
            ppx= 1.26960281e+03,
            ppy= 4.70476746e+02,
            model= 'radtan',
            coeffs= np.array([0, 0, 0, 0, 0])
        )
        self.calib = self._get_cam_calib(intrinsics)

        self.resize_images = True
        if self.resize_images:
            self.output_image_size = [315, 420] # h, w 
            h0, w0  = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

    def _get_cam_calib(self, intrinsics: Intrinsic):
        """ intrinsics: 
            model	Distortion model of the image
            coeffs	Distortion coefficients
            fx	    Focal length of the image plane, as a multiple of pixel width
            fy	    Focal length of the image plane, as a multiple of pixel height
            ppx	    Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
            ppy	    Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
            height	Height of the image in pixels
            width	Width of the image in pixels
        """
        w, h = intrinsics.width, intrinsics.height
        fx, fy, cx, cy= intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

        distortion_coeffs = intrinsics.coeffs
        distortion_model  = intrinsics.model
        k1, k2, p1, p2 = 0, 0, 0, 0
        body_T_cam0 = np.eye(4)
        rate_hz = self.rate_hz

        resolution  = Resolution(w, h)
        pinhole0    = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)

        aabb        = (2*np.array([[-2, -2, -2], [2, 2, 2]])).tolist() # Computed automatically in to_nerf()
        depth_scale = 1.0 # TODO # Since we multiply as gt_depth *= depth_scale, we need to invert camera["scale"]

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)


    def stream(self):
        self.viz=True

        timestamps = []
        poses      = []
        images     = []
        depths     = []
        calibs     = []

        got_image = False
        while not got_image:
            # Wait for a coherent pair of frames: depth and color
            ret, color_frame = self.cap.read()


            if color_frame is None:
                print("No color frame parsed.")
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame)


            if self.resize_images:
                color_image = cv2.resize(color_image, (self.w1, self.h1))

            if self.viz:
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                cv2.imshow(f"Color Img", color_image)
                cv2.waitKey(1)

            self.timestamp += 1
            if self.args.img_stride > 1 and self.timestamp % self.args.img_stride == 0:
                # Software imposed fps to rate_hz/img_stride
                continue

            timestamps += [self.timestamp]
            poses      += [np.eye(4)] # We don't have poses
            images     += [color_image]
            depths     += [color_image] # We don't use depth
            calibs     += [self.calib]
            got_image  = True

        return {"k":      np.arange(self.timestamp-1,self.timestamp),
                "t_cams": np.array(timestamps),
                "poses":  np.array(poses),
                "images": np.array(images),
                "depths": [None],
                "calibs": np.array(calibs),
                "is_last_frame": False, #TODO
                }
    
    def shutdown(self):
        # Stop streaming
        self.cap.release()
        cv2.destroyAllWindows()

    def to_nerf_format(self, data_packets):
        print("Exporting Camera dataset to Nerf")
        OUT_PATH = "transforms.json"
        AABB_SCALE = 4
        out = {
            "fl_x": self.calib.camera_model.fx,
            "fl_y": self.calib.camera_model.fy,
            "k1": self.calib.distortion_model.k1,
            "k2": self.calib.distortion_model.k2,
            "p1": self.calib.distortion_model.p1,
            "p2": self.calib.distortion_model.p2,
            "cx": self.calib.camera_model.cx,
            "cy": self.calib.camera_model.cy,
            "w": self.calib.resolution.width,
            "h": self.calib.resolution.height,
            "aabb": self.calib.aabb,
            "aabb_scale": AABB_SCALE,
            "integer_depth_scale": self.calib.depth_scale,
            "frames": [],
        }

        from PIL import Image

        c2w = np.eye(4).tolist()
        for data_packet in tqdm(data_packets):
            # Image
            ic(data_packet["k"])
            k = data_packet["k"][0]
            i = data_packet["images"][0]
            d = data_packet["depths"][0]

            # Store image paths
            color_path = os.path.join(self.args.dataset_dir, "results", f"frame{k:05}.png")
            depth_path = os.path.join(self.args.dataset_dir, "results", f"depth{k:05}.png")

            # Save image to disk
            color = Image.fromarray(i)
            depth = Image.fromarray(d)
            color.save(color_path)
            depth.save(depth_path)

            # Sharpness
            sharp = sharpness(i)

            # Store relative path
            relative_color_path = os.path.join("results", os.path.basename(color_path))
            relative_depth_path = os.path.join("results", os.path.basename(depth_path))

            frame = {"file_path": relative_color_path, 
                     "sharpness": sharp,
                     "depth_path": relative_depth_path,
                     "transform_matrix": c2w}
            out["frames"].append(frame)

        with open(os.path.join(self.args.dataset_dir, OUT_PATH), "w") as outfile:
            import json
            json.dump(out, outfile, indent=2)