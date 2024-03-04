# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture("/dev/video2")

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

if __name__ == "__main__":

    dir_prefix = "/home/leev/monodepth-ros/src/monodepth_ros_noetic/scripts/"

    with torch.no_grad():

        print("Loading the pretrained network")
        encoder = ResnetEncoder(152, False)
        loaded_dict_enc = torch.load(
            dir_prefix + "ckpts/encoder.pth",
            map_location=device,
        )

        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
        }
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(
            dir_prefix + "ckpts/depth.pth",
            map_location=device,
        )
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        exit_signal = False

        while not exit_signal:
            _, frame = cap.read()

            raw_img = np.transpose(frame[:, :, :3], (2, 0, 1))
            input_image = torch.from_numpy(raw_img).float().to(device)
            input_image = (input_image / 255.0).unsqueeze(0)

            # resize to input size
            input_image = torch.nn.functional.interpolate(
                input_image, (256, 256), mode="bilinear", align_corners=False
            )
            features = encoder(input_image)
            outputs = depth_decoder(features)

            out = outputs[("out", 0)]
            # print(out.min())
            # print(out.max())
            # exit()
            # resize to original size
            out_resized = torch.nn.functional.interpolate(
                out, (512, 512), mode="bilinear", align_corners=False
            )
            # convert disparity to depth
            depth = output_to_depth(out_resized, 0.1, 10)
            metric_depth = depth.cpu().numpy().squeeze()

            # Please adjust vmax for visualization. 10.0 means 10 meters which is the whole prediction range.
            normalizer = mpl.colors.Normalize(vmin=0.1, vmax=4.0)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="viridis")
            colormapped_im = (mapper.to_rgba(metric_depth)[:, :, :3] * 255).astype(np.uint8)

            # cv2.imwrite(os.path.join(output_path, f"{idx:02d}.png"), colormapped_im[:,:,[2,1,0]])
            cv2.imshow('depth', colormapped_im[:,:,[2,1,0]])
            cv2.imshow('orig', frame)
            if cv2.waitKey(0) == ord('q'):
                exit_signal = True
        
        cap.release()
        exit()