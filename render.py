#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from utils.system_utils import autoChooseCudaDevice
autoChooseCudaDevice()
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import numpy as np


def render_set_for_FPS_test(args, model_path, name, iteration, views, gaussians, pipeline, background):
    """
    input: Keep the same input parameters as render_set(...)
    output: the output is a more accurate FPS.
    """
    t_list_len = 200
    warmup_times = 5
    test_times = 10
    t_list = np.array([1.0] * t_list_len)
    step = 0
    fps_list = []
    while True:
        for view in views:
            step += 1
            torch.cuda.synchronize();
            t0 = time.time()
            rendering = render(view, gaussians, pipeline, background)["render"]
            torch.cuda.synchronize();
            t1 = time.time()
            t_list[step % t_list_len] = t1 - t0

            if step % t_list_len == 0 and step > t_list_len * warmup_times:
                fps = 1.0 / t_list.mean()
                print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
                fps_list.append(fps)
            if step > t_list_len * (test_times + warmup_times):
                # write fps info to a txt file
                with open(os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "FPS.txt"), 'w') as f:
                    f.write("Average FPS: {:.5f}\n".format(np.mean(fps_list)))
                    f.write("FPS std: {:.5f}\n".format(np.std(fps_list)))
                print("Average FPS: {:.5f}, FPS std: {:.5f}".format(np.mean(fps_list), np.std(fps_list)))
                return
                
def render_set(args, model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    errors_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(errors_path, exist_ok=True)
    psnr_test = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :].cuda()
        # error_map = torch.mean(torch.abs(rendering - gt), dim=0)
        error_map = torch.abs(rendering - gt)
        if args.store_image:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(error_map, os.path.join(errors_path, '{0:05d}'.format(idx) + ".png"))
        psnr_test += psnr(rendering, gt).mean().double()
    psnr_test /= len(views)
    print("PSNR: ", psnr_test)

def render_sets(args, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.depth_correct)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(args, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(args, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        
        render_set_for_FPS_test(args, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--store_image", action='store_true', default=False)  # store render image
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)