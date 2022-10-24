#  Copyright 2018 Algolux Inc. All Rights Reserved.
import os
import cv2
import numpy as np
import json

crop_size = 150


json_file = '/external/10g/dense2/fs1/datasets/202210_GatedStereoDatasetv3/filenames_dict_gatedstereo.json'
print("Loading Filnames Dict info from JSON file {} ....".format(json_file))
with open(json_file, 'r') as fh:
    json_data = json.load(fh)
    print("Loading complete...")


def read_gated_image(base_dir, gta_pass, img_id, data_type, dataset, num_bits=10, scale_images=False,
                     scaled_img_width=None, scaled_img_height=None,
                     normalize_images=False):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for gate_id in range(3):
        if dataset == 'g2d':
            gate_dir = os.path.join(base_dir, gta_pass, 'gated%d_10bit' % gate_id)
            img = cv2.imread(os.path.join(gate_dir, img_id + '.png'), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        elif dataset == 'stf':
            mapx = np.load('./mapx.npz')['arr_0']
            mapy = np.load('./mapy.npz')['arr_0']
            gate_dir = os.path.join(base_dir, gta_pass, 'gated%d' % gate_id)
            img = cv2.imread(os.path.join(gate_dir, img_id + '.tiff'), cv2.IMREAD_UNCHANGED)
            img = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)
        elif dataset == 'gatedstereo':
            rec = img_id.split(',')[0]
            idx = img_id.split(',')[1]
            mapx = np.load('./mapx_gated_left.npz')['arr_0']
            mapy = np.load('./mapy_gated_left.npz')['arr_0']
            gate_dir = os.path.join(base_dir, json_data[rec][idx]['gated']['left'][gate_id][1:])
            img = cv2.imread(os.path.join(gate_dir, img_id + '.tiff'), cv2.IMREAD_UNCHANGED)
            img = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)


        if data_type == 'real':
            img = img[crop_size:(img.shape[0] - crop_size), crop_size:(img.shape[1] - crop_size)]
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))

    img = np.concatenate(gated_imgs, axis=2)
    if normalize_images:
        mean = np.mean(img, axis=2, keepdims=True)
        std = np.std(img, axis=2, keepdims=True)
        img = (img - mean) / (std + np.finfo(float).eps)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return np.expand_dims(img, axis=0)


def read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance, dataset, scale_images=False,
                  scaled_img_width=None,
                  scaled_img_height=None, raw_values_only=False, use_filtered_lidar=False):
    if data_type == 'real':
        if dataset == 'g2d':
            depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, "depth_hdl64_gated_compressed", img_id + '.npz'))['arr_0']
        elif dataset == 'stf':
            if not use_filtered_lidar:
                depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, 'lidar_hdl64_strongest_gated', img_id + '.npz'))['arr_0']
            else:
                depth_lidar1 = np.load(os.path.join(base_dir, gta_pass, 'lidar_hdl64_strongest_filtered_gated', img_id + '.npz'))['arr_0']
        elif dataset == 'gatedstereo':
            rec = img_id.split(',')[0]
            idx = img_id.split(',')[1]
            depth_lidar1 = np.load(os.path.join(base_dir,  json_data[rec][idx]['gated']['left']['lidar_gated'][1:]))['arr_0']


        depth_lidar1 = depth_lidar1[crop_size:(depth_lidar1.shape[0] - crop_size),
                       crop_size: (depth_lidar1.shape[1] - crop_size)]
        if raw_values_only:
            return depth_lidar1, None

        gt_mask = (depth_lidar1 > 0.)

        depth_lidar1 = np.float32(np.clip(depth_lidar1, min_distance, max_distance) / max_distance)

        return np.expand_dims(np.expand_dims(depth_lidar1, axis=2), axis=0), \
               np.expand_dims(np.expand_dims(gt_mask, axis=2), axis=0)

    img = np.load(os.path.join(base_dir, gta_pass, 'depth_compressed', img_id + '.npz'))['arr_0']

    if raw_values_only:
        return img, None

    img = np.clip(img, min_distance, max_distance) / max_distance
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)

    return np.expand_dims(np.expand_dims(img, axis=2), axis=0), None
