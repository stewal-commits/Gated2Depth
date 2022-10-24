#  Copyright 2018 Algolux Inc. All Rights Reserved.
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2
import os
from metrics import calc_metrics, metric_str
import LSGAN as lsgan
import dataset_util as dsutil
import visualize2D

def calc_bins(clip_min, clip_max, nb_bins):
    bins = np.linspace(clip_min, clip_max, num=nb_bins + 1)
    mean_bins = np.array([0.5 * (bins[i + 1] + bins[i]) for i in range(0, nb_bins)])
    return bins, mean_bins


def run(results_dir, model_dir, base_dir, file_names, data_type, use_multi_scale=False,
        exported_disc_path=None, use_3dconv=False, compute_metrics=False, min_distance=3., max_distance=150., show_result=False, dataset='g2d', use_filtered_lidar=False, binned_metric=False):
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, None, None, 3])

    gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
    gt_mask = None
    if data_type == 'real':
        gt_mask = tf.placeholder(tf.float32, [None, None, None, 1])
        model = lsgan.build_model(in_image, gt_image, data_type=data_type, gt_mask=gt_mask,
                                  smooth_weight=0.1, adv_weight=0.0001, discriminator_ckpt=exported_disc_path,
                                  use_multi_scale=use_multi_scale, use_3dconv=use_3dconv)
        min_eval_distance = min_distance
        max_eval_distance = 80.
        nb_bins = 11
        nb_metrics = 7
    else:
        model = lsgan.build_model(in_image, gt_image, data_type=data_type, gt_mask=gt_mask, smooth_weight=1e-4,
                                  adv_weight=0.0001, use_multi_scale=use_multi_scale, use_3dconv=use_3dconv)
        min_eval_distance = min_distance
        max_eval_distance = max_distance
        nb_bins = 21
        nb_metrics = 7

    bins, mean_bins = calc_bins(min_eval_distance, max_eval_distance, nb_bins)
    out_image = model['out_image']

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)

    per_image_metrics = []
    per_image_metrics_binned = []
    points = []


    #mae = []

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # results_folder = ['gated2depth', 'gated2depth_img', 'all']
    # results_folder = ['gated2depth_original']
    # for result_folder in results_folder:
    #     if not os.path.exists(os.path.join(results_dir, result_folder)):
    #         os.makedirs(os.path.join(results_dir, result_folder))
    # print(len(file_names))

    for ind in range(len(file_names)):
        # get the path from image id
        train_fn = file_names[ind]
        if data_type == 'real':
            img_id = train_fn
            gta_pass = ''

        else:
            img_id = train_fn
            gta_pass = ''
        if not 'gatedstereo' == dataset:
            img_id = img_id.replace(',', '_')

        in_img = dsutil.read_gated_image(base_dir, gta_pass, img_id, data_type, dataset)

        input_patch = in_img
        output = sess.run(out_image, feed_dict={in_image: input_patch})
        output = np.clip(output * max_distance, min_distance, max_distance)

        gt_patch, _ = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, raw_values_only=True, min_distance=min_distance, max_distance=max_distance, dataset=dataset, use_filtered_lidar=use_filtered_lidar)
        
        if compute_metrics:
            #if data_type != 'real':
                #curr_mae = np.mean(np.abs(output - gt_patch), dtype=np.float64)
            if np.sum(gt_patch > 0.0) > 0.:
                curr_metrics = calc_metrics(output[0, :, :, 0], gt_patch, min_distance=min_eval_distance,max_distance=max_eval_distance)
                per_image_metrics.append(curr_metrics)
                    #mae.append(curr_mae)

                if binned_metric:
                    results = np.vstack([output[:, :, :, 0], np.expand_dims(gt_patch, axis=0)])
                    if nb_bins == 1:
                        inds = np.zeros((results[0].shape), dtype=int)
                    else:
                        inds = np.digitize(results[1, :], bins)

                    error_binned = np.zeros((len(bins) - 1, nb_metrics))
                    point = np.zeros((len(bins)-1))
                    for i, bin in enumerate(bins[:-1]):
                        try:
                            metrics = calc_metrics(results[0, inds == i+1], results[1, inds == i+1], min_distance=min_eval_distance, max_distance=max_eval_distance)
                            error_binned[i, :] = metrics
                            point[i] = np.sum(results[0,inds == i+1] != 0)
                        except ValueError:
                            error_binned[i, :] = np.nan

                    points.append(point)
                    mean_error_binned = np.zeros((nb_metrics,))
                    for i in range(0, nb_metrics):
                        mean_error_binned[i] = np.mean(error_binned[~np.isnan(error_binned[:, i]), i])

                    error_binned = np.hstack([mean_bins.reshape((-1, 1)), error_binned])
                    mean_error_binned = np.hstack([np.zeros((1,)), mean_error_binned])
                    curr_metrics_binned = np.vstack([mean_error_binned, error_binned])
                    per_image_metrics_binned.append(curr_metrics_binned)

        if not os.path.exists(os.path.join(results_dir, 'gated2depth_original')):
            os.mkdir(os.path.join(results_dir, 'gated2depth_original'))


        np.savez_compressed(os.path.join(results_dir, 'gated2depth_original', '{}'.format(img_id)), output)

        # #depth_lidar1, _ = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, raw_values_only=True, min_distance=min_distance, max_distance=max_distance)
        #
        # if data_type != 'real':
        #     #print(depth_lidar1.shape)
        #     depth_lidar1_color = visualize2D.colorize_depth(gt_patch, min_distance=min_eval_distance, max_distance=max_eval_distance)
        # else:
        #     #print(depth_lidar1.shape)
        #     depth_lidar1_color = visualize2D.colorize_pointcloud(gt_patch, min_distance=min_eval_distance,
        #                                                      max_distance=max_eval_distance, radius=3)
        #
        # depth_map_color = visualize2D.colorize_depth(output[0, :, :, 0], min_distance=min_eval_distance,
        #                                              max_distance=max_eval_distance)
        #
        # in_out_shape = (int(depth_map_color.shape[0] + depth_map_color.shape[0] / 3. +
        #                     gt_patch.shape[0]), depth_map_color.shape[1], 3)
        #
        # input_output = np.zeros(shape=in_out_shape)
        # scaled_input = cv2.resize(input_patch[0, :, :, :],
        #                           dsize=(int(input_patch.shape[2] / 3), int(input_patch.shape[1] / 3)),
        #                           interpolation=cv2.INTER_AREA) * 255
        #
        # for i in range(3):
        #     input_output[:scaled_input.shape[0], :scaled_input.shape[1], i] = scaled_input[:, :, 0]
        #     input_output[:scaled_input.shape[0], scaled_input.shape[1]: 2 * scaled_input.shape[1], i] = scaled_input[:,
        #                                                                                                 :, 1]
        #     input_output[:scaled_input.shape[0], scaled_input.shape[1] * 2:scaled_input.shape[1] * 3, i] = scaled_input[
        #                                                                                                    :, :, 2]
        #
        # input_output[scaled_input.shape[0]: scaled_input.shape[0] + depth_map_color.shape[0], :, :] = depth_map_color
        # input_output[scaled_input.shape[0] + depth_map_color.shape[0]:, :, :] = depth_lidar1_color
        # cv2.imwrite(os.path.join(results_dir, 'gated2depth_img', '{}.jpg'.format(img_id)), depth_map_color.astype(np.uint8))
        # cv2.imwrite(os.path.join(results_dir, 'all', '{}.jpg'.format(img_id)), input_output.astype(np.uint8))
        #
        # if show_result:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(cv2.cvtColor(input_output.astype(np.uint8), cv2.COLOR_BGR2RGB))
        #     plt.show()

    if compute_metrics:
        print(np.sum(np.asarray(points), axis=0))
        res = np.mean(per_image_metrics, axis=0)
        res_str = ''
        for i in range(res.shape[0]):
            res_str += '{}={:.2f} \n'.format(metric_str[i], res[i])
        print(res_str)
        with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
            f.write(res_str)
        with open(os.path.join(results_dir, 'results.tex'), 'w') as f:
            f.write(' & '.join(metric_str) + '\n')
            f.write(' & '.join(['{:.2f}'.format(r) for r in res]))
        res = np.zeros_like(per_image_metrics_binned[0])
        if binned_metric:
            per_image_metrics_binned = np.asarray(per_image_metrics_binned)
            for i in range(0, res.shape[0]):
                for j in range(0, res.shape[1]):
                    res[i,j] = np.mean(per_image_metrics_binned[:,i,j][~np.isnan(per_image_metrics_binned[:,i,j])])
        res = np.vstack([np.hstack([np.zeros((1,)), np.mean(res[1:, 1:][~np.isnan(res[1:, 1]), :], axis=0)]), res])
        np.savez(os.path.join(results_dir, 'results_binned.npz'), res)
        res = np.around(res, decimals=2)
        np.set_printoptions(suppress=True)

        print(res)
        np.savetxt(os.path.join(results_dir, 'results_binned.txt'), res, fmt='%1.2f')
        np.savetxt(os.path.join(results_dir, 'results_binned_mean.tex'), res[0:2,1:].reshape(2,-1), delimiter=' & ',fmt='%1.2f')
        np.savetxt(os.path.join(results_dir, 'results_binned.tex'), np.transpose(np.nan_to_num(res[2:,:])), delimiter=' & ',fmt='%1.2f')