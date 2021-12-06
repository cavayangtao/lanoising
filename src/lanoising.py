#! /usr/bin/env python

import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import joblib

import keras
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense, Dropout
from keras.callbacks import Callback
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras.models import load_model


# functions for MDN
def elu_modif(x, a=1.):
    return ELU(alpha=a)(x) + 1. + 1e-8

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                       axis=axis, keepdims=True)) + x_max

def mean_log_Gaussian_like_disappear(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = K.reshape(parameters, [-1, c + 2, md])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha, 1e-8, 1.))

    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
               - float(c) * K.log(sigma) \
               - K.sum((K.expand_dims(y_true, 2) - mu) ** 2, axis=1) / (2 * (sigma) ** 2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

def mean_log_Gaussian_like_range(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = K.reshape(parameters, [-1, c + 2, mr])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha, 1e-8, 1.))

    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
               - float(c) * K.log(sigma) \
               - K.sum((K.expand_dims(y_true, 2) - mu) ** 2, axis=1) / (2 * (sigma) ** 2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

def mean_log_Gaussian_like_intensity(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = K.reshape(parameters, [-1, c + 2, mi])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha, 1e-8, 1.))

    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
               - float(c) * K.log(sigma) \
               - K.sum((K.expand_dims(y_true, 2) - mu) ** 2, axis=1) / (2 * (sigma) ** 2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

def LaPlacePDF(x, mu, b):
    return np.exp(-np.abs(x - mu) / b) / (2 * b)

def LaGaussianPDF(x, mu, sigma):
    return np.exp(-np.square(x - mu) / (2 * np.square(sigma))) / (sigma * np.sqrt(2 * np.pi))

def callback(msg):

    pt_x = []
    pt_y = []
    pt_z = []
    pt_d = []
    pt_i = []
    pt_x_new1 = []
    pt_y_new1 = []
    pt_z_new1 = []
    pt_i_new1 = []
    pt_x_new2 = []
    pt_y_new2 = []
    pt_z_new2 = []
    pt_i_new2 = []

    # get all the points
    points = pc2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=False)
    for point in points:
        pt_x.append(point[0])
        pt_y.append(point[1])
        pt_z.append(point[2])
        pt_d.append(np.sqrt(np.square(point[0]) + np.square(point[1]) + np.square(point[2])))
        pt_i.append(point[3])

    pt_x = np.array(pt_x)
    pt_y = np.array(pt_y)
    pt_z = np.array(pt_z)
    pt_d = np.array(pt_d)
    pt_i = np.array(pt_i)
    print('maximal intensity: ', np.max(pt_i))
    print('median intensity: ', np.median(pt_i))

    # only deal with points in max_range
    index_outrange = np.where(pt_d > max_range)
    # pt_x_new0 = pt_x[index_outrange[0]]
    # pt_y_new0 = pt_y[index_outrange[0]]
    # pt_z_new0 = pt_z[index_outrange[0]]
    # pt_i_new0 = pt_i[index_outrange[0]]
    pt_x = np.delete(pt_x, index_outrange[0])
    pt_y = np.delete(pt_y, index_outrange[0])
    pt_z = np.delete(pt_z, index_outrange[0])
    pt_d = np.delete(pt_d, index_outrange[0])
    pt_i = np.delete(pt_i, index_outrange[0])

    # process the points for diffuse reflectors
    index = np.where(pt_i <= intensity)
    if np.size(index[0]) > 0:

        pt_x_new = pt_x[index[0]]
        pt_y_new = pt_y[index[0]]
        pt_z_new = pt_z[index[0]]
        pt_d_new = pt_d[index[0]]
        pt_i_new = pt_i[index[0]]

        # disappear visibility prediction
        x_pred = np.transpose(np.vstack((pt_d_new, pt_i_new)))
        if disappear_model == 0:
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            y_pred, y_sigm = gpr1_disappear.predict(x_pred, return_std=True)
        else:
            x_pred[:, 0] = x_pred[:, 0] / 30.0
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            parameters = mdn1_disappear.predict(x_pred)
            comp = np.reshape(parameters, [-1, c + 2, md])
            mu_pred = comp[:, :c, :]
            sigma_pred = comp[:, c, :]
            alpha_pred = comp[:, c + 1, :]
            y_pred = np.zeros(len(mu_pred))
            y_sigm = np.zeros(len(mu_pred))
            for mx in range(mu_pred.shape[-1]):
                y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
            y_sigm = np.sqrt(y_sigm)
        vdis = np.random.normal(y_pred, y_sigm)

        # range prediction
        x_pred = np.vstack((np.ones(np.size(pt_i_new)) * visibility, pt_i_new, pt_d_new))
        x_pred = np.atleast_2d(x_pred).T
        x_pred[:, 0] = x_pred[:, 0] / 200.0
        x_pred[:, 1] = x_pred[:, 1] / 100.0
        x_pred[:, 2] = x_pred[:, 2] / 30.0
        if range_model == 0:
            y_pred, y_sigm = gpr1_range.predict(x_pred, return_std=True)
        else:
            parameters = mdn1_range.predict(x_pred)
            comp = np.reshape(parameters, [-1, c + 2, mr])
            mu_pred = comp[:, :c, :]
            sigma_pred = comp[:, c, :]
            alpha_pred = comp[:, c + 1, :]
            y_pred = np.zeros(len(mu_pred))
            y_sigm = np.zeros(len(mu_pred))
            for mx in range(mu_pred.shape[-1]):
                y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
            y_sigm = np.sqrt(y_sigm)

        # sampling
        pt_d_new1_c = np.random.normal(y_pred, y_sigm)
        ratio1 = pt_d_new1_c / pt_d_new
        index_neg = np.where(ratio1 < 0)
        ratio1[index_neg[0]] = 0
        index_big = np.where(ratio1 > 1)
        ratio1[index_big[0]] = 1
        pt_x_new1_c = pt_x_new * ratio1
        pt_y_new1_c = pt_y_new * ratio1
        pt_z_new1_c = pt_z_new * ratio1
        errors = abs(pt_d_new1_c - pt_d_new)
        index_vis = np.where((vdis <= visibility) & (errors <= sigma))

        # get good points
        if np.size(index_vis[0]) > 0:
            pt_x_new1_a = pt_x_new[index_vis[0]]
            pt_y_new1_a = pt_y_new[index_vis[0]]
            pt_z_new1_a = pt_z_new[index_vis[0]]
            pt_d_new1_a = pt_d_new[index_vis[0]]
            pt_i_new1_a = pt_i_new[index_vis[0]]

            x_pred = np.vstack((np.ones(np.size(pt_i_new1_a)) * visibility, pt_i_new1_a, pt_d_new1_a))
            x_pred = np.atleast_2d(x_pred).T
            x_pred[:, 0] = x_pred[:, 0] / 200.0
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            x_pred[:, 2] = x_pred[:, 2] / 30.0

            # intensity estimation
            if intensity_model == 0:
                y_pred, y_sigm = gpr1_intensity.predict(x_pred, return_std=True)
            else:
                parameters = mdn1_intensity.predict(x_pred)
                comp = np.reshape(parameters, [-1, c + 2, mi])
                mu_pred = comp[:, :c, :]
                sigma_pred = comp[:, c, :]
                alpha_pred = comp[:, c + 1, :]
                y_pred = np.zeros(len(mu_pred))
                y_sigm = np.zeros(len(mu_pred))
                for mx in range(mu_pred.shape[-1]):
                    y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                    y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
                y_sigm = np.sqrt(y_sigm)

            pt_i_new1_tmp = np.random.normal(y_pred, y_sigm)
            index_neg = np.where(pt_i_new1_tmp < 0)
            pt_i_new1_tmp[index_neg[0]] = 0
            index_big = np.where(pt_i_new1_tmp > pt_i_new1_a)
            pt_i_new1_tmp[index_big[0]] = pt_i_new1_a[index_big[0]]
            pt_i_new1_a = pt_i_new1_tmp

        # get noisy points
        if (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_d_new1_b = np.delete(pt_d_new, index_vis[0])
            pt_i_new1_b = np.delete(pt_i_new, index_vis[0])
            x_pred = np.vstack((np.ones(np.size(pt_i_new1_b)) * visibility, pt_i_new1_b, pt_d_new1_b))
            x_pred = np.atleast_2d(x_pred).T
            x_pred[:, 0] = x_pred[:, 0] / 200.0
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            x_pred[:, 2] = x_pred[:, 2] / 30.0

            pt_x_new1_c = np.delete(pt_x_new1_c, index_vis[0])
            pt_y_new1_c = np.delete(pt_y_new1_c, index_vis[0])
            pt_z_new1_c = np.delete(pt_z_new1_c, index_vis[0])

            # intensity estimation
            if intensity_model == 0:
                y_pred, y_sigm = gpr1_intensity.predict(x_pred, return_std=True)
            else:
                parameters = mdn1_intensity.predict(x_pred)
                comp = np.reshape(parameters, [-1, c + 2, mi])
                mu_pred = comp[:, :c, :]
                sigma_pred = comp[:, c, :]
                alpha_pred = comp[:, c + 1, :]
                y_pred = np.zeros(len(mu_pred))
                y_sigm = np.zeros(len(mu_pred))
                for mx in range(mu_pred.shape[-1]):
                    y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                    y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
                y_sigm = np.sqrt(y_sigm)

            pt_i_new1_tmp = np.random.normal(y_pred, y_sigm)
            index_neg = np.where(pt_i_new1_tmp < 0)
            pt_i_new1_tmp[index_neg[0]] = 0
            index_big = np.where(pt_i_new1_tmp > pt_i_new1_b)
            pt_i_new1_tmp[index_big[0]] = pt_i_new1_b[index_big[0]]
            pt_i_new1_c = pt_i_new1_tmp

        # put the good points and noisy points together
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new1 = np.hstack((pt_x_new1_a, pt_x_new1_c))
            pt_y_new1 = np.hstack((pt_y_new1_a, pt_y_new1_c))
            pt_z_new1 = np.hstack((pt_z_new1_a, pt_z_new1_c))
            pt_i_new1 = np.hstack((pt_i_new1_a, pt_i_new1_c))
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) == 0:
            pt_x_new1 = pt_x_new1_a
            pt_y_new1 = pt_y_new1_a
            pt_z_new1 = pt_z_new1_a
            pt_i_new1 = pt_i_new1_a
        if np.size(index_vis[0]) == 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new1 = pt_x_new1_c
            pt_y_new1 = pt_y_new1_c
            pt_z_new1 = pt_z_new1_c
            pt_i_new1 = pt_i_new1_c

    # process the points for retro-reflectors
    index = np.where(pt_i > intensity)
    if np.size(index[0]) > 0:
        pt_x_new = pt_x[index[0]]
        pt_y_new = pt_y[index[0]]
        pt_z_new = pt_z[index[0]]
        pt_d_new = pt_d[index[0]]
        pt_i_new = pt_i[index[0]]

        # disappear visibility prediction
        x_pred = np.transpose(np.vstack((pt_d_new, pt_i_new)))
        if disappear_model == 0:
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            y_pred, y_sigm = gpr2_disappear.predict(x_pred, return_std=True)
        else:
            x_pred[:, 0] = x_pred[:, 0] / 30.0
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            parameters = mdn2_disappear.predict(x_pred)
            comp = np.reshape(parameters, [-1, c + 2, md])
            mu_pred = comp[:, :c, :]
            sigma_pred = comp[:, c, :]
            alpha_pred = comp[:, c + 1, :]
            y_pred = np.zeros(len(mu_pred))
            y_sigm = np.zeros(len(mu_pred))
            for mx in range(mu_pred.shape[-1]):
                y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
            y_sigm = np.sqrt(y_sigm)
        vdis = np.random.normal(y_pred, y_sigm)

        # range prediction
        x_pred = np.vstack((np.ones(np.size(pt_i_new)) * visibility, pt_i_new, pt_d_new))
        x_pred = np.atleast_2d(x_pred).T
        x_pred[:, 0] = x_pred[:, 0] / 200.0
        x_pred[:, 1] = x_pred[:, 1] / 100.0
        x_pred[:, 2] = x_pred[:, 2] / 30.0
        if range_model == 0:
            y_pred, y_sigm = gpr2_range.predict(x_pred, return_std=True)
        else:
            parameters = mdn2_range.predict(x_pred)
            comp = np.reshape(parameters, [-1, c + 2, mr])
            mu_pred = comp[:, :c, :]
            sigma_pred = comp[:, c, :]
            alpha_pred = comp[:, c + 1, :]
            y_pred = np.zeros(len(mu_pred))
            y_sigm = np.zeros(len(mu_pred))
            for mx in range(mu_pred.shape[-1]):
                y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
            y_sigm = np.sqrt(y_sigm)

        # sampling
        pt_d_new2_c = np.random.normal(y_pred, y_sigm)
        ratio2 = pt_d_new2_c / pt_d_new
        index_neg = np.where(ratio2 < 0)
        ratio2[index_neg[0]] = 0
        index_big = np.where(ratio2 > 1)
        ratio2[index_big[0]] = 1
        pt_x_new2_c = pt_x_new * ratio2
        pt_y_new2_c = pt_y_new * ratio2
        pt_z_new2_c = pt_z_new * ratio2
        errors = abs(pt_d_new2_c - pt_d_new)
        index_vis = np.where((vdis <= visibility) & (errors <= sigma))

        # get good points
        if np.size(index_vis[0]) > 0:
            pt_x_new2_a = pt_x_new[index_vis[0]]
            pt_y_new2_a = pt_y_new[index_vis[0]]
            pt_z_new2_a = pt_z_new[index_vis[0]]
            pt_d_new2_a = pt_d_new[index_vis[0]]
            pt_i_new2_a = pt_i_new[index_vis[0]]

            x_pred = np.vstack((np.ones(np.size(pt_i_new2_a)) * visibility, pt_i_new2_a, pt_d_new2_a))
            x_pred = np.atleast_2d(x_pred).T
            x_pred[:, 0] = x_pred[:, 0] / 200.0
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            x_pred[:, 2] = x_pred[:, 2] / 30.0

            # intensity estimation
            if intensity_model == 0:
                y_pred, y_sigm = gpr2_intensity.predict(x_pred, return_std=True)
            else:
                parameters = mdn2_intensity.predict(x_pred)
                comp = np.reshape(parameters, [-1, c + 2, mi])
                mu_pred = comp[:, :c, :]
                sigma_pred = comp[:, c, :]
                alpha_pred = comp[:, c + 1, :]
                y_pred = np.zeros(len(mu_pred))
                y_sigm = np.zeros(len(mu_pred))
                for mx in range(mu_pred.shape[-1]):
                    y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                    y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
                y_sigm = np.sqrt(y_sigm)

            pt_i_new2_tmp = np.random.normal(y_pred, y_sigm)
            index_neg = np.where(pt_i_new2_tmp < 0)
            pt_i_new2_tmp[index_neg[0]] = 0
            index_big = np.where(pt_i_new2_tmp > pt_i_new2_a)
            pt_i_new2_tmp[index_big[0]] = pt_i_new2_a[index_big[0]]
            pt_i_new2_a = pt_i_new2_tmp

        # get noisy points
        if (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_d_new2_b = np.delete(pt_d_new, index_vis[0])
            pt_i_new2_b = np.delete(pt_i_new, index_vis[0])
            x_pred = np.vstack((np.ones(np.size(pt_i_new2_b)) * visibility, pt_i_new2_b, pt_d_new2_b))
            x_pred = np.atleast_2d(x_pred).T
            x_pred[:, 0] = x_pred[:, 0] / 200.0
            x_pred[:, 1] = x_pred[:, 1] / 100.0
            x_pred[:, 2] = x_pred[:, 2] / 30.0

            pt_x_new2_c = np.delete(pt_x_new2_c, index_vis[0])
            pt_y_new2_c = np.delete(pt_y_new2_c, index_vis[0])
            pt_z_new2_c = np.delete(pt_z_new2_c, index_vis[0])

            # intensity estimation
            if intensity_model == 0:
                y_pred, y_sigm = gpr2_intensity.predict(x_pred, return_std=True)
            else:
                parameters = mdn2_intensity.predict(x_pred)
                comp = np.reshape(parameters, [-1, c + 2, mi])
                mu_pred = comp[:, :c, :]
                sigma_pred = comp[:, c, :]
                alpha_pred = comp[:, c + 1, :]
                y_pred = np.zeros(len(mu_pred))
                y_sigm = np.zeros(len(mu_pred))
                for mx in range(mu_pred.shape[-1]):
                    y_pred[:] += alpha_pred[:, mx] * mu_pred[:, 0, mx]
                    y_sigm[:] += np.square(alpha_pred[:, mx] * sigma_pred[:, mx])
                y_sigm = np.sqrt(y_sigm)

            pt_i_new2_tmp = np.random.normal(y_pred, y_sigm)
            index_neg = np.where(pt_i_new2_tmp < 0)
            pt_i_new2_tmp[index_neg[0]] = 0
            index_big = np.where(pt_i_new2_tmp > pt_i_new2_b)
            pt_i_new2_tmp[index_big[0]] = pt_i_new2_b[index_big[0]]
            pt_i_new2_c = pt_i_new2_tmp

        # put the good points and noisy points together
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new2 = np.hstack((pt_x_new2_a, pt_x_new2_c))
            pt_y_new2 = np.hstack((pt_y_new2_a, pt_y_new2_c))
            pt_z_new2 = np.hstack((pt_z_new2_a, pt_z_new2_c))
            pt_i_new2 = np.hstack((pt_i_new2_a, pt_i_new2_c))
        if np.size(index_vis[0]) > 0 and (np.size(pt_x_new) - np.size(index_vis[0])) == 0:
            pt_x_new2 = pt_x_new2_a
            pt_y_new2 = pt_y_new2_a
            pt_z_new2 = pt_z_new2_a
            pt_i_new2 = pt_i_new2_a
        if np.size(index_vis[0]) == 0 and (np.size(pt_x_new) - np.size(index_vis[0])) > 0:
            pt_x_new2 = pt_x_new2_c
            pt_y_new2 = pt_y_new2_c
            pt_z_new2 = pt_z_new2_c
            pt_i_new2 = pt_i_new2_c

    # put all the points together
    if np.size(pt_x_new1) > 0:
        cloud_points1 = np.transpose(np.vstack((pt_x_new1, pt_y_new1, pt_z_new1, pt_i_new1)))
    if np.size(pt_x_new2) > 0:
        cloud_points2 = np.transpose(np.vstack((pt_x_new2, pt_y_new2, pt_z_new2, pt_i_new2)))

    if np.size(pt_x_new1) > 0 and np.size(pt_x_new2) > 0:
        cloud_points = np.vstack((cloud_points1, cloud_points2))
    if np.size(pt_x_new1) > 0 and np.size(pt_x_new2) == 0:
        cloud_points = cloud_points1
    if np.size(pt_x_new1) == 0 and np.size(pt_x_new2) > 0:
        cloud_points = cloud_points2

    # if np.size(pt_x_new0) > 0:
    #     cloud_points0 = np.transpose(np.vstack((pt_x_new0, pt_y_new0, pt_z_new0)))
    #     cloud_points = np.vstack((cloud_points, cloud_points0))

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('intensity', 12, PointField.FLOAT32, 1),
          # PointField('rgba', 12, PointField.UINT32, 1),
          ]

    #header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'velodyne'
    #create pcl from points
    new_points = pc2.create_cloud(header, fields, cloud_points)
    pub.publish(new_points)
    # rospy.loginfo(points)

if __name__ == '__main__':

    visibility = 50  # set the visibility
    sigma = 10 # set the threshold for good points
    intensity = 120 # set the intensity threshold to distinguish diffuse reflectors and retro-reflectors
    max_range = 35 # maximal simulation range
    disappear_model = 0 # choose the model for disappear visibility, set 0 for GPR and set 1 for MDN
    range_model = 1 # choose the model for ranging estimation, set 0 for GPR and set 1 for MDN
    intensity_model = 1 # choose the model for intensity estimation, set 0 for GPR and set 1 for MDN
    c = 1 # parameters for MDN, don't change
    md = 1  # parameters for MDN, don't change
    mr = 1 # parameters for MDN, don't change
    mi = 1  # parameters for MDN, don't change

    np.random.seed(1)
    rospy.init_node('lanoising', anonymous=True)
        
    gpr1_disappear_path = rospy.get_param('~gpr1_disappear_path')
    gpr2_disappear_path = rospy.get_param('~gpr2_disappear_path')
    gpr1_disappear = joblib.load(gpr1_disappear_path)
    gpr2_disappear = joblib.load(gpr2_disappear_path)
    mdn1_disappear_path = rospy.get_param('~mdn1_disappear_path')
    mdn2_disappear_path = rospy.get_param('~mdn2_disappear_path')
    mdn1_disappear = load_model(mdn1_disappear_path, custom_objects={'elu_modif': elu_modif, 'log_sum_exp': log_sum_exp, 'mean_log_Gaussian_like': mean_log_Gaussian_like_disappear})
    mdn2_disappear = load_model(mdn2_disappear_path, custom_objects={'elu_modif': elu_modif, 'log_sum_exp': log_sum_exp, 'mean_log_Gaussian_like': mean_log_Gaussian_like_disappear})

    gpr1_range_path = rospy.get_param('~gpr1_range_path')
    gpr2_range_path = rospy.get_param('~gpr2_range_path')
    gpr1_range = joblib.load(gpr1_range_path)
    gpr2_range = joblib.load(gpr2_range_path)
    mdn1_range_path = rospy.get_param('~mdn1_range_path')
    mdn2_range_path = rospy.get_param('~mdn2_range_path')
    mdn1_range = load_model(mdn1_range_path, custom_objects={'elu_modif': elu_modif, 'log_sum_exp': log_sum_exp, 'mean_log_Gaussian_like': mean_log_Gaussian_like_range})
    mdn2_range = load_model(mdn2_range_path, custom_objects={'elu_modif': elu_modif, 'log_sum_exp': log_sum_exp, 'mean_log_Gaussian_like': mean_log_Gaussian_like_range})

    gpr1_intensity_path = rospy.get_param('~gpr1_intensity_path')
    gpr2_intensity_path = rospy.get_param('~gpr2_intensity_path')
    gpr1_intensity = joblib.load(gpr1_intensity_path)
    gpr2_intensity = joblib.load(gpr2_intensity_path)
    mdn1_intensity_path = rospy.get_param('~mdn1_intensity_path')
    mdn2_intensity_path = rospy.get_param('~mdn2_intensity_path')
    mdn1_intensity = load_model(mdn1_intensity_path, custom_objects={'elu_modif': elu_modif, 'log_sum_exp': log_sum_exp, 'mean_log_Gaussian_like': mean_log_Gaussian_like_intensity})
    mdn2_intensity = load_model(mdn2_intensity_path, custom_objects={'elu_modif': elu_modif, 'log_sum_exp': log_sum_exp, 'mean_log_Gaussian_like': mean_log_Gaussian_like_intensity})

    mdn1_disappear._make_predict_function()
    mdn2_disappear._make_predict_function()
    mdn1_range._make_predict_function()
    mdn2_range._make_predict_function()
    mdn1_intensity._make_predict_function()
    mdn2_intensity._make_predict_function()

    sub = rospy.Subscriber('velodyne_points', PointCloud2, callback)
    pub = rospy.Publisher('filtered_points', PointCloud2, queue_size=10)

    rospy.spin()
