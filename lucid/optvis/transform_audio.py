# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Tranformations you might want neural net visualizations to be robust to.

This module provides a variety of functions which stochastically transform a
tensorflow tensor. The functions are of the form:

  (config) => (tensor) => (stochastic transformed tensor)

"""

import tensorflow as tf
import numpy as np
import uuid
import math

from lucid.optvis import param


def jitter(d, seed=None):
    def inner(t_audio):
        t_audio = tf.convert_to_tensor(t_audio, preferred_dtype=tf.float32)
        t_shp = tf.shape(t_audio)
        crop_shape = [t_shp[0], t_shp[1]-d, t_shp[2]]
        crop = tf.random_crop(t_audio, crop_shape, seed=seed)
        shp = t_audio.get_shape().as_list()
        mid_shp_changed = [
            shp[0],
            shp[1] - d if shp[1] is not None else None,
            shp[2]
        ]
        crop.set_shape(mid_shp_changed)
        return crop

    return inner


def pad(w, mode="REFLECT", constant_value=0.5):
    def inner(t_audio):
        if constant_value == "uniform":
            constant_value_ = tf.random_uniform([], 0, 1)
        else:
            constant_value_ = constant_value
        return tf.pad(
            t_audio,
            [(0, 0), (w, w), (0, 0)],
            mode=mode,
            constant_values=constant_value_,
        )

    return inner

def amplitude_scaling(scales, seed=None):
    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        scale = _rand_select(scales, seed=seed)
        return scale*t
    return inner

def zero_mean():
    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        return t - tf.reduce_mean(t)
    return inner

def norm_pdf(x):
    return (1/tf.sqrt(2*math.pi))*tf.exp(-tf.pow(x,2)/2)

def random_muffle(sds, seed=None):
    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        sd = _rand_select(sds, seed=seed)
        sd = tf.cast(80*sd, dtype=tf.int32)
        norm = tf.expand_dims(tf.expand_dims(norm_pdf(tf.linspace(-1.,1.,sd)),-1),-1)
        return tf.nn.convolution(t, norm, padding="SAME")

    return inner

def normalize_gradient(grad_scales=None):

    if grad_scales is not None:
        grad_scales = np.float32(grad_scales)

    op_name = "NormalizeGrad_" + str(uuid.uuid4())

    @tf.RegisterGradient(op_name)
    def _NormalizeGrad(op, grad):
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, [1, 2, 3], keepdims=True))
        if grad_scales is not None:
            grad *= grad_scales[:, None, None, None]
        return grad / grad_norm

    def inner(x):
        with x.graph.gradient_override_map({"Identity": op_name}):
            x = tf.identity(x)
        return x

    return inner


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner

def _rand_select(xs, seed=None):
    xs_list = list(xs)
    rand_n = tf.random_uniform((), 0, len(xs_list), "int32", seed=seed)
    return tf.constant(xs_list)[rand_n]