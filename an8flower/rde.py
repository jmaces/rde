import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.optimize import Bounds, minimize


def get_distortion(x, node, target, mean, covariance, model, mode="diag"):
    x_tensor = tf.constant(x, dtype=tf.float32)
    m_tensor = tf.constant(mean, dtype=tf.float32)
    c_tensor = tf.constant(covariance, dtype=tf.float32)
    s_flat = tf.placeholder(tf.float32, (np.prod(x_tensor.shape),))
    s_tensor = tf.reshape(s_flat, x.shape)
    mean_in = s_tensor * x_tensor + (1 - s_tensor) * m_tensor
    if mode == "diag":
        covariance_in = tf.square(1 - s_tensor) * c_tensor
    elif mode == "half":
        covariance_in = c_tensor * (1 - s_tensor)
    elif mode == "full":
        covrank = len(c_tensor.get_shape().as_list())
        perm = (
            [0]
            + list(range((covrank - 1) // 2 + 1, covrank))
            + list(range(1, (covrank - 1) // 2 + 1))
        )
        covariance_in = c_tensor * (1 - s_tensor)
        covariance_in = K.permute_dimensions(covariance_in, perm,)
        covariance_in = covariance_in * (1 - s_tensor)
        covariance_in = K.permute_dimensions(covariance_in, perm,)
    out_mean, out_covariance = model([mean_in, covariance_in])
    if mode == "diag":
        loss = (
            1
            / 2
            * (
                K.mean(K.square(out_mean[..., node] - target))
                + K.mean(out_covariance[..., node])
            )
        )
    elif mode == "half":
        out_covariance = K.sum(K.square(out_covariance), axis=1)
        loss = (
            1
            / 2
            * (
                K.mean(K.square(out_mean[..., node] - target))
                + K.mean(out_covariance[..., node])
            )
        )
    elif mode == "full":
        loss = (
            1
            / 2
            * (
                K.mean(K.square(out_mean[..., node] - target))
                + K.mean(out_covariance[..., node, node])
            )
        )
    gradient = K.gradients(loss, [s_flat])[0]
    f_out = K.function([s_flat], [loss])
    f_gradient = K.function([s_flat], [gradient])
    return lambda s: f_out([s])[0], lambda s: f_gradient([s])[0]


def get_sparsity_regularizer():
    def g(s):
        return np.mean(np.abs(s))

    def dg(s):
        return 1 / np.prod(s.shape) * np.sign(s)

    return g, dg


def get_bounds():
    return Bounds(0.0, 1.0)


def adf_rde_lagrangian(
    x, node, target, s0, mean, covariance, model, mu, mode="diag"
):
    f, df = get_distortion(x, node, target, mean, covariance, model, mode)
    g, dg = get_sparsity_regularizer()

    def obj(s):
        return f(s) + np.square(target) * mu * g(s)

    def dobj(s):
        return df(s) + np.square(target) * mu * dg(s)

    result = minimize(
        obj,
        s0.flatten(),
        jac=dobj,
        method="l-bfgs-b",  # alternatively 'tnc'
        bounds=get_bounds(),
        options={"disp": True, "verbose": 2},
    )
    if not result.success:
        print("Warning: Optimization failed. Result might be useless.")
        print(result.message)

    print("distortion pre:\t {:1.2e}".format(f(s0)))
    print("distortion post: {:1.2e}".format(f(result.x)))
    print("sparsity pre:\t {:1.2e}".format(g(s0)))
    print("sparsity post:\t {:1.2e}".format(g(result.x)))
    return np.reshape(result.x, x.shape)
