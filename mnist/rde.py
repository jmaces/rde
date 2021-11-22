import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


# projected gradient decent with line search
def ProjectedGD(
    fun,
    x0,
    jac,
    maxiter=1500,
    eta=1e0,
    ftol=1e-18,
    xtol=1e-18,
    c=0.65,
    tau=0.7,
    maxls=30,
    momentum=0.0,
):
    """ Constrained minimization by projected gradient descent.

    Minimizes a function within a [0, 1]-box.
    """
    # init variables
    xk = x0
    fk, dfk = fun(xk), jac(xk)
    mk = 0.0 * dfk
    iter, stop = 0, [False, False, False, False]
    ls_init_step = eta
    # report and store pre optimization status
    print(
        "{:^10s}\t{:^10s}\t{:^10s}\t{:^10s}\t{:^10s}\t{:^10s}\t{:^10s}".format(
            "iter",
            "objective",
            "|gradient|",
            "x range",
            "sparsity",
            "ls steps",
            "step size",
        )
    )
    print(
        "{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}".format(
            *(7 * [10 * "-"])
        )
    )
    print(
        "{:10d}\t{:1.3e}\t{:1.3e}\t[{:.1f}, {:.1f}]\t{:1.3e}\t{:10s}\t{:10s}"
        "".format(
            iter,
            fk.squeeze(),
            np.mean(np.square(dfk)),
            xk.min(),
            xk.max(),
            np.sum(xk),
            "---",
            "---",
        )
    )
    # main loop
    while not (stop[0] or np.all(stop[1:])):
        xkold, fkold = xk, fk
        mk = momentum * mk + (1 - momentum) * dfk
        # backtracking line search
        ls_count, ls_step = 0, ls_init_step
        for ls_count in range(1, maxls + 1):
            # do projected gradient step
            xk = np.clip(xkold - ls_step * mk, 0.0, 1.0)
            fk = fun(xk)
            if fkold - fk >= c / ls_step * np.sum(np.square(xk - xkold)):
                break  # line search sucess
            ls_step *= tau
        # adapt initial step size guess if necessary
        if ls_count == 1:
            ls_init_step /= tau
        elif ls_count >= min(5, maxls):
            ls_init_step *= tau
        # report post step status and check stopping criteria
        dfk = jac(xk)
        iter += 1
        print(
            "{:10d}\t{:1.3e}\t{:1.3e}\t[{:.1f}, {:.1f}]\t{:1.3e}\t{:10d}"
            "\t{:1.3e}".format(
                iter,
                fk.squeeze(),
                np.mean(np.square(dfk)),
                xk.min(),
                xk.max(),
                np.sum(xk),
                ls_count,
                ls_step,
            )
        )
        stop = [
            iter >= maxiter,
            np.mean(np.square(fk - fkold))
            < ftol * min(np.mean(np.square(fkold)), np.mean(np.square(fk))),
            np.mean(np.square(xk - xkold))
            < xtol * min(np.mean(np.square(xkold)), np.mean(np.square(xk))),
            ls_count == maxls,
        ]
    print("stopping criterion:\t", stop)
    return xk.squeeze()


# squared distance distortion objective
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


# l1 regularizer for sparse relevance maps
def get_sparsity_regularizer():
    def g(s):
        return np.mean(np.abs(s))

    def dg(s):
        return 1 / np.prod(s.shape) * np.sign(s)

    return g, dg


# Lagrangian RDE with assumed density filtering model
def adf_rde_lagrangian(
    x, node, target, s0, mean, covariance, model, mu, mode="diag"
):
    f, df = get_distortion(x, node, target, mean, covariance, model, mode)
    g, dg = get_sparsity_regularizer()

    def obj(s):
        return f(s) + np.square(target) * mu * g(s)

    def dobj(s):
        return df(s) + np.square(target) * mu * dg(s)

    result = ProjectedGD(obj, s0.flatten(), jac=dobj, eta=1e0, momentum=0.85,)
    print("distortion pre:\t {:1.2e}".format(f(s0)))
    print("distortion post: {:1.2e}".format(f(result)))
    print("sparsity pre:\t {:1.2e}".format(g(s0)))
    print("sparsity post:\t {:1.2e}".format(g(result)))
    return np.reshape(result, x.shape)
