import numpy as np

sigma_jitter = 10
sigma_scaling = 0.3

def same(x):
    return x+0

def jitter(x, sigma=sigma_jitter):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=(4097,))


def scaling(x, sigma=sigma_scaling):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma)
    return np.multiply(x, factor)


def rotation(x):
    flip = np.random.choice([-1, 1], size=(100, 1))
    rotate_axis = np.arange(1)
    np.random.shuffle(rotate_axis)
    x_aug_newax = x[np.newaxis, :, np.newaxis]
    x_rotation = flip[:, np.newaxis, :] * x_aug_newax[:, :, rotate_axis]
    x_rotation = x_rotation[0][:,0]
    return x_rotation


def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(4097)

    num_segs = np.random.randint(1, max_segments, size=(4097,))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(4097 - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat
        else:
            ret[i] = pat

    return ret

