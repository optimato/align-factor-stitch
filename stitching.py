import numpy as np
import ccshift
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
import h5py
import os
import sys
import ast
from scipy.signal import convolve


def reshape_array(a, shifts, sh):
    """
    Change the dimensions of 2d array 'a' to fit tightly the total viewport given by
    the list of shifts and view shape.
    Return a_new, shifts_new
    where a_new is the cropped and/or padded array, and shifts_new are the new equivalent shifts
    """
    shifts = np.asarray(shifts)
    min0, min1 = shifts.min(axis=0)
    max0, max1 = shifts.max(axis=0) + sh

    shifts_new = shifts - (min0, min1)

    ash = a.shape
    nash = (max0 - min0, max1 - min1)

    if ash == nash:
        return a.copy(), shifts_new, np.array([min0, min1])

    a_new = np.zeros(nash, dtype=a.dtype)
    s0 = max(min0, 0)
    new_s0 = max(-min0, 0)
    e0 = min(ash[0], nash[0] + min0)
    new_e0 = min(nash[0], ash[0] - min0)

    s1 = max(min1, 0)
    new_s1 = max(-min1, 0)
    e1 = min(ash[1], nash[1] + min1)
    new_e1 = min(nash[1], ash[1] - min1)

    a_new[new_s0:new_e0, new_s1:new_e1] = a[s0:e0, s1:e1]

    return a_new, shifts_new, np.array([min0, min1])


def merge_image_stack(frames, positions=None, flat=None, mask=None, refine_flat=True, max_iter=50, max_shift=None):
    """
    Merge a stack of image that have been collected by moving the samples at various positions

    :param frames: The stack of images
    :param positions: the list of pixel shifts. If None, will be extracted from frames
    :param flat: flat field (frame without the sample in)
    :param mask: binary mask of valid image pixels
    :param refine_flat: if True, flat will be recovered from the data. If False, provided flat is used.
    :return:
    """
    N = len(frames)
    fsh = frames[0].shape

    offset = np.array([0, 0])
    
    auto_max_shift = False
    if max_shift is None:
        max_shift = 300
        auto_max_shift = True

    # Crop out 10% of the edges - this could be removed or parametrised.
    mask_crop = min(fsh) // 10

    if mask is None:
        raise RuntimeError('Not yet implemented (need a lot of guesswork to generate it from the data')
        mask_threshold = 50  # For 1 s exposure time
        mask_medfilt_size = 20
        # Generate mask from flat
        if flat is None:
            # Generate mask from frame stack
            f = frames.mean(axis=0)
        else:
            f = flat
        mf = median_filter(f, mask_medfilt_size)
        mask = np.abs(f - mf) < mask_threshold * exp_time
        masks = N * [mask]

    # Check if mask is a stack of masks or a single one to be used for all
    mask = np.asarray(mask)
    if mask.ndim == 3:
        assert len(mask) == N
        masks = mask
    else:
        masks = N * [mask]

    # Create frame mask
    m0 = np.zeros_like(masks[0])
    m0[mask_crop:-mask_crop, mask_crop:-mask_crop] = 1.

    if positions is None:
        if flat is None:
            raise RuntimeError('No clever way of aligning the images without a flat has been found yet')
        # Initial alignment based on one frame as a reference
        img = frames[0] / flat
        positions = np.empty((N, 2), dtype=int)
        for i in range(N):
            result = ccshift.match(img, frames[i], flat, mtmp=m0 * masks[i], scale=False, mref=m0*masks[0], max_shift=max_shift)
            positions[i] = np.round(result['r']).astype(int)

    if not refine_flat:
        if flat is None:
            raise RuntimeError('flat must be refined if no flat is provided')

    # Initial estimate for img
    img, positions, new_offset = reshape_array(np.zeros((10, 10)), positions, fsh)
    offset += new_offset
    img_renorm = img.copy()
    for i in range(N):
        i0, i1 = positions[i]
        img[i0:i0 + fsh[0], i1:i1 + fsh[1]] += masks[i] * frames[i] * flat
        img_renorm[i0:i0 + fsh[0], i1:i1 + fsh[1]] += masks[i] * flat ** 2
    # Identify possible regions that were masked for all positions
    img_mask = (img_renorm != 0)
    # Normalise image
    img /= img_renorm + ~img_mask

    # Iteratively refine img and f
    alpha = 1.
    if flat is None:
        f = frames.mean(axis=0)
    else:
        f = flat.copy()
    f_renorm = np.zeros_like(f)

    # TODO: maybe provide a mask for flat as well
    fmask = np.ones_like(masks[0])
    for m in masks:
        fmask &= m

    # TODO: This is an assumption that needs to be documented:
    # The allowed maximum shift in cross-correlation fitting is twice the current shifts
    if auto_max_shift:
        max_shift = 2 * (positions.max(axis=0) - positions.min(axis=0)).max()

    #return img, img_mask, positions, flat, masks, frames
    
    refine_pos = True
    for ii in range(max_iter):

        # Find f
        if refine_flat:
            fbefore = f.copy()
            f *= 1e-6
            f_renorm.fill(1e-6)
            if flat is not None:
                f += fmask * alpha * flat
                f_renorm += alpha * fmask
            for i in range(N):
                i0, i1 = positions[i]
                f += masks[i] * frames[i] * img[i0:i0 + fsh[0], i1:i1 + fsh[1]]
                f_renorm += masks[i] * img[i0:i0 + fsh[0], i1:i1 + fsh[1]] ** 2
            f /= f_renorm
            # print ii, (np.abs(f-fbefore)**2).sum()
            # Here implement some breaking criterion

        # Find img
        img *= 1e-6
        img_renorm.fill(1e-6)
        for i in range(N):
            i0, i1 = positions[i]
            img[i0:i0 + fsh[0], i1:i1 + fsh[1]] += masks[i] * frames[i] * f
            img_renorm[i0:i0 + fsh[0], i1:i1 + fsh[1]] += masks[i] * f ** 2
        img /= img_renorm
        img_mask = img_renorm > 1e-5

        # Refine positions
        if refine_pos:
            # This filter is needed to avoid matching noise.
            # Not clear yet what is the optimal sigma.
            #img = gaussian_filter(img, 10.)

            old_positions = positions.copy()
            for i in range(N):
                i0, i1 = positions[i]
                result = ccshift.match(Ib=img[i0:i0 + fsh[0], i1:i1 + fsh[1]], It=frames[i], Ifact=f, mt=m0 * masks[i],
                                       scale=False, mb=img_mask[i0:i0 + fsh[0], i1:i1 + fsh[1]], max_shift=max_shift)
                positions[i] += np.round(result['r']).astype(int)
                # print '%d: %s -> %s' % (i, str([i0, i1]), str(positions[i].tolist()))
            if np.all(old_positions == positions):
                # Convergence
                refine_pos = False
            else:
                img, positions, new_offset = reshape_array(img, positions, fsh)
                offset += new_offset
                img_renorm = reshape_array(img_renorm, positions, fsh)[0]
                img_mask = reshape_array(img_mask, positions, fsh)[0]
    return img, f, positions + offset