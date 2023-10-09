"""


"""
from .utils import reshape_to_view, match

import numpy as np

class AlignFactorStitch():
    """


    """
    def __init__(self, images, shifts=None, scalar=None, mask=None, crop=0.1):
        """
        
        An iterative solver to align, factor and stitch a stack of N images each of shape (X,Y).
        
        :param images: The stack of images
        :type images: ndarray of shape (N,X,Y)
        :param shifts: A list of pixel shifts. If None, will be extracted by the solver.
        :type shifts: 2d array of shape (N,2)
        :param scalar: A image of scalars, equivalent to flat field for some applications.
        :type scalar: 2d array of shape (X,Y)
        :param mask: A binary mask of valid image pixels
        :type mask: 2d array of shape (X,Y) or 3d array of shape (N,X,Y)
        :param crop: Percentage to be cropped at the edges, default is 10 percent.
        :type crop: int

        """
        self.n = len(images)
        self.fsh = images[0].shape
        self._offset = np.array([0, 0])
        self._crop = int(min(self.fsh) * crop) #TODO: expand to both dimensions
        self._alpha = 1.
        self._shifts = None
        self._scalar = None
        self._mask = None

        # stitched image
        self._img = None
        self._img_renorm = None
        self._img_mask = None

        # properties
        self.images = images
        self.shifts = shifts
        self.scalar = scalar
        self.mask = mask
        
        self.initialize()

    @property
    def stitched(self):
        if self._img is not None:
            return self._img
        
    @property
    def shifts(self):
        return self._shifts + self._offset

    @shifts.setter
    def shifts(self, val):
        if val is None:
            self._shifts = np.zeros((self.n, 2), dtype=int)
            self._shifts_initialized = False
        else:
            assert val.shape == (self.n,2), "Provided shifts should have shape (N,2)"
            self._shifts = val
            self._shifts_initialized = True
                
    @property
    def scalar(self):
        return self._scalar

    @scalar.setter
    def scalar(self, val):
        if val is None:
            # Alternatively, this could be np.ones???
            self._scalar = np.median(self.images, axis=0)
        else:
            assert (val.shape) == (self.fsh), "Provided scalar image should have shape {}".format(self.fsh)
            self._scalar = val
        self._scalar_renorm = np.zeros_like(self._scalar)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        if val is None:
            raise RuntimeError('Not yet implemented (need a lot of guesswork to generate it from the data')
            # mask_threshold = 50  # For 1 s exposure time
            # mask_medfilt_size = 20
            # # Generate mask from flat
            # if flat is None:
            #     # Generate mask from frame stack
            #     f = frames.mean(axis=0)
            # else:
            #     f = flat
            # mf = median_filter(f, mask_medfilt_size)
            # mask = np.abs(f - mf) < mask_threshold * exp_time
            # masks = N * [mask]
        else:
            if (val.shape == self.images.shape):
                self._mask = np.asarray(val)
            elif (val.shape == self.fsh):
                self._mask = self.n * [np.asarray(val)]
            else:
                raise RuntimeError("Invalid mask shape")
            # Create frame mask
            self._m0= np.zeros_like(self._mask[0])
            self._m0[self._crop:-self._crop, self._crop:-self._crop] = 1.
        
    def _refine_image(self):
        if self._img is None:
            return
        self._img *= 1e-6
        self._img_renorm.fill(1e-6)
        for i in range(self.n):
            i0, i1 = self._shifts[i]
            self._img[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]] += self.mask[i] * self.images[i] * self._scalar
            self._img_renorm[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]] += self.mask[i] * self._scalar ** 2
        self._img_mask = (self._img_renorm != 0)#self._img_renorm > 1e-5
        self._img /= self._img_renorm + ~self._img_mask

    def _refine_scalar(self):
        for i in range(self.n):
            i0, i1 = self._shifts[i]
            self._scalar += self._mask[i] * self.images[i] * self._img[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]]
            self._scalar_renorm += self.mask[i] * self._img[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]] ** 2
        self._scalar /= self._scalar_renorm

    def initialize(self):
        """

        """
        # Initial alignment based on one frame as a reference
        if not self._shifts_initialized:
            ref_img = self.images[0] / self._scalar
            for i in range(self.n):
                result = match(Ib=ref_img, It=self.images[i], Ifact=self._scalar,
                               mb=self._m0 * self._mask[0], mt=self._m0 * self._mask[i],
                               scale=False, max_shift=max_shift)
                self._shifts[i] += np.round(result["r"]).astype(int)

        # Initial estimate for image
        self._img, self._shifts, new_offset = reshape_to_view(np.zeros((10, 10)), self._shifts, self.fsh)
        self._offset += new_offset
        self._img_renorm = self._img.copy()
        self._refine_image()

    def solve(self, refine_scalar=True, max_iter=50, max_shift=None):
        """

        """

        # Automatic determination of max_shift
        auto_max_shift = False
        if max_shift is None:
            max_shift = 300
            auto_max_shift = True

        # # TODO: maybe provide a mask for flat as well
        scalar_mask = np.ones_like(self.mask[0])
        for m in self.mask:
            scalar_mask &= m

        # TODO: This is an assumption that needs to be documented:
        # The allowed maximum shift in cross-correlation fitting is twice the current shifts
        if auto_max_shift:
            max_shift = 2 * (self._shifts.max(axis=0) - self._shifts.min(axis=0)).max()
    
        refine_shifts = True
        for ii in range(max_iter):

            # Find f
            if refine_scalar:
                #fbefore = f.copy()
                self._scalar *= 1e-6
                self._scalar_renorm.fill(1e-6)
                # BD: is this necessary? here flat is the user-provided scalar image
                if self._scalar is not None:
                    self._scalar += scalar_mask * self._alpha * self._scalar
                    self._scalar_renorm += self._alpha * scalar_mask
                self._refine_scalar()
                # print ii, (np.abs(f-fbefore)**2).sum()
                # Here implement some breaking criterion

            # Find img
            self._refine_image()
            
            # Refine positions
            if refine_shifts:
                prev_shifts = self._shifts.copy()
                for i in range(self.n):
                    i0,i1 = self._shifts[i]
                    result = match(Ib=self._img[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]],
                                   It=self.images[i], Ifact=self._scalar,
                                   mb=self._img_mask[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]],
                                   mt=self._m0 * self._mask[i],
                                   scale=False, max_shift=max_shift)
                    self._shifts[i] += np.round(result["r"]).astype(int)
                if np.all(prev_shifts == self._shifts):
                    # Convergence
                    refine_shifts = False
                else:
                    self._img, self._shifts, new_offset = reshape_to_view(self._img, self._shifts, self.fsh)
                    self._offset += new_offset
                    self._img_renorm = reshape_to_view(self._img_renorm, self._shifts, self.fsh)[0]
                    self._img_mask = reshape_to_view(self._img_mask, self._shifts, self.fsh)[0]

