"""


"""


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
        
        self.images = images
        self.shifts = shifts
        self.scalar = scalar
        self.mask = mask

    @property
    def shifts(self):
        return self._shifts

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
            assert (val.shape == self.images.shape) or , 
            if (val.shape == self.images.shape):
                self._mask = np.asarray(val)
            elif (val.shape == self.fsh):
                self._mask = N * [np.asarray(val)]
            else:
                raise RuntimeError("Invalid mask shape")
            # Create frame mask
            self._m0= np.zeros_like(self._mask[0])
            self._m0[self._crop:-self._crop, self._crop:-self._crop] = 1.

    
    def refine_shifts(self, ref, ref_mask, max_shift):
        """

        """
        for i in range(self.n):
            result = match(Ib=ref, It=self.images[i], Ifact=self._scalar,
                           mb = ref_mask, mt= self._mo * self._mask[i], scale=False, max_shift=max_shift)
            self._shifts[i] += np.round(result["r"]).astype(int)
        
    def refine_image(self, img, img_renorm):
        img *= 1e-6
        img_renorm.fill(1e-6)
        for i in range(self.n):
            i0, i1 = self._shifts[i]
            img[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]] += self.mask[i] * self.images[i] * self._scalar
            img_renorm[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]] += self.mask[i] * self._scalar ** 2
        img_mask = img_renorm > 1e-5
        img /= img_renorm + ~img_mask
        return img, img_mask

    def solve(self, refine_scalar=True, max_iter=50, max_shift=None):
        """

        """

        # Automatic determination of max_shift
        auto_max_shift = False
        if max_shift is None:
            max_shift = 300
            auto_max_shift = True

        # Initial alignment based on one frame as a reference
        if not self._shifts_initialized:
            ref_img = self._images[0] / self._scalar
            self.refine_shifts(ref_img, self._m0 * self._mask[0], max_shift)

        # Initial estimate for image
        image, self._shifts, new_offset = reshape_array(np.zeros((10, 10)), self._shifts, self.fsh)
        self._offset += new_offset
        image_renorm = image.copy()
        image, image_renorm = self.refine_image(image, image_renorm)

        # Iteratively refine img and f
        scalar_renorm = np.zeros_like(self._scalar)

        # TODO: maybe provide a mask for flat as well
        scalar_mask = np.ones_like(self.mask[0])
        for m in self.mask:
            scalar_mask &= m

        # TODO: This is an assumption that needs to be documented:
        # The allowed maximum shift in cross-correlation fitting is twice the current shifts
        if auto_max_shift:
            max_shift = 2 * (self._shifts.max(axis=0) - self._shifts.min(axis=0)).max()
    
        refine_pos = True
        for ii in range(max_iter):

            # Find f
            if refine_scalar:
                #fbefore = f.copy()
                self._scalar *= 1e-6
                scalar_renorm.fill(1e-6)
                if self._scalar is not None:
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
            self.refine_image(self, img, img_renorm)
            
            # Refine positions
            if refine_pos:
                # This filter is needed to avoid matching noise.
                # Not clear yet what is the optimal sigma.
                #img = gaussian_filter(img, 10.)

                old_shifts = self._shifts.copy()
                self.refine_shifts(image[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]],
                                   img_mask[i0:i0 + self.fsh[0], i1:i1 + self.fsh[1]], max_shift)
                if np.all(old_shifts == self._shifts):
                    # Convergence
                    refine_pos = False
                else:
                    img, self._shifts, new_offset = reshape_array(image, self._shifts, self.fsh)
                    self._offset += new_offset
                    img_renorm = reshape_array(image_renorm, self._shifts, self.fsh)[0]
                    img_mask = reshape_array(img_mask, self._shifts, self.fsh)[0]
        return image, self._scalar, self._shifts + self._offset

