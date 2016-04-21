import numpy
from matplotlib import pyplot as plt
import smooth_map

meerkat=plt.imread('meerkat_small.jpg')

smoothed_map=numpy.zeros(meerkat.shape)
unsmoothed_map=numpy.zeros(meerkat.shape)
npix_smooth=3.5
npix_restore=4
for i in range(3):
    tmp=numpy.squeeze(meerkat[:,:,i])
    tmp_smooth=smooth_map.smooth_map(tmp,npix_smooth)
    smoothed_map[:,:,i]=tmp_smooth
    tmp2=smooth_map.smooth_map(tmp_smooth,npix_restore,False)
    unsmoothed_map[:,:,i]=tmp2

plt.imsave('meerkat_smoothed_' + repr(npix_smooth) + '.jpg',smoothed_map/smoothed_map.max())
plt.imsave('meerkat_unsmoothed_' + repr(npix_restore) + '.jpg',unsmoothed_map/unsmoothed_map.max())






#plt.clf()
#plt.imshow(smoothed_map/smoothed_map.max())
#plt.savefig('meerkat_smoothed_' + repr(npix_smooth) + '.jpg')

#plt.clf()
#plt.imshow(unsmoothed_map/unsmoothed_map.max())
#plt.savefig('meerkat_unsmoothed_' + repr(npix_restore) + '.jpg')

