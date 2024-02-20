# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:34:08 2024

@author: santhoshi_t
"""

bitres = 11


from osgeo import gdal
from osgeo.gdalconst import  *
from datetime import datetime
import numpy as np
import glob
from skimage import feature


bitres = 11
kernel_width = 9
kernel_size = (kernel_width - 1) // 2
ang = [0, np.pi/4, np.pi/2, 3*np.pi/4]
pis = ['0pib4', '1pib4', '2pib4', '3pib4']
lev = 32
dist = 4

gdal.PushErrorHandler('CPLQuietErrorHandler')

def read_gdal(fn1):
    gdal_file  = gdal.Open(fn1, GA_ReadOnly)
    data_mat = []
    for band_idx in range(gdal_file.RasterCount):
        bn = gdal_file.GetRasterBand(band_idx+1)
        data_mat.append(bn.ReadAsArray().astype(np.uint16))
    data_mat = np.stack(data_mat,axis=2)
    return data_mat

def write_img(prediction, filename, predpath):
    ds = gdal.Open(filename)
    band_ds = ds.GetRasterBand(1)    
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xsize = band_ds.XSize
    ysize = band_ds.YSize
    
    driver = gdal.GetDriverByName('GTiff')
    new_tiff = driver.Create(predpath, xsize, ysize, 1, gdal.GDT_Byte)
    new_tiff.SetGeoTransform(geotransform)
    new_tiff.SetProjection(projection)
    new_tiff.GetRasterBand(1).SetNoDataValue(-1)
    new_tiff.GetRasterBand(1).WriteArray(prediction)
    new_tiff.FlushCache()
    new_tiff = None
    driver = None
    
def get_glcmprops_chip(chip):
    glcm = feature.graycomatrix(chip, distances=[dist], angles=ang, symmetric=True, normed=True, levels=lev)
    contrast = feature.graycoprops(glcm, prop='contrast')[0]
    dissimilarity = feature.graycoprops(glcm, prop='dissimilarity')[0]
    homogeneity = feature.graycoprops(glcm, prop='homogeneity')[0]
    energy = feature.graycoprops(glcm, prop='energy')[0]
    correlation = feature.graycoprops(glcm, prop='correlation')[0]
    return contrast, dissimilarity, homogeneity, energy, correlation


infiles = glob.glob('../input/*.tif')

for inpath in infiles:
    img = read_gdal(inpath)
    
    for band in range(img.shape[2]):
        bandstart = datetime.now()
        contrast = np.zeros((img.shape[0], img.shape[1], len(ang)))
        dissimilarity = np.zeros((img.shape[0], img.shape[1], len(ang)))
        homogeneity = np.zeros((img.shape[0], img.shape[1], len(ang)))
        energy = np.zeros((img.shape[0], img.shape[1], len(ang)))
        correlation = np.zeros((img.shape[0], img.shape[1], len(ang)))

        for i in range(kernel_size, img.shape[0]-kernel_size):
            for j in range(kernel_size, img.shape[1]-kernel_size):
                contrast[i,j], dissimilarity[i,j], homogeneity[i,j], energy[i,j], correlation[i,j] = get_glcmprops_chip(((img[i - kernel_size:i + kernel_size + 1, j - kernel_size:j + kernel_size + 1, band]/ (2**bitres - 1)) * (lev-1)).astype(np.uint8))
        bandend = datetime.now()
        print('GLCM matrix size:', lev,'x',lev)
        print('Time taken without any parallelization is ', bandend - bandstart)
        break
    
        # uncomment to save
        # contrast_pathpath = inpath[:-4]+f'contrast_kernel{kernel_width}_band{band+1}.tif'
        # dissimilarity_pathpath = inpath[:-4]+f'dissimilarity_kernel{kernel_width}_band{band+1}.tif'
        # homogeneity_pathpath = inpath[:-4]+f'homogeneity_kernel{kernel_width}_band{band+1}.tif'
        # energy_pathpath = inpath[:-4]+f'energy_kernel{kernel_width}_band{band+1}.tif'
        # correlation_pathpath = inpath[:-4]+f'correlation_kernel{kernel_width}_band{band+1}.tif'
        # write_img(contrast, inpath, contrast_pathpath)
        # write_img(dissimilarity, inpath, dissimilarity_pathpath)
        # write_img(homogeneity, inpath, homogeneity_pathpath)
        # write_img(energy, inpath, energy_pathpath)
        # write_img(correlation, inpath, correlation_pathpath)
        