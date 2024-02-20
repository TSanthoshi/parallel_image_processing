# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:34:54 2024

@author: santhoshi_t
"""

from osgeo import gdal
from osgeo.gdalconst import  *
import os
import numpy as np
import glob
from skimage import feature
from multiprocessing import Pool, cpu_count
from datetime import datetime

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
    data_mat = (np.stack(data_mat,axis=2) / (2**bitres - 1) ) * (lev-1) 
    return data_mat.astype(np.uint8)

def write_img(prediction, filename, predpath):
    ds = gdal.Open(filename)
    band_ds = ds.GetRasterBand(1)    
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xsize = band_ds.XSize
    ysize = band_ds.YSize
    
    driver = gdal.GetDriverByName('GTiff')
    new_tiff = driver.Create(predpath, xsize, ysize, 1, gdal.GDT_Float32)
    new_tiff.SetGeoTransform(geotransform)
    new_tiff.SetProjection(projection)
    new_tiff.GetRasterBand(1).SetNoDataValue(-1)
    new_tiff.GetRasterBand(1).WriteArray(prediction)
    new_tiff.FlushCache()
    new_tiff = None
    driver = None
    
def calculate_glcm_properties(args):
    i, j, img_slice = args
    glcm = feature.graycomatrix(img_slice, distances=[dist], angles=ang, symmetric=True, normed=True, levels=lev)
    contrast = feature.graycoprops(glcm, prop='contrast')[0]
    dissimilarity = feature.graycoprops(glcm, prop='dissimilarity')[0]
    homogeneity = feature.graycoprops(glcm, prop='homogeneity')[0]
    energy = feature.graycoprops(glcm, prop='energy')[0]
    correlation = feature.graycoprops(glcm, prop='correlation')[0]
    return i, j,contrast, dissimilarity, homogeneity, energy, correlation

def process_image_band(args):
    i, j, arr = args
    
    img_slice = arr[i - kernel_size:i + kernel_size + 1, j - kernel_size:j + kernel_size + 1]
    return i, j, img_slice

def parallel_glcm_computation(arr):
    num_processes = cpu_count() - 3 # change as required and available
    

    def get_indices():
        for i in range(kernel_size, arr.shape[0] - kernel_size):
            for j in range(kernel_size, arr.shape[1] - kernel_size):
                yield i, j, arr

    with Pool(processes=num_processes) as pool:
        img_slices = pool.map(process_image_band, get_indices())
        results = pool.map(calculate_glcm_properties, img_slices)

    contrast = np.zeros((arr.shape[0],arr.shape[1],len(ang)))
    dissimilarity = np.zeros((arr.shape[0],arr.shape[1],len(ang)))
    homogeneity = np.zeros((arr.shape[0],arr.shape[1],len(ang)))
    energy = np.zeros((arr.shape[0],arr.shape[1],len(ang)))
    correlation = np.zeros((arr.shape[0],arr.shape[1],len(ang)))

    for i, j, c, d, h, e, corr in results:
        contrast[i, j, :] = c
        dissimilarity[i, j, :] = d
        homogeneity[i, j, :] = h
        energy[i, j, :] = e
        correlation[i, j, :] = corr

    return contrast, dissimilarity, homogeneity, energy, correlation


if __name__ == '__main__':
    infiles = glob.glob('../input/*.tif')
    
    for inpath in infiles:
        outpath = os.path.join(os.path.dirname(inpath), 'params_lev32_dist4')
        if not os.path.exists(outpath):
            os.makedirs(outpath, exist_ok = False) 

        img = read_gdal(inpath)
        
        
        for band in range(img.shape[2]):
            bandstart = datetime.now()
            c, d, h, e, corr = parallel_glcm_computation(((img[:,:, band]/ (2**bitres - 1)) * (lev-1)).astype(np.uint8))
            bandend = datetime.now()
            print('GLCM matrix size:', lev,'x',lev)
            print('Time taken with multiprocessing is ', bandend - bandstart)
            break
            
            # uncomment to save
            # for i, angle in enumerate(ang):
            #     contrast_path = os.path.join(outpath, os.path.basename(inpath)[:-4]+f'_con_ker{kernel_width}_ang{pis[i]}_band{band+1}.tif')
            #     dissimilarity_path = os.path.join(outpath, os.path.basename(inpath)[:-4]+f'_dis_ker{kernel_width}_ang{pis[i]}_band{band+1}.tif')
            #     homogeneity_path = os.path.join(outpath, os.path.basename(inpath)[:-4]+f'_hom_ker{kernel_width}_ang{pis[i]}_band{band+1}.tif')
            #     energy_path = os.path.join(outpath, os.path.basename(inpath)[:-4]+f'_ener_ker{kernel_width}_ang{pis[i]}_band{band+1}.tif')
            #     correlation_path = os.path.join(outpath, os.path.basename(inpath)[:-4]+f'_corr_ker{kernel_width}_ang{pis[i]}_band{band+1}.tif')
            #     write_img(c[:,:,i], inpath, contrast_path)
            #     write_img(d[:,:,i], inpath, dissimilarity_path)
            #     write_img(h[:,:,i], inpath, homogeneity_path)
            #     write_img(e[:,:,i], inpath, energy_path)
            #     write_img(corr[:,:,i], inpath, correlation_path)





