"""
Collection of different usefull functions
"""

from skimage import exposure
import pydicom
import pytorch_tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches


def show_dicom_info():
    print("Filename.........:", file_path)
    dicom_data = pydicom.dcmread(file_path)
    print("Instance UID......:", dicom_data.SOPInstanceUID)
    print("Storage type.....:", dicom_data.SOPClassUID)
    print()

    pat_name = dicom_data.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dicom_data.PatientID)
    print("Patient's Age.......:", dicom_data.PatientAge)
    print("Patient's Sex.......:", dicom_data.PatientSex)
    print("Modality............:", dicom_data.Modality)
    print("Body Part Examined..:", dicom_data.BodyPartExamined)
    print("View Position.......:", dicom_data.ViewPosition)
    
    if 'PixelData' in dicom_data:
        rows = int(dicom_data.Rows)
        cols = int(dicom_data.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dicom_data.PixelData)))
        if 'PixelSpacing' in dicom_data:
            print("Pixel spacing....:", dicom_data.PixelSpacing)


def dicom_to_dict(file_path, rles_df, encoded_pixels=True, ):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): Search for annotation.
        
    Returns:
        dict: contains metadata of relevant fields.
    """
    
    data = {}
    dicom_data = pydicom.dcmread(file_path)
    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID
    
    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
        assert len(encoded_pixels) == 1, "Use rle-train-filtered.csv"
        pneumothorax = False if encoded_pixels == ' -1' else True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data

def plot_with_mask(file_path, mask_encoded, figsize=(20,10), relative=True):    
    """Plot Chest Xray image with mask (annotation or label) and without mask.

    Args:
        file_path (str): file path of the dicom data.
        mask_encoded : Pandas dataframe of the RLE.
        
    Returns:
        plots the image with and without mask.
    """
    
    pixel_array = pydicom.dcmread(file_path).pixel_array
    print(np.max(pixel_array))
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    clahe_pixel_array = clahe.apply(pixel_array)
    adapteq_pixel_array = exposure.equalize_adapthist(pixel_array, clip_limit=0.03)
    
    # use the masking function to decode RLE
    mask_decoded = rle_decode(mask_encoded, (1024, 1024), relative=relative)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(20,10))
    
    # print out the xray
    ax[0].imshow(pixel_array, cmap=plt.cm.bone)
    # print out the annotated area
    ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
    ax[0].set_title('With Mask')
    
    # plot image with clahe processing and no mask
    ax[1].imshow(clahe_pixel_array, cmap=plt.cm.bone)
    ax[1].set_title('Without Mask - Clahe')
    
    # plot image with adapteq processing and no mask
    ax[2].imshow(adapteq_pixel_array, cmap=plt.cm.bone)
    ax[2].set_title('Without Mask - Adapteq')
    plt.show()