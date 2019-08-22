# Collection of different usefull functions
from skimage import exposure
import pydicom
import pytorch_tools
# from pytorch_tools.utils import rle_decode

def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.
        
    Returns:
        dict: contains metadata of relevant fields.
    """
    
    data = {}
    
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
        encoded_pixels_list = rles_df[rles_df['ImageId']==dicom_data.SOPInstanceUID]['EncodedPixels'].values
       
        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != ' -1':
                pneumothorax = True
        
        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)
        
    return data

def rle_decode(rle_str, shape, fill_value=1, dtype=int, relative=False):
    
    """
    Args:
        rle_str (str): rle string
        shape (Tuple[int, int]): shape of the output mask
        relative: if True, rle_str is relative encoded string
    """
    s = rle_str.strip().split(" ")
    starts, lengths = np.array([np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])])
    mask = np.zeros(np.prod(shape), dtype=dtype)
    if relative:
        start = 0
        for index, length in zip(starts, lengths):
            start = start + index
            end = start + length
            mask[start: end] = fill_value
            start = end
        return mask.reshape(shape[::-1]).T
    else:
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        return mask.reshape(shape[::-1]).T