import os
from os.path import join as pjoin
from os.path import exists as pexists
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path

#dossier_segmentation = Path("/home/jdrochmans/data/juliette/segmentations2/segmentations")

dossier_brainmask = "/home/jdrochmans/data/juliette/synthstrip_raw"
dossier_image_MNI = "/home/jdrochmans/data/juliette/training_label_maps"
label_map = [os.path.join(dossier_image_MNI, f) for f in os.listdir(dossier_image_MNI) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_image_MNI, f)) ]
output_seg = "/home/jdrochmans/data/juliette/training_label_maps/crop"

output_brainmask = "/home/jdrochmans/data/juliette/brainmask_crop"

def generate_brain_mask(label_map):
    
    for map in label_map:
        path_in_name = os.path.basename(map)
        name = path_in_name.split("_")[2].split('.')[0]
        img = nib.load(map)
        img_data = img.get_fdata()
        img_data[img_data>0] = 1
        brain_mask = nib.Nifti1Image(img_data,img.affine)
        nib.save(brain_mask, os.path.join(dossier_brainmask, f'Synthseg_{name}.nii.gz'))




#code de Maxence

def find_bounding_box(mask_data):
    """
    Finds the bounding box of the brain mask.
    """
    indices = np.where(mask_data > 0)
    x_min, x_max = np.min(indices[0]), np.max(indices[0])
    y_min, y_max = np.min(indices[1]), np.max(indices[1])
    z_min, z_max = np.min(indices[2]), np.max(indices[2])
    return x_min, x_max, y_min, y_max, z_min, z_max

def crop_to_brainmask(image_path, brainmask_path, image_output_path, brainmask_output_path):
    # Load lesion mask and aseg mask
    image_img = nib.load(image_path)
    brainmask_img = nib.load(brainmask_path)
    print(image_path)
    print(brainmask_path)
    image = image_img.get_fdata()
    brainmask = brainmask_img.get_fdata()
    if(image.shape != brainmask.shape):
         brainmask = np.transpose(brainmask, (1, 0, 2))

    assert image.shape == brainmask.shape, f"image shape {image.shape} does not correspond with brainmask shape {brainmask.shape}"

    x_min, x_max, y_min, y_max, z_min, z_max = find_bounding_box(brainmask)
    cropped_image = image[x_min:x_max, y_min:y_max, z_min:z_max]

    output_nifti = nib.Nifti1Image(cropped_image, affine=image_img.affine)
    nib.save(output_nifti, image_output_path)
 
    if not pexists(brainmask_output_path):
        cropped_brainmask = brainmask[x_min:x_max, y_min:y_max, z_min:z_max]

        output_nifti = nib.Nifti1Image(cropped_brainmask, affine=image_img.affine)
        nib.save(output_nifti, brainmask_output_path)
 # dossier pour brainmask /dir/scratchL/mwynen/data/cusl_wml/all/synthstrip_raw/
 # dossier avec les segmentations à mettre sur bajoo 
 #uniquement le sub8!
# brain_mask_files = [
#     os.path.join(dossier_brainmask, f)
#     for f in os.listdir(dossier_brainmask)
#     if f.endswith(('.nii', '.nii.gz'))
#     and os.path.isfile(os.path.join(dossier_brainmask, f))
# ]
# brain_files_sorted = sorted(brain_mask_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
# print(f'brain :{brain_files_sorted}')
# for i in range(len(all_files)):
#     path_in_name = os.path.basename(all_files_sorted[i])
#     name = path_in_name.split("_")[0]
    
#     path_in_name2 = os.path.basename(brain_files_sorted[i])
#     name2 = path_in_name.split("_")[0] 
#     output_path_seg = os.path.join(output_seg, f'{name}_seg')
#     output_path_brain = os.path.join(output_brainmask, f'{name}_brainmask')
#     print(name)
#     print(name2)
#     crop_to_brainmask(all_files_sorted[i],brain_files_sorted[i],output_path_seg,output_path_brain)
    

for i in range(len(label_map)):
    path_in_name = os.path.basename(label_map[i])
    name = path_in_name.split("_")[2].split('.')[0]
    output_path_seg = os.path.join(output_seg, f'Synthseg_{name}_seg')
    output_path_brain = os.path.join(output_brainmask, f'Synthseg_{name}_brainmask')
    generate_brain_mask(label_map)
    crop_to_brainmask(label_map[i],os.path.join(dossier_brainmask, f'Synthseg_{name}.nii.gz'),output_path_seg,output_path_brain)
    