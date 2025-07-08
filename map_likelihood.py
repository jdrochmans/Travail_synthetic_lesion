#import mne
import numpy as np
import nibabel as nib
import nilearn
#from nilearn.datasets import (load_mni152_template)
import os
import re
from nilearn import plotting, image
import matplotlib as plt
import ants
from fill_img import lesion_fill
import shutil
import time
import subprocess
from nilearn.image import resample_to_img


dossier_map_likelihood = "/home/jdrochmans/data/juliette/"
dossier_mask = "/home/jdrochmans/data/juliette/Dataset001_BrainLesion/labelsTr"
dossier_image = "/home/jdrochmans/data/juliette/Dataset001_BrainLesion/imagesTr"

dossier_WM = '/home/jdrochmans/data/juliette/WM_mask'

template_p_T1 = "/home/jdrochmans/data/juliette/template.nii"

dossier_registered_mask = "/home/jdrochmans/data/juliette/register_mask/"
dossier_segmentation = "/home/jdrochmans/data/juliette/seg"
dossier_lesion_50_100 = '/home/jdrochmans/data/juliette/labels-database/50-100'
dossier_lesion_100_500 = '/home/jdrochmans/data/juliette/labels-database/50-100'
dossier_lesion_500_1000 = '/home/jdrochmans/data/juliette/labels-database/500-1000'
dossier_lesion_1000_5000 = '/home/jdrochmans/data/juliette/labels-database/1000-5000'
dossier_lesion_5000_more = '/home/jdrochmans/data/juliette/labels-database/5000-more'

save_dir_transforms = '/home/jdrochmans/data/juliette/transforms_reg/'
os.makedirs(save_dir_transforms, exist_ok=True) 


def likelihood(dossier_mask, dossier_segmentation, dossier_image, template_p, dossier_lesion_50_100, dossier_lesion_100_500, dossier_lesion_500_1000, dossier_lesion_1000_5000, dossier_lesion_5000_more, dossier_map_likelihood):
    """
    Build and save a normalized 3D likelihood map by aggregating masks from MS-MIST and our private database.

    Parameters
    ----------
    dossier_mask : Path to the directory of segmented lesions (masks in our private DS).
    dossier_segmentation : Path to the directory of brain tissue segmentations associated to images from our private DS.
    dossier_image : Path to the directory of FLAIR images from our private dataset.
    template_p : Path to the template T1-weighted NIfTI file used as registration target.
    dossier_lesion_50->more : dossiers containing segmented lesions from MS-MIST, classed by the total lesion weight in the masks.
    
    Returns
    -------
    None
        The function writes out a single normalized likelihood map:
          `likelihood_map_norm_WM30.nii` in `dossier_map_likelihood`.
    """
    
    dossiers_lesions = [
    dossier_lesion_50_100,
    dossier_lesion_100_500,
    dossier_lesion_500_1000,
    dossier_lesion_1000_5000,
    dossier_lesion_5000_more
]
    mask_files = [os.path.join(dossier_mask, f) for f in os.listdir(dossier_mask) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_mask, f))and 'mask-instances' in f]
    image_files = [os.path.join(dossier_image, f) for f in os.listdir(dossier_image) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_image, f))and 'FLAIR' in f.upper()]
    seg_files = [os.path.join(dossier_segmentation, f) for f in os.listdir(dossier_segmentation) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_segmentation, f))]
    mask_files_sorted = sorted(mask_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    image_files_sorted = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    seg_files_sorted = sorted(seg_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    template_T1 = ants.image_read(template_p)
    template_nib_T1 = nib.load(template_p)
    map = np.zeros(template_T1.numpy().shape)
    count = 0
    number_data = len(mask_files)
    for i in range(number_data) :
        
        path_in_name = os.path.basename(mask_files_sorted[i])
        name = path_in_name.split("_")[0]
        mask_reg = [os.path.join(dossier_registered_mask, f) 
        for f in os.listdir(dossier_registered_mask) 
        if f.startswith(f"{name}")]
        seg_mask = nib.load(seg_files_sorted[i]).get_fdata()
        ref_img = nib.load(seg_files_sorted[i])
        spacing = ref_img.header.get_zooms()
        origin = ref_img.affine[:3,3] 
        direction = np.eye(3)
        WM_mask = np.isin(seg_mask, [41, 2, 46, 7]).astype(np.uint8)
        WM_mask_cereb = np.isin(seg_mask, [46, 7]).astype(np.uint8)
        if(mask_reg == []):
            transformations_fwd = sorted(
            [os.path.join(save_dir_transforms, f) 
            for f in os.listdir(save_dir_transforms) 
            if f.startswith(f"{name}_") and
            (re.search(r"_0GenericAffine\.mat$", f) or
            re.search(r"_1Warp\.nii(\.gz)?$", f))
            ]
        )
            if transformations_fwd :
                output =  os.path.join(dossier_registered_mask, f'{name}.nii.gz')
                cmd = f"antsApplyTransforms -i {mask_files_sorted[i]} -r {template_p} -n {'genericLabel'} -t {transformations_fwd[1]} -t {transformations_fwd[0]} -o {output}"
                subprocess.Popen(cmd, shell = True).wait()
                aligned_mask = ants.image_read(output)
                mask_p = output
            else : 
                print(f"No transformations found for {name}, registration of the image needed")

                image_filled_p = lesion_fill(mask_files_sorted[i],image_files_sorted[i])
                subprocess.Popen(
                    f"antsRegistrationSyN.sh -f {template_p} -m {image_filled_p} -t s -o {save_dir_transforms}{name}_",
                    shell=True
                ).wait()

                transformations_fwd = sorted(
                [os.path.join(save_dir_transforms, f) 
                for f in os.listdir(save_dir_transforms) 
                if f.startswith(f"{name}_") and
                (re.search(r"_0GenericAffine\.mat$", f) or
                re.search(r"_1Warp\.nii(\.gz)?$", f))
            ]
            )
               
                output =  os.path.join(dossier_registered_mask, f'mask_{name}.nii.gz')
                cmd = f"antsApplyTransforms -i {mask_files_sorted[i]} -r {template_p} -n {'NearestNeighbor'} -t {transformations_fwd[1]} -t {transformations_fwd[0]} -o {output}"
                subprocess.Popen(cmd, shell = True).wait()
                aligned_mask = ants.image_read(os.path.join(dossier_registered_mask,name))
                
        else : 
            aligned_mask = ants.image_read(mask_reg[0])
            mask_p = mask_reg[0]
        WM_reg = [os.path.join(dossier_WM, f) 
        for f in os.listdir(dossier_WM) 
        if f.startswith(f"mask_WM_registered_{name}")]
        if(WM_reg == []):
            transformations_fwd = sorted(
                [os.path.join(save_dir_transforms, f) 
                for f in os.listdir(save_dir_transforms) 
                if f.startswith(f"{name}_") and
                (re.search(r"_0GenericAffine\.mat$", f) or
                re.search(r"_1Warp\.nii(\.gz)?$", f))
            ])
            WM_mask_ants = ants.from_numpy(WM_mask,
                            spacing=spacing,
                            origin=origin,
                            direction=direction)
            path_WM_patient_ref = os.path.join(dossier_WM,f'mask_WM_patient_{name}.nii.gz')
            ants.image_write(WM_mask_ants, path_WM_patient_ref)
            output_WM =  os.path.join(dossier_WM, f'mask_WM_registered_{name}.nii.gz')
            cmd = f"antsApplyTransforms -i {path_WM_patient_ref} -r {template_p} -n {'genericLabel'} -t {transformations_fwd[1]} -t {transformations_fwd[0]} -o {output_WM}"
            subprocess.Popen(cmd, shell = True).wait()
            WM_mask_reg = ants.image_read(output_WM)
            WM_mask_p = output_WM
        else : 
            WM_mask_reg = ants.image_read(WM_reg[0])
            WM_mask_reg_p = WM_reg[0]
        
        WM_cereb_reg = [os.path.join(dossier_WM, f) 
        for f in os.listdir(dossier_WM) 
        if f.startswith(f"mask_WM_cereb_registered_{name}")]
        if(WM_cereb_reg == []):
            transformations_fwd = sorted(
                [os.path.join(save_dir_transforms, f) 
                for f in os.listdir(save_dir_transforms) 
                if f.startswith(f"{name}_") and
                (re.search(r"_0GenericAffine\.mat$", f) or
                re.search(r"_1Warp\.nii(\.gz)?$", f))
            ])    
            WM_cereb_nifti = nib.Nifti1Image(WM_mask_cereb, ref_img.affine)
            path_WM_cereb_patient_ref = os.path.join(dossier_WM,f'mask_WM_cereb_patient_{name}.nii.gz')
            nib.save(WM_cereb_nifti,path_WM_cereb_patient_ref)
            output_WM_cereb =  os.path.join(dossier_WM, f'mask_WM_cereb_registered_{name}.nii.gz')
            cmd = f"antsApplyTransforms -i {path_WM_cereb_patient_ref} -r {template_p} -n {'genericLabel'} -t {transformations_fwd[1]} -t {transformations_fwd[0]} -o {output_WM_cereb}"
            subprocess.Popen(cmd, shell = True).wait()
            WM_cereb_mask_reg = ants.image_read(output_WM_cereb)
            WM_mask_p = output_WM_cereb
        else : 
            WM_cereb_mask_reg = ants.image_read(WM_cereb_reg[0])
            WM_cereb_mask_reg_p = WM_cereb_reg[0]  
            
        WM_mask_npy = WM_mask_reg.numpy()
        WM_cereb_npy = WM_cereb_mask_reg.numpy()
        map[WM_mask_npy>0] += 0.8
        map[WM_cereb_npy>0] += 1
        mask_npy = aligned_mask.numpy()
        mask_npy[mask_npy>0] = 1
        map += mask_npy
        count+=1
    
    for dossier in dossiers_lesions:
        lesion_files = [os.path.join(dossier, f) 
                    for f in os.listdir(dossier) 
                    if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier, f))]
        for lesion_file in lesion_files:
        
            lesion_img = nib.load(lesion_file)
            lesion_data = lesion_img.get_fdata()
            lesion_mask = (lesion_data > 0).astype(np.uint8)
            lesion_mask_img = nib.Nifti1Image(lesion_mask, lesion_img.affine)
            # Rééchantillonner le masque pour qu'il corresponde au template
            lesion_mask_resampled_img = resample_to_img(lesion_mask_img, template_nib_T1, interpolation='nearest', force_resample=True, copy_header=True)
            # Récupérer les données rééchantillonnées
            lesion_mask = lesion_mask_resampled_img.get_fdata()
            if(dossier == dossier_lesion_5000_more):
                map += 3*lesion_mask
            else :
                map += lesion_mask
            count+=1
    map = map/count 
    # likelihood_img = nib.Nifti1Image(map, template_nib_T1.affine)
    # nib.save(likelihood_img, "/home/jdrochmans/data/juliette/likelihood_map_WM30.nii")
    likelihood_data_normalized = (map - np.min(map)) / (np.max(map) - np.min(map))
    likelihood_norm = nib.Nifti1Image(likelihood_data_normalized,template_nib_T1.affine)
    nib.save(likelihood_norm, os.path.join(dossier_map_likelihood, f'likelihood_map_norm_WM30.nii'))



