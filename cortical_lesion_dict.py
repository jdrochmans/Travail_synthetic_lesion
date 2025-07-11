#Ajoute des lésions sur un masque de segmentation synthMS 
import os
import re
import subprocess
import ants
import numpy as np
import nibabel as nib
import random
from skimage.measure import label, regionprops
from skimage.morphology import ball, binary_dilation, binary_erosion
from scipy.ndimage import zoom
from avoiding_map import otsu

#synLes_imagesSains.py to understand all the folders cited here
folder_seg = "seg"
reg_dir = 'transforms_reg/'
likelihood_map_path = "likelihood_map_norm_WM30.nii"
path_dir = "shape_dir_corticales/"
template_p_T1 = "template.nii"
folder_mask = "from our private DS : labelsTr"
folder_registered_mask = "register_mask"
#folder_registered_image = "/home/jdrochmans/data/juliette/register_image"
folder_cortex = 'cortex_mask'


def create_points(likelihood_map_path,path_dir, min_distance=20):
    """
    Select points from a 3D likelihood map and make a folder for each.

    Parameters
    ----------
    likelihood_map_path : path to the likelihood map
    path_dir : path to the directory where the folders are stored
    min_distance : float, optional
        Minimum Euclidean distance (in voxels) between any two selected points (default=20).
    Returns
    -------
    selected_points : Coordonate of each folder from the dictionary
    dict_point_keys : Dictionary associating each selected points to the name of the folder in the dictionary
    """
    likelihood_map = nib.load(likelihood_map_path).get_fdata()
    points = np.argwhere((likelihood_map > 0.55))
    selected_points = []

    for point in points:
        if len(selected_points) == 0:
            selected_points.append(point)
        else:
            if all(np.linalg.norm(point - existing_point) > min_distance for existing_point in selected_points):
                selected_points.append(point)
    selected_points = np.array(selected_points)
    

    dict_point_clés = {}
    
    for i,point in enumerate(selected_points):
        dict_point_clés[tuple(point)] = i 
        dir_point = os.path.join(path_dir, str(i))
        os.makedirs(dir_point, exist_ok=True)
    return selected_points, dict_point_clés


def shape_dir(likelihood_map_path,path_dir, folder_seg,reg_dir, template_p_T1,folder_registered_mask) :
    
    """
    Extract and save in subdirectories individual lesion masks from cortical segmentations, registered in the MNI space.

    Parameters
    ----------
    likelihood_map_path : Path to the 3D likelihood file.
    path_dir : Base directory where subfolders for each index (from create_points)
        already exist.
    folder_seg : Directory containing cortical segmentation NIfTI files named “<subject>_cort*.nii”.
    reg_dir : Directory holding ANTs forward‐transform files (“<subject>_0GenericAffine.mat”,
        “<subject>_1Warp.nii[.gz]”) for each subject.
    template_p_T1 : Path to the T1‐weighted template NIfTI used as registration reference.
    folder_registered_mask : Directory where registered cortical masks are stored.

    Returns
    -------
    points : Coordinates associated to the keys of the dictionary.
    dict_point_clés : Dictionary associating points to the directory (0,1,..N) containing the lesions from an area.
    """
    points, dict_point_clés = create_points(likelihood_map_path,path_dir)
    template_nib = nib.load(template_p_T1)
    fichiers_cort = [f for f in os.listdir(folder_seg) if "_cort" in f and f.endswith(".nii")]
    mask_files_sorted = sorted(fichiers_cort, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    count_mask = 0
    for i in range(len(mask_files_sorted)):
        #registration of the lesion mask
        path_in_name = os.path.basename(mask_files_sorted[i])
        name = path_in_name.split("_")[0]
        forward_transforms = []
        forward_transforms= sorted(
        [os.path.join(reg_dir, f) 
        for f in os.listdir(reg_dir) 
        if(f.startswith(f'{name}'))
        if (
            re.search(r"_0GenericAffine\.mat$", f) or
            re.search(r"_1Warp\.nii(\.gz)?$", f)
        )]
        )
        mask_reg = [os.path.join(folder_registered_mask, f) 
        for f in os.listdir(folder_registered_mask) 
        if f.startswith(f"{name}_cortical")]
        if(mask_reg == []):
            output =  os.path.join(folder_registered_mask, f'{name}_cortical.nii.gz')
            cmd = f"antsApplyTransforms -i {os.path.join(folder_seg,mask_files_sorted[i])} -r {template_p_T1} -n {'genericLabel'} -t {forward_transforms[1]} -t {forward_transforms[0]} -o {output}"
            subprocess.Popen(cmd, shell = True).wait()
            aligned_mask = ants.image_read(output)
            mask_p = output
       
        else : 
            aligned_mask = ants.image_read(mask_reg[0])
            mask_p = mask_reg[0]
        mask = aligned_mask.numpy()
        if(np.sum(mask) == 0):
            print('Empty mask!')
        labels = np.unique(mask)
        #Use of the cortex mask to see if the lesion is intracortical or juxtacortical
        cortex_reg = [os.path.join(folder_cortex, f) 
        for f in os.listdir(folder_cortex) 
        if f.startswith(f"mask_cortex_registered_{name}")]
        count_lesion = 0
        for lab in labels:
            cortex = nib.load(cortex_reg[0]).get_fdata()
            
            mask_lesion_i = np.array(mask == lab).astype(np.uint8)
            prop = regionprops(mask_lesion_i)
            centroid_region = prop[0]['centroid']
        
            indices = np.argwhere(mask_lesion_i>0)
            indice2 = np.max(indices,axis = 0)
        
            x_max = indice2[0] +1
            y_max = indice2[1] +1
            z_max = indice2[2] +1
            
            indice3 = np.min(indices,axis = 0)
            x_min = indice3[0]
            y_min = indice3[1]
            z_min = indice3[2] 

            mask_lesion_i = mask_lesion_i[x_min:x_max, y_min:y_max, z_min: z_max].astype(np.uint8)
            cnt = 1000
            total = np.sum(mask_lesion_i)
            cortex_crop = cortex[x_min:x_max, y_min:y_max, z_min:z_max]
            mask_local = mask_lesion_i > 0
            cortex_total = np.sum(cortex_crop[mask_local])
            intracort = False
            if(total == cortex_total):
                intracort = True
            for point in points :
                #find where to place the shape following the centroid associated to each folder
                if  np.linalg.norm(centroid_region - point) < cnt :
                    point_mei = point 
                    cnt = np.linalg.norm(centroid_region - point)
            
            num = dict_point_clés[tuple(point_mei)]
            point_dir = os.path.join(path_dir, str(num)) 
            if(intracort == False):
                
                file_path = os.path.join(point_dir, f"lesion_juxtacort_{num + count_lesion}_mask_{count_mask}.nii.gz") 
            else : 
                file_path = os.path.join(point_dir, f"lesion_intracort_{num + count_lesion}_mask_{count_mask}.nii.gz") 
            nifti_lesion = nib.Nifti1Image(mask_lesion_i, template_nib.affine)
            nib.save(nifti_lesion,file_path)
            count_lesion +=1
        count_mask+=1
    return points, dict_point_clés

if __name__ == "__main__":
    print("Beginning")
    points, dict_point_clés = create_points(likelihood_map_path,path_dir,20)
    points, dict_point_clés = shape_dir(likelihood_map_path,path_dir,folder_seg,reg_dir,template_p_T1)
    print("End")