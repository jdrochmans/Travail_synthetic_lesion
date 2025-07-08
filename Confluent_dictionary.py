#creation d'un dictionnaire de lésions confluentes 
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
#synLes_imagesSains.py to understand all the folders cited here
reg_dir = 'transforms_reg/'
likelihood_map_path = "likelihood_map_norm_WM30.nii"
template_p_T1 = "template.nii"
path_dir = "shape_dir_confluent/"
folder_mask = "from our DS : labelsTr"
folder_registered_mask = "register_mask"
#folder_registered_image = "/home/jdrochmans/data/juliette/register_image"


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


def shape_dir(likelihood_map_path,path_dir, folder_mask,reg_dir, template_p_T1, folder_registered_mask) :
        
    """
    Extract and save in subdirectories confluents lesion from our private DS, registered in the MNI space.

    Parameters
    ----------
    likelihood_map_path : Path to the 3D likelihood file.
    path_dir : Base directory where subfolders for each index (from create_points)
        already exist.
    folder_mask : Directory containing lesion segmentation NIfTI files.
    reg_dir : Directory holding ANTs forward‐transform files (“<subject>_0GenericAffine.mat”,
        “<subject>_1Warp.nii[.gz]”) for each subject.
    template_p_T1 : Path to the T1‐weighted template NIfTI used as registration reference.
    folder_registered_mask : Directory where registered lesion masks are stored.

    Returns
    -------
    points : Coordinates associated to the keys of the dictionary.
    dict_point_clés : Dictionary associating points to the directory (0,1,..N) containing the lesions from an area.
    """
    
    points, dict_point_clés = create_points(likelihood_map_path,path_dir)
    template_nib = nib.load(template_p_T1)
    
    mask_files = [os.path.join(folder_mask, f) for f in os.listdir(folder_mask) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(folder_mask, f)) and 'mask-instances' in f]
    mask_files_sorted = sorted(mask_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    count_mask = 0
    for i in range(len(mask_files_sorted)):
        
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
        if f.startswith(f"{name}")]
        if(mask_reg == []):
            output =  os.path.join(folder_registered_mask, f'{name}.nii.gz')
            cmd = f"antsApplyTransforms -i {mask_files[i]} -r {template_p_T1} -n {'genericLabel'} -t {forward_transforms[1]} -t {forward_transforms[0]} -o {output}"
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
        regions = regionprops(mask.astype(np.int32))
        count_lesion = 0
        #Loop on area, if contain more than one lesion => the area is added on the dictionary with the label mask of the two lesions
        for region in regions :
            count = 0
            minx, miny, minz = region.bbox[0], region.bbox[1], region.bbox[2]
            maxx,maxy,maxz = region.bbox[3], region.bbox[4], region.bbox[5]
            mask_region = mask[minx:maxx,miny:maxy,minz:maxz]
            labels_in_region = np.unique(mask_region[mask_region > 0])
            
            if(len(labels_in_region)>1) :
                centroid_region = region.centroid
                mask_lesion_i = mask_region.astype(np.uint32)
                cnt = 1000
                
                for point in points :
                    #adding the lesion to the closest region, found via comparison with the list of centroid 'points'.
                    if  np.linalg.norm(centroid_region - point) < cnt :
                        point_mei = point 
                        cnt = np.linalg.norm(centroid_region - point)
                
                num = dict_point_clés[tuple(point_mei)]
                point_dir = os.path.join(path_dir, str(num)) 
                file_path = os.path.join(point_dir, f"lesion_{num + count_lesion}_mask_{count_mask}.nii.gz") 
                nifti_lesion = nib.Nifti1Image(mask_lesion_i, template_nib.affine)
                nib.save(nifti_lesion,file_path)
            count_lesion +=1    
        count_mask+=1
    return points, dict_point_clés


if __name__ == "__main__":
    print("Beginning")
    points, dict_point_clés = create_points(likelihood_map_path,path_dir,20)
    points, dict_point_clés = shape_dir(likelihood_map_path,path_dir,folder_mask,reg_dir,template_p_T1)
    print("End")