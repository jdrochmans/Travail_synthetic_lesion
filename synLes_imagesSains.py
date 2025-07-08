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

reg_dir = '/home/jdrochmans/data/juliette/transforms_reg/' #folder containing all the transforms matrices
folder_segmentation = "/home/jdrochmans/data/juliette/seg" #folder with the freesurfer segmentation of all images
likelihood_map_path = "/home/jdrochmans/data/juliette/likelihood_map_norm_WM30.nii" #likelihood file
folder_image = "/home/jdrochmans/data/juliette/HC/" #folder with all the healthy subject
folder_WM = '/home/jdrochmans/data/juliette/WM_mask' #folder containing all the WM mask associated to subjects (in MNI and in the patient space)
folder_cortex = '/home/jdrochmans/data/juliette/cortex_mask' #folder containing all the cortex mask associated to subjects (in MNI and in the patient space)
path_dir = "/home/jdrochmans/data/juliette/shape_dir/" #folder containing the shape of lesions (regular)
template_p_T1 = "/home/jdrochmans/data/juliette/template.nii" #template for the MNI registration
folder_mask = "/home/jdrochmans/data/juliette/Dataset001_BrainLesion/labelsTr" #lesion masks from the subjects
folder_registered_mask = "/home/jdrochmans/data/juliette/register_mask" #folder containing all the lesions masks registered in the MNI space 
folder_registered_image = "/home/jdrochmans/data/juliette/register_image" #folder containing all the images registered in the MNI space 
dict_lesions_confluent = "/home/jdrochmans/data/juliette/shape_dir_confluent/" #folder containing the shape of lesions (confluent)
dict_lesions_corticales = "/home/jdrochmans/data/juliette/shape_dir_corticales/" #folder containing the shape of lesions (corticals)
folder_reg = '/home/jdrochmans/data/juliette/reg' #folder where the label masks are registered in the MNI space 
folder_new_label = "/home/jdrochmans/data/juliette/label/" #folder where the label masks in the patient space are found
folder_new_mask = "/home/jdrochmans/data/juliette/mask/" #folder where the masks in the patient space are found (synthetic lesions, mask with only cortical ones or confluent ones)



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
    dict_point_key = {}
    for i,point in enumerate(selected_points):
        dict_point_key[tuple(point)] = i 
        dir_point = os.path.join(path_dir, str(i))
        os.makedirs(dir_point, exist_ok=True)
    return selected_points, dict_point_key


def create_likelihood_ventricles(ventricles_mask, label_MNI, likelihood_map_path):
    
    """
    Generate a normalized likelihood map around ventricles.

    Parameters
    ----------
    ventricles_mask : Binary mask of ventricular voxels in the MNI space.
    label_MNI : Atlas labels in MNI space. Voxels with labels 11 or 50 are excluded from the interest area.
    likelihood_map_path : Path to the original likelihood map.
    output_path : Path where the output NIfTI file will be saved.

    Returns
    -------
    area_interest : Normalized likelihood values within a dilated ring around ventricles,
        masked to exclude non-ventricular labels and outside the extended region.
    
    """
    likelihood_map = nib.load(likelihood_map_path).get_fdata()
    ventricles_mask[ventricles_mask>0]=1
    dilated_ventricle_mask = binary_dilation(ventricles_mask,ball(4))
    dilated_ventricle_mask_limit = binary_dilation(ventricles_mask, ball(5))
    area_interest = dilated_ventricle_mask - ventricles_mask
    area_interest_extended = dilated_ventricle_mask_limit - ventricles_mask
    avoid = np.isin(label_MNI,[11,50]).astype(np.uint8)
    avoid[avoid>0] = 1
    area_interest[avoid>0] = 0
    area_interest = area_interest + likelihood_map
    area_interest = (area_interest - area_interest.min())/(area_interest.max() - area_interest.min())
    area_interest[area_interest_extended==0] = 0
    return area_interest


def create_likelihood_cortex_juxta(cortex_mask):
    """
    Generate a normalized likelihood map for the juxtacortical cortex.

    Parameters
    ----------
    cortex_mask : Binary MNI cortex mask

    Returns
    -------
    juxta_cortex : Likelihood map around the juxtacortical cortex.
    """
    cortex_mask[cortex_mask>0] = 1
    dilated_cortex_mask = binary_dilation(cortex_mask,ball(7))
    juxta_cortex = dilated_cortex_mask-cortex_mask
    return  juxta_cortex.astype(np.float64)

def create_likelihood_cortex_intra(cortex_mask):
    """
    Generate a normalized likelihood map for the intracortical cortex.

    Parameters
    ----------
    cortex_mask : Binary MNI cortex mask

    Returns
    -------
    intra_cortex : Likelihood map around the intracortical cortex.
    
    """
    cortex_mask[cortex_mask>0] =1
    eroded_cortex_mask = binary_erosion(cortex_mask, ball(1))
    eroded_cortex_mask[eroded_cortex_mask>0] = 1
    around_cortex = cortex_mask - eroded_cortex_mask
    around_cortex[around_cortex>0] = 0.5
    intra_cortex = cortex_mask-around_cortex
    return intra_cortex

def calculate_transforms(reg_dir, name) :
    
    """
    Gather forward and inverse registration files for a given subject.

    Parameters
    ----------
    reg_dir : Path to the directory containing registration outputs.
    name : Subject identifier; files must start with "{name}_HC".
    
    
    Returns
    -------
    forward_transforms : Sorted paths to files matching
        "{name}_HC*0GenericAffine.mat" or "{name}_HC*1Warp.nii[.gz]".
    backward_transforms : Sorted paths to files matching
        "{name}_HC*0GenericAffine.mat" or "{name}_HC*1InverseWarp.nii[.gz]".
    """
    #Assume that all transformations are stored in reg_dir already
    forward_transforms = sorted([
    os.path.join(reg_dir, f)
    for f in os.listdir(reg_dir)
    if(f.startswith(f'{name}_HC'))
    if (
        re.search(r"0GenericAffine\.mat$", f) or
        re.search(r"1Warp\.nii(\.gz)?$", f)
    )
    
])
        
    backward_transforms = sorted(
        [os.path.join(reg_dir, f) 
        for f in os.listdir(reg_dir) 
        if(f.startswith(f'{name}_HC'))
        if (
            re.search(r"0GenericAffine\.mat$", f) or
            re.search(r"1InverseWarp\.nii(\.gz)?$", f)
        )]
        ) 
    return forward_transforms, backward_transforms

def calculate_centroid(flat_likelihood,ventricle_mask, cortex_mask, label_MNI,mask_synth_lesions,lesion_mask, bool_cortex, likelihood_map_path):
    
    """
    Select a random voxel weighted by likelihood and check it against anatomical masks to see if its a good candidate.

    Parameters
    ----------
    flat_likelihood : Flattened likelihood map
    ventricle_mask : Binary mask of ventricular voxels
    cortex_mask : Binary mask of cortical voxels.
    label_MNI : Atlas labels in MNI space.
    mask_synth_lesions : Mask of synthetic lesions.
    lesion_mask : Mask of real lesions.
    bool_cortex : bool
        If True, point must be in the cortex area.
        If False, point cannot be in the cortex area.

    Returns
    -------
    new_centroid : The (x, y, z) coordinates of the sampled voxel.
    good_cand : bool
        True if the sampled voxel meets the mask criteria, False otherwise.
    """
    
    likelihood_map = nib.load(likelihood_map_path).get_fdata()
    good_cand = False
    valid_indices = np.where(flat_likelihood > 0)[0]
    valid_probs = flat_likelihood[valid_indices]
    valid_probs /= valid_probs.sum()
    voxel_index = np.random.choice(valid_indices, p=valid_probs)
    new_centroid = np.unravel_index(voxel_index, likelihood_map.shape)
    candidate = new_centroid
    if(bool_cortex == True):
         if(label_MNI[candidate[0], candidate[1], candidate[2]]!= 0 and mask_synth_lesions[candidate[0], candidate[1], candidate[2]] == 0 and  lesion_mask[candidate[0], candidate[1], candidate[2]]==0) :
            good_cand = True
            new_centroid = candidate
    
    else : 
        if(ventricle_mask[candidate[0], candidate[1], candidate[2]] == 0 and cortex_mask[candidate[0], candidate[1], candidate[2]] == 0 and label_MNI[candidate[0], candidate[1], candidate[2]]!= 0 and mask_synth_lesions[candidate[0], candidate[1], candidate[2]] == 0 and lesion_mask[candidate[0], candidate[1], candidate[2]]==0) :
            good_cand = True
            new_centroid = candidate
    return new_centroid, good_cand


def open_folder_lesion(folder_lesion,num,folder_lesion_list, bool_intracortical, bool_juxtacortical):
    
    """
    Load a random lesion volume file from a numbered subfolder, filtered by lesion type (for cortical lesions, either intracortical or juxtacortical),
    retrying until a volume is found.

    Parameters
    ----------
    folder_lesion : Base directory containing numbered lesion subfolders.
    num : Name of the subfolder to search in. 
    folder_lesion_list : List of filenames to choose from.
    bool_intracortical : bool
        If True, only files whose name contains "intracort" are considered.
    bool_juxtacortical : bool
        If True only files whose name contains "juxtacort" are considered.

    Returns
    -------
    volume : The image data array from the randomly selected NIfTI file, guaranteed non-empty.
    random_file_path : Full path to the file that was loaded.
    """    

    volume_random_data_shape = None
    while(np.prod(volume_random_data_shape) == 0 or volume_random_data_shape == None):
        if(bool_intracortical):
            folder_lesion_intra = os.path.join(folder_lesion,str(num))
            intracort_files = [f for f in os.listdir(folder_lesion_intra) if 'intracort' in f.lower()]
            if intracort_files == []:
                raise RuntimeError(f"No intracortical files found in {os.path.join(folder_lesion,str(num))}")
            random_file = random.choice(intracort_files)
            random_file_path = os.path.join(os.path.join(folder_lesion,str(num)), random_file) 
            volume = (nib.load(random_file_path)).get_fdata()
            volume_random_data_shape = volume.shape 
        elif(bool_juxtacortical): 
            folder_lesion_juxta = os.path.join(folder_lesion,str(num))
            juxta_files = [f for f in os.listdir(folder_lesion_juxta) if 'juxtacort' in f.lower()]
            random_file = random.choice(juxta_files)
            random_file_path = os.path.join(os.path.join(folder_lesion,str(num)), random_file) 
            volume = (nib.load(random_file_path)).get_fdata()
            volume_random_data_shape = volume.shape 
        else:
            random_file = random.choice(folder_lesion_list)
            random_file_path = os.path.join(os.path.join(folder_lesion,str(num)), random_file) 
            volume = (nib.load(random_file_path)).get_fdata()
            volume_random_data_shape = volume.shape 
            
    return volume, random_file_path
    
    
def label_map_synLes(folder_new_label, folder_new_mask, label_map, points,dict_point_key, facteur_confluence, template_p_T1, nb_lesions,likelihood_map_path, nb_les_ventricles, nb_les_intra, nb_les_juxta, nb_les_conf, nb_loop,label = 86):
    """
    For each subject mask in in the list of paths `label_map`, add synthetic lesions (i.e. cortical, confluent, ventricular...) in anatomically relevant locations. Save also multiple output masks representing cortical lesions, confluents lesions and all the synthetic lesions added 
    to the image.
    Parameters
    ----------
    label_map : Paths to subject-specific label NIfTI files to process.
    points : Voxel coordinates  associated to the dictionary(from create_points).
    dict_point_key : DIctionary of shape, associated to points to find suited area.
    facteur_confluence : Maximum allowed lesion overlap ratio when placing synthetic lesions.
    template_p_T1 : Path to the T1-weighted template NIfTI used as registration target.
    nb_lesions : Total number of lesions to generate per subject. 
    likelihood_map_path : Path to the original likelihood map NIfTI.
    nb_les_ventricles : Number of lesions to force into periventricular regions.
    nb_les_intra : Number of intracortical lesions to generate.
    nb_les_juxta : Number of juxtacortical lesions to generate.
    nb_les_conf : Number of confluent lesions to generate.
    label : Starting label value for synthetic lesions (default=86).

    nb_lesions MUST BE >= nb_les_ventricles + nb_les_intra + nb_les_juxta + nb_les_conf
    
    Returns
    -------
    all_output_new_mask : List of path to final synthetic lesion label maps in original subjects spaces.
    names : List of subject identifier extracted from each input.
    """

    all_output_path = []
    names = []
    for map in label_map:
        label = 86
        print(map)
        path_in_name = os.path.basename(map)
        name = path_in_name.split("_")[0]
        template_nib_T1 = nib.load(template_p_T1)
        points, dict_point_key = create_points(likelihood_map_path,path_dir,20)
        forward_transforms, backward_transforms = calculate_transforms(reg_dir,name)
        image_p = map
        output =  os.path.join(folder_registered_mask, f'label_HC_{name}_reg.nii.gz')
        cmd = f"antsApplyTransforms -i {image_p} -r {template_p_T1} -n {'genericLabel'} -t {forward_transforms[1]} -t {forward_transforms[0]} -o {output}"
        subprocess.Popen(cmd, shell = True).wait()

        label_MNI = nib.load(output)
        dim_x,dim_y,dim_z = label_MNI.shape
        label_sem = label
        label_MNI = label_MNI.get_fdata()
        label_MNI_lesion_sem = label_MNI.copy()
        likelihood_map = nib.load(likelihood_map_path).get_fdata()
        flat_likelihood = likelihood_map.flatten()
        ventricle_mask = np.isin(label_MNI, [4,14,15,43,72,49,10]).astype(np.uint32)
        avoid_CC = np.isin(label_MNI, [251,252,253,254,255]).astype(np.uint32)
        lesion_mask = np.isin(label_MNI, 77).astype(np.uint32)
        norm_map = create_likelihood_ventricles(ventricle_mask,label_MNI, likelihood_map_path)
        norm_map_flatten = norm_map.flatten()
        
        mask_synth_lesions = np.zeros((label_MNI.shape), dtype = np.float32)
        mask_synth_lesions_binary = np.zeros((label_MNI.shape), dtype = np.float32)
        mask_lesion_conf = np.zeros((label_MNI.shape), dtype = np.float32)
        mask_lesion_intracorticales = np.zeros((label_MNI.shape), dtype = np.float32)
        mask_lesion_juxtacorticales = np.zeros((label_MNI.shape), dtype = np.float32)
        cortex_reg = [os.path.join(folder_cortex, f) 
            for f in os.listdir(folder_cortex) 
            if f.startswith(f"mask_cortex_registered_{name}_HC")]
        cortex_mask = nib.load(cortex_reg[0]).get_fdata()
        cortex_map_juxta = create_likelihood_cortex_juxta(cortex_mask) 
        flat_likelihood_cortex_juxta = cortex_map_juxta.flatten()
        cortex_map_intra = create_likelihood_cortex_intra(cortex_mask) 
        flat_likelihood_cortex_intra = cortex_map_intra.flatten()
        for i in range(nb_lesions):
            cortex_les = False
            conf = False
            
            max_attempt = 20
            attempt = 0
            bool_dict = False
            new_mask_lesion = np.zeros((label_MNI.shape), dtype = np.float32)
            #loop to find a right position and a right shape 
            while(bool_dict == False and attempt < max_attempt):
                max_iter = 10
                iteration = 0
                folder_assoc_num = []
                bool_intra = False
                bool_juxta = False
                #loop to find a good candidate as centroid
                while ((folder_assoc_num == [] or folder_assoc_num_conf == []) and iteration < max_iter):
                    
                    good_cand = False
                    while good_cand == False :
                        if(nb_lesions < nb_les_ventricles or (i > nb_les_ventricles+ nb_les_intra+ nb_les_juxta)):
                            new_centroid, good_cand = calculate_centroid(flat_likelihood,ventricle_mask,cortex_mask,label_MNI,mask_synth_lesions,lesion_mask, False,likelihood_map_path)
                        else :
                            if(i < nb_les_ventricles):
                                new_centroid, good_cand = calculate_centroid(norm_map_flatten,ventricle_mask,cortex_mask,label_MNI,mask_synth_lesions,lesion_mask, False,likelihood_map_path)
                            
                            if(nb_les_ventricles<= i <= nb_les_ventricles + nb_les_intra+nb_les_juxta) :
                                
                                if(nb_les_ventricles + nb_les_juxta <= i):
                                    bool_intra = True
                                    bool_juxta = False
                                    new_centroid, good_cand = calculate_centroid(flat_likelihood_cortex_intra,ventricle_mask,cortex_mask,label_MNI,mask_synth_lesions,lesion_mask, True,likelihood_map_path)
                                else :
                                    new_centroid, good_cand = calculate_centroid(flat_likelihood_cortex_juxta,ventricle_mask,cortex_mask,label_MNI,mask_synth_lesions,lesion_mask, True,likelihood_map_path)
                                    bool_intra = False
                                    bool_juxta = True
                                cortex_les = True
                                
                                
                    point_chosen = 0
                    cnt = 1000
                    for point in points :
                        if  np.linalg.norm(new_centroid - point) < cnt : 
                            point_chosen = point
                            cnt = np.linalg.norm(new_centroid - point)
                    
                    num = dict_point_key[tuple(point_chosen)]

                    folder_assoc_num =  os.listdir(os.path.join(path_dir,str(num)) ) 
                    folder_assoc_num_conf = os.listdir(os.path.join(dict_lesions_confluent,str(num)))
                    folder_assoc_num_cortex = os.listdir(os.path.join(dict_lesions_corticales,str(num)))
                    if(bool_intra):
                        intracortical_files = [f for f in folder_assoc_num_cortex if 'intracort' in f.lower()]
                        var = 1
                        iter = 0
                        while(intracortical_files == [] and iter < 100):
                            num = (num + var)%16
                            folder_assoc_num_cortex = os.listdir(os.path.join(dict_lesions_corticales,str((num))))
                            intracortical_files = [f for f in folder_assoc_num_cortex if 'intracort' in f.lower()]
                            iter +=1
                    iteration +=1
                    
                        
                if((folder_assoc_num == [] or folder_assoc_num_cortex == [] or folder_assoc_num_conf == []) and iteration >= max_iter ):
                    print('Couldnt find a file in a reasonable number of iterations')

                else :
                    
                    if(nb_les_ventricles + nb_les_intra + nb_les_juxta < i <= nb_les_ventricles+ nb_les_intra+ nb_les_juxta + nb_les_conf):
                        
                        volume, random_file_path = open_folder_lesion(dict_lesions_confluent,num, folder_assoc_num_conf, bool_intra, False)   
                        conf = True
                    if(cortex_les == True):
                        volume, random_file_path = open_folder_lesion(dict_lesions_corticales,num, folder_assoc_num_cortex, bool_intra,bool_juxta) 
                        
                        type_les = random_file_path.split('/')[7].split('_')[1]

                    else:
                        volume, random_file_path = open_folder_lesion(path_dir,num, folder_assoc_num, bool_intra, False) 
                        if(i > nb_les_ventricles+ nb_les_intra+ nb_les_juxta + nb_les_conf):
                            mult = np.random.normal(loc=1.2, scale=0.5)
                            mult = np.clip(mult, 1.0, 1.5)   
                        else:
                            mult = np.random.normal(loc=1.0, scale=0.3)
                            mult = np.clip(mult, 0.2, 1.1)
                        if(conf==True):
                            volume = volume
                            
                        else:
                            volume =  zoom(volume, zoom=mult, order=1) 
                            volume = binary_dilation(volume)
                            volume = binary_erosion(volume)
                        
                        
                    volume_random_data_shape = volume.shape
                    start_x = int(new_centroid[0] - volume_random_data_shape[0]/2)
                    start_y = int(new_centroid[1] - volume_random_data_shape[1]/2)
                    start_z = int(new_centroid[2] - volume_random_data_shape[2]/2)
                    
                    stop_x = int(new_centroid[0] + volume_random_data_shape[0]/2)
                    stop_y = int(new_centroid[1] + volume_random_data_shape[1]/2)
                    stop_z = int(new_centroid[2] + volume_random_data_shape[2]/2)
                    
                    start_x = max(0,min(dim_x,start_x))
                    start_y = max(0,min(start_y,dim_y))
                    start_z = max(0,min(start_z,dim_z))
                    
                    stop_x = max(0,min(stop_x,dim_x))
                    stop_y = max(0,min(stop_y,dim_y))
                    stop_z = max(0,min(stop_z,dim_z)) 

                    tol_ratio_2 = np.sum(mask_synth_lesions[start_x:stop_x, start_y:stop_y, start_z:stop_z] > 0) / ((stop_x-start_x) * (stop_y-start_y) * (stop_z-start_z))
                    if(start_x<stop_x and start_y < stop_y and start_z < stop_z  and tol_ratio_2 < facteur_confluence): 
                        bool_dict = True  
                    else :
                        attempt +=1
            if(attempt == max_attempt):
                print(f"No valid position found after {max_attempt} trials, change of lesion's form.")
                
                continue 
            else :
                lesion_patch = volume
                
                #Paste the lesion on a new mask and ensure that the lesion is not placed on avoiding areas, following also the type of lesions (conf, ventricular, cortical...)
                new_mask_lesion[start_x:stop_x,start_y:stop_y,start_z:stop_z] = lesion_patch
                if(cortex_les== True):
                    new_mask_lesion[ventricle_mask>0] = 0
                    new_mask_lesion[label_MNI==0] = 0
                    new_mask_lesion[lesion_mask>0] = 0
                    new_mask_lesion[mask_synth_lesions>0] = 0
                    new_mask_lesion[avoid_CC>0] = 0
                    
                    if(type_les == 'juxtacort'):
                        new_mask_lesion[cortex_mask>0] = 0
                        mask_lesion_juxtacorticales[new_mask_lesion>0] = 1
                    else :
                        inv_cortex_mask = 1 - cortex_mask
                        new_mask_lesion[inv_cortex_mask>0] = 0
                        mask_lesion_intracorticales[new_mask_lesion>0] = 1
                    while(np.isin(label_MNI, label).astype(np.uint32).any()):
                        label +=1
                    
                    label_MNI[new_mask_lesion>0] = np.uint32(label)
                    mask_synth_lesions[new_mask_lesion>0] = np.uint32(label)
                    label_MNI_lesion_sem[new_mask_lesion>0] = np.uint32(label_sem)
                    mask_synth_lesions_binary[new_mask_lesion>0] = np.uint32(label_sem)
                        
                        
                
                else :
                    new_mask_lesion[ventricle_mask>0] = 0
                    new_mask_lesion[cortex_mask>0] = 0
                    new_mask_lesion[label_MNI==0] = 0
                    new_mask_lesion[lesion_mask>0] = 0
                    new_mask_lesion[mask_synth_lesions>0] = 0
                    new_mask_lesion[avoid_CC>0] = 0
                    if(conf == False):
                        while(np.isin(label_MNI, label).astype(np.uint32).any()):
                            label +=1
                        
                        label_MNI[new_mask_lesion>0] = np.uint32(label)
                        mask_synth_lesions[new_mask_lesion>0] = np.uint32(label)
                        label_MNI_lesion_sem[new_mask_lesion>0] = np.uint32(label_sem)
                        mask_synth_lesions_binary[new_mask_lesion>0] = np.uint32(label_sem)
                    else : 
                        labels_conf = np.unique(new_mask_lesion).astype(np.uint32)
                        for lab in labels_conf:
                            
                            if lab == 0:
                                continue
                            else :
                                while(np.isin(label_MNI, label).astype(np.uint32).any()):
                                    label +=1
                                label_MNI[new_mask_lesion==lab] = np.uint32(label)
                                mask_synth_lesions[new_mask_lesion==lab] = np.uint32(label)
                                label_MNI_lesion_sem[new_mask_lesion==lab] = np.uint32(label)
                                mask_synth_lesions_binary[new_mask_lesion==lab] = np.uint32(label)
                                mask_lesion_conf[new_mask_lesion==lab] = np.uint32(label)
                                new_mask_lesion[new_mask_lesion==lab] = np.uint32(label)
        #Register all the masks (label mask with all the brain structure, masks to highlight confluent lesions and cortical lesions, and instantial mask)               
        lesion_mask_nib = nib.Nifti1Image(mask_synth_lesions,template_nib_T1.affine)
        label_MNI_nib =  nib.Nifti1Image(label_MNI,template_nib_T1.affine)
        label_MNI_semantic_nib = nib.Nifti1Image(label_MNI_lesion_sem,template_nib_T1.affine)
        mask_lesion_conf_nib = nib.Nifti1Image(mask_lesion_conf,template_nib_T1.affine)
        mask_lesion_intracorticales = nib.Nifti1Image(mask_lesion_intracorticales, template_nib_T1.affine)
        mask_lesion_juxtacorticales = nib.Nifti1Image(mask_lesion_juxtacorticales,template_nib_T1.affine)
        nib.save(lesion_mask_nib,os.path.join(folder_reg, f'mask_synLes_HC_inst_{name}.nii.gz'))
        nib.save(label_MNI_nib,os.path.join(folder_new_label, f'label_MNI_HC_inst_{name}.nii.gz'))
        nib.save(label_MNI_semantic_nib, os.path.join(folder_reg, f'label_MNI_sem_{name}.nii.gz'))
        nib.save(mask_lesion_conf_nib,os.path.join(folder_new_label,f'mask_les_conf_MNI_{name}.nii.gz'))
        nib.save(mask_lesion_intracorticales, os.path.join(folder_new_label, f'mask_lesion_intracorticales_MNI_{name}.nii.gz'))
        nib.save(mask_lesion_juxtacorticales, os.path.join(folder_new_label, f'mask_lesion_juxtacorticales_MNI_{name}.nii.gz'))

        lesion_mask_p = os.path.join(folder_reg, f'mask_synLes_HC_inst_{name}.nii.gz')
        label_MNI_p = os.path.join(folder_new_label, f'label_MNI_HC_inst_{name}.nii.gz')
        mask_les_conf_p = os.path.join(folder_new_label,f'mask_les_conf_MNI_{name}.nii.gz')
        mask_les_intra_p = os.path.join(folder_new_label, f'mask_lesion_intracorticales_MNI_{name}.nii.gz')
        mask_les_juxta_p = os.path.join(folder_new_label, f'mask_lesion_juxtacorticales_MNI_{name}.nii.gz')

        output_Newmask1 = os.path.join(folder_new_label, f'label_lesion_HC_inst_{name}_{nb_loop}.nii.gz')
        cmd = f"antsApplyTransforms -d 3 -i {label_MNI_p} -r {map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask1}"
        subprocess.Popen(cmd, shell = True).wait()
 
        output_Newmask2 = os.path.join(folder_new_mask, f'mask_lesion_HC_inst_{name}_{nb_loop}.nii.gz')
        cmd = f"antsApplyTransforms -d 3 -i {lesion_mask_p} -r {map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask2}"
        subprocess.Popen(cmd, shell = True).wait()
        
        
        output_Newmask4 = os.path.join(folder_new_mask, f'mask_lesion_conf_{name}_{nb_loop}.nii.gz')
        cmd = f"antsApplyTransforms -d 3 -i {mask_les_conf_p} -r {map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask4}"
        subprocess.Popen(cmd, shell = True).wait()
        
        
        output_Newmask5 = os.path.join(folder_new_mask, f'mask_lesion_intracorticales_{name}_{nb_loop}.nii.gz')
        cmd = f"antsApplyTransforms -d 3 -i {mask_les_intra_p} -r {map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask5}"
        subprocess.Popen(cmd, shell = True).wait()
        
        output_Newmask6 = os.path.join(folder_new_mask, f'mask_lesion_juxtacorticales_{name}_{nb_loop}.nii.gz')
        cmd = f"antsApplyTransforms -d 3 -i {mask_les_juxta_p} -r {map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask6}"
        subprocess.Popen(cmd, shell = True).wait()
        all_output_path.append(output_Newmask1)
        names.append(name)
    
    return all_output_path, names


if __name__ == "__main__":
    print("Beginning")
    facteur_confluence = 0.05
    
    points, dict_point_key = create_points(likelihood_map_path,path_dir,20) #send back always the same list of centroids associated folder n°, containing lesions shape
    label_map = []
    for root, dirs, files in os.walk(folder_image):
        for f in files:
            if f.endswith(('.nii', '.nii.gz')) and 'mask-FSaseg_T2' in f:
                label_map.append(os.path.join(root, f))
    #Create 3 images different for each healthy patient
    for i in range(3):
        all_output_path, names = label_map_synLes(folder_new_label, folder_new_mask,label_map, points, dict_point_key, facteur_confluence, template_p_T1, 300, likelihood_map_path, 100,100, 20, 50, i, label=86)
    all_output_path =  sorted(all_output_path, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    names = sorted(names, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    #Create a semantic mask with the instantial for each 
    for i in range(len(all_output_path)):
        img_inst_nib = nib.load(all_output_path[i])
        filename = os.path.basename(all_output_path[i])
        base, ext = os.path.splitext(filename)
        nb_loop = base.split('_')[5]
        img_inst = img_inst_nib.get_fdata()
        
        img_inst[(img_inst >= 86) & (img_inst < 251)] = 86
        img_inst[img_inst>255] = 86
        
        mask_sem = nib.Nifti1Image(img_inst, img_inst_nib.affine)
        nib.save(mask_sem,os.path.join(folder_new_label,f'label_lesion_sem2_HC_{names[i]}_{nb_loop}.nii.gz'))
    print("End")
    
    
    
    