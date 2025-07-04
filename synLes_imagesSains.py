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
reg_dir = '/home/jdrochmans/data/juliette/transforms_reg/'
dossier_segmentation = "/home/jdrochmans/data/juliette/seg"
likelihood_map_path = "/home/jdrochmans/data/juliette/likelihood_map_norm_WM30.nii"
likelihood_map = nib.load(likelihood_map_path).get_fdata()
template_p = "/home/jdrochmans/data/juliette/template.nii"

template_T1 = "/home/jdrochmans/data/juliette/template.nii"

big_les_texture = np.load('/home/jdrochmans/data/juliette/big_les.npz', mmap_mode='r')

small_les_texture = np.load('/home/jdrochmans/data/juliette/small_les.npz',mmap_mode='r')
dossier_image = "/home/jdrochmans/data/juliette/HC/sub-162"
dossier_WM = '/home/jdrochmans/data/juliette/WM_mask'
dossier_cortex = '/home/jdrochmans/data/juliette/cortex_mask'

likelihood_map_path = "/home/jdrochmans/data/juliette/likelihood_map_norm_WM30.nii"
likelihood_map = nib.load(likelihood_map_path).get_fdata()
path_dir = os.path.join("/home/jdrochmans/data/juliette/shape_dir/")

template_p_T1 = "/home/jdrochmans/data/juliette/template.nii"
dossier_mask = "/home/jdrochmans/data/juliette/Dataset001_BrainLesion/labelsTr"
dossier_registered_mask = "/home/jdrochmans/data/juliette/register_mask"
dossier_registered_image = "/home/jdrochmans/data/juliette/register_image"
#dossier_image = "/home/jdrochmans/data/juliette/Dataset001_BrainLesion/imagesTr"
dossier_cortex = '/home/jdrochmans/data/juliette/cortex_mask'
dict_lesions_confluent = "/home/jdrochmans/data/juliette/shape_dir_confluent/"
dict_lesions_corticales = "/home/jdrochmans/data/juliette/shape_dir_corticales/"
likelihood_map = nib.load(likelihood_map_path).get_fdata()
#template_T1 = ants.image_read(template_p_T1)
template_nib_T1 = nib.load(template_p_T1)


label_map = [os.path.join(dossier_image, f) for f in os.listdir(dossier_image) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_image, f)) and f.startswith('sub-162_ses-01_mask-FSaseg_T2starw')]
print(label_map)
def create_points(likelihood_map,path_dir, min_distance=20):
    points = np.argwhere((likelihood_map > 0.55))
    selected_points = []

    for point in points:
        if len(selected_points) == 0:
            selected_points.append(point)
        else:
            if all(np.linalg.norm(point - existing_point) > min_distance for existing_point in selected_points):
                selected_points.append(point)

            # if np.all(np.isinf(distances)): 
            #     selected_points.append(point)

    selected_points = np.array(selected_points)
    

    dict_point_clés = {}
    
    for i,point in enumerate(selected_points):
        print(point)
        dict_point_clés[tuple(point)] = i 
        dir_point = os.path.join(path_dir, str(i))
        os.makedirs(dir_point, exist_ok=True)
    return selected_points, dict_point_clés


def shape_dir(likelihood_map,path_dir, dossier_mask,reg_dir, template_p_T1) :
    
    points, dict_point_clés = create_points(likelihood_map,path_dir)
    template = ants.image_read(template_p_T1)
    template_nib = nib.load(template_p_T1)
     #utilise le mask recaled
    mask_files = [os.path.join(dossier_mask, f) for f in os.listdir(dossier_mask) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_mask, f)) and 'mask-instances' in f]
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
        mask_reg = [os.path.join(dossier_registered_mask, f) 
        for f in os.listdir(dossier_registered_mask) 
        if f.startswith(f"{name}")]
        print(mask_reg)
        if(mask_reg == []):
            output =  os.path.join(dossier_registered_mask, f'{name}.nii.gz')
            cmd = f"antsApplyTransforms -i {mask_files[i]} -r {template_p_T1} -n {'genericLabel'} -t {forward_transforms[1]} -t {forward_transforms[0]} -o {output}"
            subprocess.Popen(cmd, shell = True).wait()
            aligned_mask = ants.image_read(output)
            mask_p = output
       
        else : 
            aligned_mask = ants.image_read(mask_reg[0])
            mask_p = mask_reg[0]
        mask = aligned_mask.numpy()
        if(np.sum(mask) == 0):
            print('probleme, masque vide!')
        labels = np.unique(mask)
        print(labels)
        
        count_lesion = 0
        for lab in labels:
            
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
            for point in points :
                
                if  np.linalg.norm(centroid_region - point) < cnt :
                    point_mei = point 
                    cnt = np.linalg.norm(centroid_region - point)
            
            num = dict_point_clés[tuple(point_mei)]
            point_dir = os.path.join(path_dir, str(num)) 
            file_path = os.path.join(point_dir, f"lesion_{num + count_lesion}_mask_{count_mask}.nii.gz") 
            nifti_lesion = nib.Nifti1Image(mask_lesion_i, template_nib.affine)
            nib.save(nifti_lesion,file_path)
            #np.savez(file_path, volume = mask_lesion_i, centroid=centroid_region, allow_pickle = True) #save dans le directory nommé num un fichier contenant le volume de la lésion
            count_lesion +=1
        count_mask+=1
    return points, dict_point_clés



def create_likelihood_ventricules(ventricules_mask, label_MNI, likelihood_map):
    
    
    likelihood_nib = nib.load(likelihood_map_path)
    ventricules_mask[ventricules_mask>0]=1
    
    dilated_ventricule_mask = binary_dilation(ventricules_mask,ball(5))
    dilated_vent2 = binary_dilation(ventricules_mask, ball(7))
    area_interest = dilated_ventricule_mask - ventricules_mask
    area_interest_extended = dilated_vent2 - ventricules_mask
    avoid = np.isin(label_MNI,[11,50]).astype(np.uint8)
    avoid[avoid>0] = 1
    area_interest[avoid>0] = 0
    area_interest = area_interest + likelihood_map
    area_interest = (area_interest - area_interest.min())/(area_interest.max() - area_interest.min())
    area_interest[area_interest_extended==0] = 0
    # area_interest -= area_interest.min()
    # max = area_interest.max()
    # norm_area = area_interest/max
    area_nift = nib.Nifti1Image(area_interest,likelihood_nib.affine)
    nib.save(area_nift, f'/home/jdrochmans/data/juliette/label/ventricule_area_interest_162.nii.gz')
    return area_interest


def create_likelihood_cortex(likelihood_map, cortex_mask):
    
    likelihood_map[cortex_mask==0] = 0
    return likelihood_map   

def label_map_synLes(label_map, points,dict_point_clés, facteur_confluence, template_p_T1, nb_lesions):
    #label_map => le masque de segmentation 
    # dict_lesion => mon dictionnaire de lésions
    # facteur confluence => ratio pour que les lésions se touchent 
    
    #dictionnaire de lésions dans l'espace MNI => registrer les label _map 
    path_in_name = os.path.basename(label_map[0]) #check comment se nomme les label_map
    name = path_in_name.split("_")[0]
    template_nib_T1 = nib.load(template_p_T1)
    print(name)
    #points, dict_points_clés = shape_dir(likelihood_map,path_dir, dossier_mask,reg_dir, template_p_T1)
    
    points, dict_point_clés = create_points(likelihood_map,path_dir,20)
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
   
    print(forward_transforms)
    print(backward_transforms)
    image_p = label_map[0]
    output =  os.path.join(dossier_registered_mask, f'label_HC_{name}_reg.nii.gz')
    cmd = f"antsApplyTransforms -i {image_p} -r {template_p_T1} -n {'genericLabel'} -t {forward_transforms[1]} -t {forward_transforms[0]} -o {output}"
    subprocess.Popen(cmd, shell = True).wait()
    aligned_image = ants.image_read(output)
    template = nib.load(image_p)
    # image_files = [os.path.join(dossier_registered_image, f) for f in os.listdir(dossier_registered_image) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_registered_image, f)) ] 
    # image_files_filtered = [f for f in image_files if name in os.path.basename(f)]
    # otsu_mask = otsu(image_files_filtered[0])
    label_MNI = nib.load(output)
    dim_x,dim_y,dim_z = label_MNI.shape
    label = 86
    label2 = 86
    label_MNI = label_MNI.get_fdata()
    label_MNI_lesion_sem = label_MNI.copy()
    mask_synth_lesions = np.zeros((label_MNI.shape), dtype = np.float32)
    mask_synth_lesions_binary = np.zeros((label_MNI.shape), dtype = np.float32)
    mask_lesion_conf = np.zeros((label_MNI.shape), dtype = np.float32)
    cortex_reg = [os.path.join(dossier_cortex, f) 
        for f in os.listdir(dossier_cortex) 
        if f.startswith(f"mask_cortex_registered_{name}_HC")]
    cortex_mask = nib.load(cortex_reg[0]).get_fdata()
    for i in range(nb_lesions):
        cortex_les = False
        bool = False
        max_attempt = 20
        attempt = 0
        print('je reviens')
       # brain_mask = np.isin(label_MNI, [507,511,515,516,514,502,509,24]).astype(np.uint8) 
        ventricule_mask = np.isin(label_MNI, [4,14,15,43,72,49,10]).astype(np.uint32)
       # left_brain_mask = np.isin(label_MNI, [2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,26]).astype(np.uint8)
      #  right_brain_mask = np.isin(label_MNI,[41,42,43,44,46,47,49,50,51,52,53,54,58,60]).astype(np.uint8)
       # lesion_mask = np.isin(label_MNI, [25,57]).astype(np.uint8)
        avoid_CC = np.isin(label_MNI, [251,252,253,254,255]).astype(np.uint32)
        lesion_mask = np.isin(label_MNI, 77).astype(np.uint32)
        bool = False
        new_mask_lesion = np.zeros((label_MNI.shape), dtype = np.float32)
        while(bool == False and attempt < max_attempt):
            max_iter = 10
            iteration = 0
            dossier_assoc_num = []
            while (dossier_assoc_num == [] and iteration < max_iter): #pas de lésions dans ce dossier 
                good_cand = False
                while good_cand == False :
                    if(nb_lesions < 20):
                        flat_likelihood = likelihood_map.flatten()
                       
                        valid_indices = np.where(flat_likelihood > 0)[0]
                        valid_probs = flat_likelihood[valid_indices]
                        valid_probs /= valid_probs.sum()  # Normalize

                        voxel_index = np.random.choice(valid_indices, p=valid_probs)
                        new_centroid = np.unravel_index(voxel_index, likelihood_map.shape)
              
                        candidate = new_centroid
                        # and => test juste couper si dans ventricule ? 
                        if(ventricule_mask[candidate[0], candidate[1], candidate[2]] == 0 and cortex_mask[candidate[0], candidate[1], candidate[2]] == 0 and label_MNI[candidate[0], candidate[1], candidate[2]]!= 0 and mask_synth_lesions[candidate[0], candidate[1], candidate[2]] == 0 and lesion_mask[candidate[0], candidate[1], candidate[2]]==0) :
                            good_cand = True
                            new_centroid = candidate
                    else :
                        if(i < 50):
                            print('jepasse')
                            norm_map = create_likelihood_ventricules(ventricule_mask,label_MNI, likelihood_map)
                            
                            flat_likelihood = norm_map.flatten()
                           
                            valid_indices = np.where(flat_likelihood > 0)[0]
                            valid_probs = flat_likelihood[valid_indices].astype(np.float64)
                            valid_probs /= valid_probs.sum()  # Normalize

                            voxel_index = np.random.choice(valid_indices, p=valid_probs)
                            new_centroid = np.unravel_index(voxel_index, norm_map.shape)
                           
                            candidate = new_centroid
                            # and => test juste couper si dans ventricule ? 
                            #test si j'enleve la condition sur masksynthlesion?
                            if(ventricule_mask[candidate[0], candidate[1], candidate[2]] == 0 and cortex_mask[candidate[0], candidate[1], candidate[2]] == 0 and label_MNI[candidate[0], candidate[1], candidate[2]]!= 0 and lesion_mask[candidate[0], candidate[1], candidate[2]]==0) :
                                good_cand = True
                                new_centroid = candidate 
                        
                        if(nb_lesions-nb_lesions/10 < i <= nb_lesions) :
                        #if(cortex_les == True):
                            #lesions corticales
                            cortex_map = create_likelihood_cortex(likelihood_map,cortex_mask) 
                            flat_likelihood = cortex_map.flatten()
                            
                            valid_indices = np.where(flat_likelihood > 0)[0]
                            valid_probs = flat_likelihood[valid_indices].astype(np.float64)
                            valid_probs /= valid_probs.sum()  # Normalize

                            voxel_index = np.random.choice(valid_indices, p=valid_probs)
                            new_centroid = np.unravel_index(voxel_index, norm_map.shape)
                            #new_cent = random.choice(np.argwhere(likelihood_map > 0.7)).astype(int)
                            #new_centroid = np.array(new_centroid).astype(int)
                            candidate = new_centroid
                            # and => test juste couper si dans ventricule ? 
                            #test si j'enleve la condition sur masksynthlesion?
                            good_cand = True
                            new_centroid = candidate 
                            cortex_les = True
                            
                            
                            
                            
                        else:
                            
                            flat_likelihood = likelihood_map.flatten()
                           
                            valid_indices = np.where(flat_likelihood > 0)[0]
                            valid_probs = flat_likelihood[valid_indices]
                            valid_probs /= valid_probs.sum()  # Normalize

                            voxel_index = np.random.choice(valid_indices, p=valid_probs)
                            new_centroid = np.unravel_index(voxel_index, likelihood_map.shape)
                           
                            candidate = new_centroid
                            # and => test juste couper si dans ventricule ? 
                            print(candidate)
                            #mask_synth_lesions[candidate[0], candidate[1], candidate[2]] == 0
                            if(ventricule_mask[candidate[0], candidate[1], candidate[2]] == 0 and cortex_mask[candidate[0], candidate[1], candidate[2]] == 0 and label_MNI[candidate[0], candidate[1], candidate[2]]!= 0 and mask_synth_lesions[candidate[0], candidate[1], candidate[2]] == 0 and lesion_mask[candidate[0], candidate[1], candidate[2]]==0) :
                                good_cand = True
                                new_centroid = candidate 
                            
                            
                point_chosen = 0
                cnt = 1000 #changer ici par une valeur de distance 
                for point in points :
                    if  np.linalg.norm(new_centroid - point) < cnt : 
                        point_chosen = point
                        cnt = np.linalg.norm(new_centroid - point)
                
                num = dict_point_clés[tuple(point_chosen)]

                dossier_assoc_num =  os.listdir(os.path.join(path_dir,str(num)) ) 
                dossier_assoc_num_conf = os.listdir(os.path.join(dict_lesions_confluent,str(num)))
                dossier_assoc_num_cortex = os.listdir(os.path.join(dict_lesions_corticales,str(num)))
                iteration +=1
            
            if(dossier_assoc_num == [] and iteration >= max_iter):
                print('Couldnt find a file in a reasonable number of iterations')

            else :
                
                volume_random_data_shape = None
                conf = False
                if(50<= i < 50+nb_lesions/5):
                    while(np.prod(volume_random_data_shape) == 0 or volume_random_data_shape == None):
                        random_file = random.choice(dossier_assoc_num_conf)
                        random_file_path = os.path.join(os.path.join(dict_lesions_confluent,str(num)), random_file) 
                        volume = (nib.load(random_file_path)).get_fdata() 
                      
                        volume_random_data_shape = volume.shape    
                        
                    conf = True
                    print(np.unique(volume).astype(np.uint32))
                    print(volume.shape)
                    print(f'file choisi : {random_file_path}')
                elif(cortex_les == True):
                    while(np.prod(volume_random_data_shape) == 0 or volume_random_data_shape == None):
                        random_file = random.choice(dossier_assoc_num_cortex)
                        random_file_path = os.path.join(os.path.join(dict_lesions_corticales,str(num)), random_file) 
                        type_les = random_file_path.split('_')[1]
                        volume = (nib.load(random_file_path)).get_fdata() 
                        volume_random_data_shape = volume.shape
                else :
                    while(np.prod(volume_random_data_shape) == 0 or volume_random_data_shape == None):
                        random_file = random.choice(dossier_assoc_num)
                        random_file_path = os.path.join(os.path.join(path_dir,str(num)), random_file) 
                        volume = (nib.load(random_file_path)).get_fdata() 
                        volume_random_data_shape = volume.shape
                    if(i<50 and nb_lesions>20):
                        mult = np.random.normal(loc=1.4, scale=0.5)
                        mult = np.clip(mult, 1.0, 2.0)   
                    else:
                        mult = np.random.normal(loc=1.0, scale=0.3)
                        mult = np.clip(mult, 0.2, 1.3)
                    print(f'multiplicateur utilisé: {mult}')
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
                
                
                # tol_ratio_1 = np.sum(lesion_mask[start_x:stop_x, start_y:stop_y, start_z:stop_z] > 0) / ((stop_x-start_x) * (stop_y-start_y) * (stop_z-start_z))
                tol_ratio_2 = np.sum(mask_synth_lesions[start_x:stop_x, start_y:stop_y, start_z:stop_z] > 0) / ((stop_x-start_x) * (stop_y-start_y) * (stop_z-start_z))
                if(start_x<stop_x and start_y < stop_y and start_z < stop_z  and tol_ratio_2 < facteur_confluence): 
                    bool = True  
                else :
                    attempt +=1
        if(attempt == max_attempt):
            print(f"Aucune position valide trouvée après {max_attempt} essais, abandon de cette lésion.")
            
            continue 
        else :
                
            x_sh = stop_x - start_x
            y_sh = stop_y - start_y
            z_sh = stop_z - start_z
            target_shape = (x_sh,y_sh,z_sh) 
            
            print(f'Centroid choisi : {new_centroid}')
            print(f'Placement prévu de la lésion : start=({start_x},{start_y},{start_z}), stop=({stop_x},{stop_y},{stop_z})')   
                #mask[start_x:stop_x,start_y:stop_y,start_z:stop_z] = np.logical_or(mask[start_x:stop_x,start_y:stop_y,start_z:stop_z],volume[:target_shape[0], :target_shape[1], :target_shape[2]]).astype(np.uint8)

            
            #changer ici : regarder les labels deja utilisés par les lésions déja placées
            lesion_patch = volume[:target_shape[0], :target_shape[1], :target_shape[2]]
            
            new_mask_lesion[start_x:stop_x,start_y:stop_y,start_z:stop_z] = lesion_patch
            print(np.unique(new_mask_lesion).astype(np.uint32))
            if(cortex_les== True):
                if(type_les == 'juxtacort'):
                    
                    new_mask_lesion[cortex_mask>0] = 0
                    
                else :
                    inv_cortex_mask = 1 - cortex_mask
                    new_mask_lesion[inv_cortex_mask>0] = 0
                while(np.isin(label_MNI, label).astype(np.uint32).any()):
                    label +=1
                
                label_MNI[new_mask_lesion>0] = np.uint32(label)
                mask_synth_lesions[new_mask_lesion>0] = np.uint32(label)
                label_MNI_lesion_sem[new_mask_lesion>0] = np.uint32(label2)
                mask_synth_lesions_binary[new_mask_lesion>0] = np.uint32(label2)
                    
                    
            
            else :
                new_mask_lesion[ventricule_mask>0] = 0
                new_mask_lesion[cortex_mask>0] = 0
                #new_mask_lesion[brain_mask>0] = 0
                new_mask_lesion[label_MNI==0] = 0
                new_mask_lesion[lesion_mask>0] = 0
                new_mask_lesion[avoid_CC>0] = 0
                if(conf == False):
                    while(np.isin(label_MNI, label).astype(np.uint32).any()):
                        label +=1
                    
                    label_MNI[new_mask_lesion>0] = np.uint32(label)
                    mask_synth_lesions[new_mask_lesion>0] = np.uint32(label)
                    label_MNI_lesion_sem[new_mask_lesion>0] = np.uint32(label2)
                    mask_synth_lesions_binary[new_mask_lesion>0] = np.uint32(label2)
                else : 
                    labels1 = np.unique(new_mask_lesion).astype(np.uint32)
                    print(f'LABELS = {labels1}')
                   # sorted_labels = np.sort(labels)
                    for lab in labels1:
                        
                        if lab == 0:
                            continue
                        else :
                            while(np.isin(label_MNI, label).astype(np.uint32).any()):
                                label +=1
                            # print('lesion_conf')
                            # print(f'label = {label}')
                            label_MNI[new_mask_lesion==lab] = np.uint32(label)
                            mask_synth_lesions[new_mask_lesion==lab] = np.uint32(label)
                            label_MNI_lesion_sem[new_mask_lesion==lab] = np.uint32(label)
                            mask_synth_lesions_binary[new_mask_lesion==lab] = np.uint32(label)
                            mask_lesion_conf[new_mask_lesion==lab] = np.uint32(label)
                            new_mask_lesion[new_mask_lesion==lab] = np.uint32(label)
                
                
                
            
                
    lesion_mask_nib = nib.Nifti1Image(mask_synth_lesions,template_nib_T1.affine)
    label_MNI_nib =  nib.Nifti1Image(label_MNI,template_nib_T1.affine)
    label_MNI_semantic_nib = nib.Nifti1Image(label_MNI_lesion_sem,template_nib_T1.affine)
    mask_lesion_conf_nib = nib.Nifti1Image(mask_lesion_conf,template_nib_T1.affine)
    #mask_lesion_binary_nib = nib.Nifti1Image(mask_synth_lesions_binary,template_nib_T1.affine) 
    nib.save(lesion_mask_nib,f'/home/jdrochmans/data/juliette/reg/mask_synLes_HC_inst_{name}.nii.gz')
    nib.save(label_MNI_nib,f'/home/jdrochmans/data/juliette/label/label_MNI_HC_inst_{name}.nii.gz')
    nib.save(label_MNI_semantic_nib, f'/home/jdrochmans/data/juliette/label/label_MNI_sem_{name}.nii.gz')
    nib.save(mask_lesion_conf_nib,f'/home/jdrochmans/data/juliette/label/mask_les_conf_MNI_{name}.nii.gz' )
    # nib.save(mask_lesion_binary_nib,f'/home/jdrochmans/data/juliette/reg/mask_lesions_classe_{name}.nii.gz')

    lesion_mask_p = f'/home/jdrochmans/data/juliette/reg/mask_synLes_HC_inst_{name}.nii.gz'
    label_MNI_p = f'/home/jdrochmans/data/juliette/label/label_MNI_HC_inst_{name}.nii.gz'
    label_MNI_sem_p = f'/home/jdrochmans/data/juliette/reg/label_MNI_sem_{name}.nii.gz'
    mask_les_conf_p = f'/home/jdrochmans/data/juliette/label/mask_les_conf_MNI_{name}.nii.gz'
    # bin_lesion_mask_p = f'/home/jdrochmans/data/juliette/reg/mask_lesions_classe_{name}.nii.gz'
    
    dossier_new_label = "/home/jdrochmans/data/juliette/label/"
    dossier_new_mask = "/home/jdrochmans/data/juliette/mask/"
    
    output_Newmask1 = os.path.join(dossier_new_label, f'label_lesion_HC_inst_{name}.nii.gz')
    cmd = f"antsApplyTransforms -d 3 -i {label_MNI_p} -r {label_map[0]} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask1}"
    subprocess.Popen(cmd, shell = True).wait()
    
    # output_Newmask_sem = os.path.join(dossier_new_label, f'label_lesion_HC_sem_{name}.nii.gz')
    # cmd = f"antsApplyTransforms -d 3 -i {label_MNI_sem_p} -r {label_map[0]} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask_sem}"
    # subprocess.Popen(cmd, shell = True).wait()
    

    
    output_Newmask2 = os.path.join(dossier_new_mask, f'mask_lesion_HC_sem_{name}.nii.gz')
    cmd = f"antsApplyTransforms -d 3 -i {lesion_mask_p} -r {label_map[0]} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask2}"
    subprocess.Popen(cmd, shell = True).wait()
    
    
    output_Newmask4 = os.path.join(dossier_new_mask, f'mask_lesion_conf_{name}.nii.gz')
    cmd = f"antsApplyTransforms -d 3 -i {mask_les_conf_p} -r {label_map[0]} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask4}"
    subprocess.Popen(cmd, shell = True).wait()
    # output_Newmask3 = os.path.join(dossier_new_mask, f'mask_lesion_classe_{name}.nii.gz')
    # cmd = f"antsApplyTransforms -d 3 -i {bin_lesion_mask_p} -r {label_map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask3}"
    # subprocess.Popen(cmd, shell = True).wait()
    
    return output_Newmask1, name


if __name__ == "__main__":
    print("Début du test")
    facteur_confluence = 0.15
    points, dict_point_clés = create_points(likelihood_map,path_dir,20)
   # nb_image_traitées = len(label_map)
        
    path_inst, name = label_map_synLes(label_map, points, dict_point_clés, facteur_confluence, template_p_T1, 150)
    print(path_inst)
    img_inst_nib = nib.load(path_inst)
    img_inst = img_inst_nib.get_fdata()
    
    img_inst[(img_inst >= 86) & (img_inst < 251)] = 86
    img_inst[img_inst>255] = 86
    
    mask_sem = nib.Nifti1Image(img_inst, img_inst_nib.affine)
    nib.save(mask_sem,f"/home/jdrochmans/data/juliette/label/label_lesion_sem2_HC_{name}.nii.gz" )
    print("Fin du test")
    
    
    