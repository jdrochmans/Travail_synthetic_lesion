
#OLD!!!


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
reg_dir = '/home/jdrochmans/data/juliette/transforms_reg/'
dossier_segmentation = "/home/jdrochmans/data/juliette/seg"
likelihood_map_path = "/home/jdrochmans/data/juliette/likelihood_map_norm_WM30.nii"
likelihood_map = nib.load(likelihood_map_path).get_fdata()
label_map = [os.path.join(dossier_segmentation, f) for f in os.listdir(dossier_segmentation) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_segmentation, f)) ]
dossier_reg_MNI_label_map = ""
template_p_T1 = "/home/jdrochmans/data/juliette/template.nii"
path_dir = os.path.join("/home/jdrochmans/data/juliette/shape_dir/")
dossier_mask = "/home/jdrochmans/data/juliette/Dataset001_BrainLesion/labelsTr"
dossier_registered_mask = "/home/jdrochmans/data/juliette/register_mask"
dossier_registered_image = "/home/jdrochmans/data/juliette/register_image"

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



def label_map_synLes(label_map, points,dict_point_clés, facteur_confluence, template_p_T1, nb_lesions):
    #label_map => le masque de segmentation 
    # dict_lesion => mon dictionnaire de lésions
    # facteur confluence => ratio pour que les lésions se touchent 
    
    #dictionnaire de lésions dans l'espace MNI => registrer les label _map 
    path_in_name = os.path.basename(label_map) #check comment se nomme les label_map
    name = path_in_name.split("_")[0]
    template_nib_T1 = nib.load(template_p_T1)
    
    #points, dict_points_clés = shape_dir(likelihood_map,path_dir, dossier_mask,reg_dir, template_p_T1)
    
    points, dict_point_clés = create_points(likelihood_map,path_dir,20)
    forward_transforms = sorted([
    os.path.join(reg_dir, f)
    for f in os.listdir(reg_dir)
    if(f.startswith(f'{name}'))
    if (
        re.search(r"_0GenericAffine\.mat$", f) or
        re.search(r"_1Warp\.nii(\.gz)?$", f)
    )
])
    
    backward_transforms = sorted(
    [os.path.join(reg_dir, f) 
    for f in os.listdir(reg_dir) 
    if(f.startswith(f'{name}'))
    if (
        re.search(r"_0GenericAffine\.mat$", f) or
        re.search(r"_1InverseWarp\.nii(\.gz)?$", f)
    )]
    ) 
    
    output =  os.path.join(dossier_reg_MNI_label_map, f'image_{name}.nii.gz')
    cmd = f"antsApplyTransforms -i {label_map} -r {template_p_T1} -n {'genericLabel'} -t {forward_transforms[1]} -t {forward_transforms[0]} -o {output}"
    subprocess.Popen(cmd, shell = True).wait()
    aligned_image = ants.image_read(output)
    image_p = output
    
    
    label_MNI = nib.load(image_p)
    dim_x,dim_y,dim_z = label_MNI.shape
    label = 86
    label2 = 86
    label_MNI = label_MNI.get_fdata()
    label_MNI_lesion_sem = label_MNI.copy()
    mask_synth_lesions = np.zeros((label_MNI.shape), dtype = np.float32)
    mask_synth_lesions_binary = np.zeros((label_MNI.shape), dtype = np.float32)
    for i in range(nb_lesions):
        bool = False
        max_attempt = 200
        attempt = 0
        cortex_mask = np.isin(label_MNI, [42, 3, 8, 47]).astype(np.uint8)
        
        ventricule_mask = np.isin(label_MNI, [4,14,15,43,72]).astype(np.uint8)
        lesion_mask = np.isin(label_MNI, 77).astype(np.uint8)
        bool = False
        new_mask_lesion = np.zeros((label_MNI.shape), dtype = np.float32)
        while(bool == False and attempt < max_attempt):
            max_iter = 100
            iteration = 0
            dossier_assoc_num = []
            while (dossier_assoc_num == [] and iteration < max_iter): #pas de lésions dans ce dossier 
                good_cand = False
                while good_cand == False :
                    flat_likelihood = likelihood_map.flatten()
                    flat_likelihood_normalized = flat_likelihood / flat_likelihood.sum()
                    voxel_index = np.random.choice(len(flat_likelihood_normalized), p=flat_likelihood_normalized)
                    new_centroid = np.unravel_index(voxel_index, likelihood_map.shape)
                    new_centroid = np.array(new_centroid).astype(int)
                    candidate = new_centroid
                    
                    #ventricule_mask[candidate[0], candidate[1], candidate[2]] == 0 and => test juste couper si dans ventricule ? 
                    if(cortex_mask[candidate[0], candidate[1], candidate[2]] == 0 and lesion_mask[candidate[0], candidate[1], candidate[2]]== 0 and label_MNI[candidate[0], candidate[1], candidate[2]]!= 0 and mask_synth_lesions[candidate[0], candidate[1], candidate[2]] == 0):
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
                iteration +=1
            
            if(dossier_assoc_num == [] and iteration >= max_iter):
                print('Couldnt find a file in a reasonable number of iterations')

            else :
                
                
                volume_random_data_shape = None
                while(np.prod(volume_random_data_shape) == 0 or volume_random_data_shape == None):
                    random_file = random.choice(dossier_assoc_num)
                    random_file_path = os.path.join(os.path.join(path_dir,str(num)), random_file) 
                    volume = (nib.load(random_file_path)).get_fdata() 
                    volume_random_data_shape = volume.shape
                    
                mult = np.random.normal(loc=1.0, scale=0.5)
                mult = np.clip(mult, 0.2, 2.0)
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
                
                
                tol_ratio_1 = np.sum(lesion_mask[start_x:stop_x, start_y:stop_y, start_z:stop_z] > 0) / ((stop_x-start_x) * (stop_y-start_y) * (stop_z-start_z))
                tol_ratio_2 = np.sum(mask_synth_lesions[start_x:stop_x, start_y:stop_y, start_z:stop_z] > 0) / ((stop_x-start_x) * (stop_y-start_y) * (stop_z-start_z))
                if(start_x<stop_x and start_y < stop_y and start_z < stop_z and tol_ratio_1 == 0 and tol_ratio_2 < facteur_confluence): 
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
            new_mask_lesion[lesion_mask>0] = 0
            new_mask_lesion[cortex_mask>0] = 0
            new_mask_lesion[ventricule_mask>0] = 0
            new_mask_lesion[label_MNI==0] = 0
            while(np.isin(label_MNI, label).astype(np.uint8).any()):
                label +=1
            
            
            label_MNI[new_mask_lesion>0] = label
            mask_synth_lesions[new_mask_lesion>0] = label
            label_MNI_lesion_sem[new_mask_lesion>0] = np.uint8(label2)
            mask_synth_lesions_binary[new_mask_lesion>0] = np.uint8(label2)
            
                
    lesion_mask_nib = nib.Nifti1Image(mask_synth_lesions,template_nib_T1.affine)
    label_MNI_nib =  nib.Nifti1Image(label_MNI,template_nib_T1.affine)
    label_MNI_semantic_nib = nib.Nifti1Image(label_MNI_lesion_sem,template_nib_T1.affine)
    mask_lesion_binary_nib = nib.Nifti1Image(mask_synth_lesions_binary,template_nib_T1.affine) 
    # nib.save(lesion_mask_nib,f'/home/jdrochmans/data/juliette/reg/mask_lesions_instance_{name}.nii.gz')
    nib.save(label_MNI_nib,f'/home/jdrochmans/data/juliette/reg/label_MNI_{name}.nii.gz')
    nib.save(label_MNI_semantic_nib, f'/home/jdrochmans/data/juliette/reg/label_MNI_sem_{name}.nii.gz')
    # nib.save(mask_lesion_binary_nib,f'/home/jdrochmans/data/juliette/reg/mask_lesions_classe_{name}.nii.gz')

    # lesion_mask_p = f'/home/jdrochmans/data/juliette/reg/mask_lesions_instance_{name}.nii.gz'
    label_MNI_p = f'/home/jdrochmans/data/juliette/reg/label_MNI_{name}.nii.gz'
    # label_MNI_sem_p = f'/home/jdrochmans/data/juliette/reg/label_MNI_sem_{name}.nii.gz'
    # bin_lesion_mask_p = f'/home/jdrochmans/data/juliette/reg/mask_lesions_classe_{name}.nii.gz'
    
    dossier_new_label = "/home/jdrochmans/data/juliette/label/"
    # dossier_new_mask = "/home/jdrochmans/data/juliette/mask/"
    
    output_Newmask1 = os.path.join(dossier_new_label, f'label_lesion_instantiel_{name}.nii.gz')
    cmd = f"antsApplyTransforms -d 3 -i {label_MNI_p} -r {label_map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask1}"
    subprocess.Popen(cmd, shell = True).wait()
    
    # output_Newmask_sem = os.path.join(dossier_new_label, f'label_lesion_sem_{name}.nii.gz')
    # cmd = f"antsApplyTransforms -d 3 -i {label_MNI_sem_p} -r {label_map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask_sem}"
    # subprocess.Popen(cmd, shell = True).wait()
    
    

    
    # output_Newmask2 = os.path.join(dossier_new_mask, f'mask_lesion_instantiel_{name}.nii.gz')
    # cmd = f"antsApplyTransforms -d 3 -i {lesion_mask_p} -r {label_map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask2}"
    # subprocess.Popen(cmd, shell = True).wait()
    
    # output_Newmask3 = os.path.join(dossier_new_mask, f'mask_lesion_classe_{name}.nii.gz')
    # cmd = f"antsApplyTransforms -d 3 -i {bin_lesion_mask_p} -r {label_map} -n genericLabel -t [{backward_transforms[0]},1] -t {backward_transforms[1]} -o {output_Newmask3}"
    # subprocess.Popen(cmd, shell = True).wait()
    
    return output_Newmask1, name


if __name__ == "__main__":
    print("Début du test")
    facteur_confluence = 0.15
    points, dict_point_clés = create_points(likelihood_map,path_dir,20)
   # nb_image_traitées = len(label_map)
    nb_image_traitées = 1
    for map in range(nb_image_traitées):
        
        path_inst, name = label_map_synLes(label_map[map], points, dict_point_clés, facteur_confluence, template_p_T1, 150)
        print(path_inst)
        img_inst_nib = nib.load(path_inst)
        img_inst = img_inst_nib.get_fdata()
        
        img_inst[(img_inst >= 86) & (img_inst < 251)] = 86
        img_inst[img_inst>255] = 86
        
        mask_sem = nib.Nifti1Image(img_inst, img_inst_nib.affine)
        nib.save(mask_sem,f"/home/jdrochmans/data/juliette/label/label_lesion_sem2_{name}.nii.gz" )
    print("Fin du test")