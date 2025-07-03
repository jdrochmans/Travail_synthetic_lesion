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
reg_dir = '/home/jdrochmans/data/juliette/transforms_reg/'
dossier_segmentation = "/home/jdrochmans/data/juliette/seg"
likelihood_map_path = "/home/jdrochmans/data/juliette/likelihood_map_norm_WM30.nii"
likelihood_map = nib.load(likelihood_map_path).get_fdata()
label_map = [os.path.join(dossier_segmentation, f) for f in os.listdir(dossier_segmentation) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_segmentation, f)) ]
dossier_reg_MNI_label_map = ""
template_p_T1 = "/home/jdrochmans/data/juliette/template.nii"
path_dir = os.path.join("/home/jdrochmans/data/juliette/shape_dir_confluent/")
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
        #refaire avec regionprops puis boucler sur toutes les régions => si plusieurs labels : ajouter au dictionnaire comme avant 
        regions = regionprops(mask.astype(np.int32))
        count_lesion = 0
        
        for region in regions :
            count = 0
            minx, miny, minz = region.bbox[0], region.bbox[1], region.bbox[2]
            maxx,maxy,maxz = region.bbox[3], region.bbox[4], region.bbox[5]
            mask_region = mask[minx:maxx,miny:maxy,minz:maxz]
            labels_in_region = np.unique(mask_region[mask_region > 0])
            
            if(len(labels_in_region)>1) :
                print(labels_in_region)
                #full_mask =  np.isin(mask_region, labels_in_region).astype(np.uint8)
                centroid_region = region.centroid
            
                indices = np.argwhere(mask_region>0)
                indice2 = np.max(indices,axis = 0)
            
                x_max = indice2[0] +1
                y_max = indice2[1] +1
                z_max = indice2[2] +1
                
                indice3 = np.min(indices,axis = 0)
                x_min = indice3[0]
                y_min = indice3[1]
                z_min = indice3[2] 

                mask_lesion_i = mask_region.astype(np.uint32)
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


if __name__ == "__main__":
    print("Début du test")
    # facteur_confluence = 0.15
    points, dict_point_clés = create_points(likelihood_map,path_dir,20)
   # nb_image_traitées = len(label_map)
    points, dict_point_clés = shape_dir(likelihood_map,path_dir,dossier_mask,reg_dir,template_p_T1)
    print("Fin du test")