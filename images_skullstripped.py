
import os
from os.path import join as pjoin
from os.path import exists as pexists
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path

dossier_image_MNI = "/home/jdrochmans/data/juliette/training_label_maps"
output_seg = "/home/jdrochmans/data/juliette/training_label_maps/crop"
label_map = [os.path.join(dossier_image_MNI, f) for f in os.listdir(dossier_image_MNI) if f.endswith(('.nii', '.nii.gz')) and os.path.isfile(os.path.join(dossier_image_MNI, f)) ]


def extract_brain(label_map):
    
    for map in label_map :
        path_in_name = os.path.basename(map)
        name = path_in_name.split("_")[2].split('.')[0]
        image = nib.load(map)
        image_data = image.get_fdata()
        
        label = np.unique(image_data).astype(np.uint8)
        for lab in label : 
            if(lab >=500):
                image_data[lab] = 0
            
            if(lab==14 or lab==15 or lab==16):
                image_data[lab] = 0
        img_no_skull = nib.Nifti1Image(image_data, image.affine)
        nib.save(img_no_skull,os.path.join(output_seg, f'Synthseg_noskull_{name}.nii.gz') )
        
        
extract_brain(label_map)