#!/usr/bin/env python

"""
Script to preprocess FreeSurfer surfaces for training
TopoFit models. If this code is useful to you, please cite:

TopoFit: Rapid Reconstruction of Topologically-Correct Cortical Surfaces
Andrew Hoopes, Juan Eugenio Iglesias, Bruce Fischl, Douglas Greve, Adrian Dalca
Medical Imaging with Deep Learning. 2022.
"""

import os
import argparse
import shutil
import surfa as sf
import topofit


parser = argparse.ArgumentParser()
parser.add_argument('subj', help='full path to recon-all FreeSurfer subject')
parser.add_argument('--xhemi', action='store_true', help='Xhemi')
args = parser.parse_args()

# make sure FS has been sourced in the env
if shutil.which('mri_surf2surf') is None:
    sf.system.fatal('cannot find mri_surf2surf in system path, has freesurfer been sourced?')

fshome = os.environ.get('FREESURFER_HOME')
if fshome is None:
    sf.system.fatal('FREESURFER_HOME is not set in the env, has freesurfer been sourced?')

# set SUBJECTS_DIR manually based on path to input subject
subjects_dir = os.path.dirname(args.subj)
os.environ['SUBJECTS_DIR'] = subjects_dir
basename = os.path.basename(args.subj)
print(subjects_dir)

# make sure fsaverage is accessible from the SUBJECTS_DIR
if not os.path.exists(os.path.join(subjects_dir, 'fsaverage')):
    sf.system.fatal(
        f'fsaverage subject does not exist in the base directory `{subjects_dir}`, '
         'so you must link it by running:\n\n'
        f'ln -s $FREESURFER_HOME/subjects/fsaverage {subjects_dir}/fsaverage\n')

# compute talairach.xfm.lta if it doesn't exist
if not os.path.isfile(f'{args.subj}/mri/transforms/talairach.xfm.lta'):
    cmd = f'lta_convert --src {args.subj}/mri/orig.mgz ' \
          f'--trg {fshome}/average/mni305.cor.mgz --inxfm {args.subj}/mri/transforms/talairach.xfm ' \
          f'--outlta {args.subj}/mri/transforms/talairach.xfm.lta --subject fsaverage --ltavox2vox'
    if sf.system.run(cmd) != 0:
        sf.system.fatal('could not run lta_convert command')

# cycle through hemispheres
if not args.xhemi: 
    for i, hemi in enumerate('lh', 'rh'):
        cmd = f'mri_surf2surf --mapmethod nnf --s {basename} --hemi {hemi} --sval-xyz white ' \
              f'--trgsubject fsaverage --tval {args.subj}/surf/{hemi}.white.ico.surf ' \
              f'--tval-xyz {args.subj}/mri/norm.mgz'
        if sf.system.run(cmd) != 0:
            sf.system.fatal('could not run mri_surf2surf command')
else:
    hemi = 'lh';
    cmd = f'mri_surf2surf --mapmethod nnf --s {basename} --hemi {hemi} --sval-xyz white ' \
          f'--trgsubject fsaverage_sym --tval {args.subj}/surf/{hemi}.white.ico.xhemi.surf ' \
          f'--tval-xyz {args.subj}/mri/norm.mgz --srcsurfreg fsaverage_sym.sphere.reg --trgsurfreg sphere.reg'
    if sf.system.run(cmd) != 0:
        sf.system.fatal('could not run mri_surf2surf command')


print('\nTopoFit preprocessing complete!')
