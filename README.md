# Semi-Supervised-nnUNet
This is a nnUNet for semi-supervised. We write a semi-supervised nnUNet for abdominal organ segmentation. Don' t doubt， although that our code is father from nnFormer.

## Environment settings

Install Package

    cd Semi-Supervised-nnUNet
    python setup_bg.py install  # we have revised package of batchgenerator for semi-supervised learning.
    python setup.py install     # install nnformer

Test Environment

    nnFormer_train -h  ## if anypackage is miss, please install yourself by pip.


## Training Stage

We default that you have train a simple nnUNet model by label data. Therefore, pseudo-labels for unlabel data have been obtained. In this case, all of the data can be regarded as label data for pre-processing and training.

    nnFormer_plan_and_preprocess -t 134 -no_pp # we set task id to be 134, if something error has happen, please set 134 like us.

After the code above, a plan file will be generated. We suggest that the batch size can be changed to 1, the patch size can be changed to smaller, and the target resolution can be changed to 1.5x1.5x2.5, which can be achieve by revised the plan file.

    nnFormer_plan_and_preprocess -t 134 -pl2d None  # preprocessing data

Train a Dual nnUNet or Light UNet by yourself.

    nnFormer_train 3d_fullres nnFormerTrainerV2_Dual -t 134 --fold all --gpu "0,1" --npz   ## Dual nnUNet, CE+DICE loss

    nnFormer_train 3d_fullres nnFormerTrainerV2_Dual_Robust -t 134 --fold all --gpu "0,1" --npz  ## Dual nnUNet, TCE+NRD loss

    nnFormer_train 3d_fullres nnFormerTrainerV2_Dual_Light -t 134 --fold all --gpu "0,1" --npz  ## Dual Light UNet, TCE+NRD loss


## Inference Stage

    nnFormer_predict -i "/workspace/inputs" -o "/workspace/outputs" -t 134 -tr nnFormerTrainerV2_Dual_Light -chk 'model_best' -m 3d_fullres -f 'all' --num_threads_preprocessing 6 --num_threads_nifti_save 2