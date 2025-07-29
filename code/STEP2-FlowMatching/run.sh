# Step 3. AutoEncoder Model for Tumour 
cd  ./STEP1-AutoencoderModel
cd ../STEP1-AutoencoderModel
mamba activate mmtumor

temp_dir=~/HDD_8T_2


gpu=2,3
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 \
        train_autoencoder.py   \
        --batch_size   1  \
        --gpu  $gpu    --n_epochs    200   \
        --lr    2.5e-5    --input_modality  t1c t1n t2w t2f  \
        --cache_dir  $temp_dir/Tumor/AE/results/cache-ae4   \
        --output_dir $temp_dir/Tumor/AE/results/output \
        --data_dir   ~/hao/data/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData     --DEBUG



