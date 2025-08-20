
cd /home/hao/repo/FM-translation
cd /home/hao/repo/FM-translation/code/STEP1-AutoencoderModel




conda activate progression

task=CT 
data_files="../data/${task}_pair.csv"
temp=/date/hao/FM

# -------------------- Train AE -----------------------
gpu=2,3,5
data_files=../data/files/CT_pair.csv

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 --main_process_port 19150 \
      train_ct_autoencoder.py --gpu $gpu  \
      --dataset_csv $data_files   --task  $task \
      --batch_size 4  --n_epochs  200    --lr 1e-5 \
      --cache_dir  ${temp}/cache/CT_single      \
      --input_modality   CT CTC    --missing_modality   CTC  \
      --output_dir  ${temp}/checkpoint/CT/AE         --DEBUG      








# -------------------- Train Flow Matching -----------------------
gpu=0,1

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --multi_gpu --mixed_precision fp16 \
        train_flowmatching.py   \
        --batch_size   4  \
        --gpu  $gpu      --input_modality  t1n   \
        --lr    2.5e-5     \
        --n_epochs  500  \
        --grad_accum_steps 3  \
        --latent_dir $temp_dir/Tumor/AE/results/brats_latent/t1n  \
        --cache_dir  $temp_dir/Tumor/AE/results/cache_dm_t1n   \
        --output_dir $temp_dir/Tumor/AE/results/output \
        --aekl_ckpt  $temp_dir/Tumor/AE/results/output/ae-50-t1n.pth   \
        --data_dir   ~/hao/data/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData     --DEBUG








