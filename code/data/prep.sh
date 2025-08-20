

cd /home/hao/repo/FM-translation/code/data
cd code/data
conda activate progression




python new_analysis.py  # ??


# Plot the original distribution of H, W and D
python s1_analyze_res_distribution.py    --task  CT   --data_dir  /home/yifan/data   \
            --out_csv  ./CT_full.csv  --plots_dir  ./plots    


# Resample CT into 256x256 spatial, 2.5x depth sampling than spatial sampling
python s2_ct_size_standardize_2.5x.py -d /home/yifan/data  --debug



# Generate csv for training
python s3_gen_csv.py   --task CT  --data_dir  /date/hao/PairedContrast/CT/low_256x256_2.5xD  

# save files to data/files/CT_pair.csv

python s4_visualization.py   --modality ct  --data_dir  /date/hao/PairedContrast/CT/low_256x256_2.5xD  --out_dir "./vis"




# This is for CT-CTC pair
python s5_3D_to_2D.py --input /date/hao/PairedContrast/CT/low_256x256_2.5xD  \
    --output /date/hao/PairedContrast/CT/low_256x256_2Dimension \
    --vis ./2Dvis





