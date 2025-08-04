# flow-matching-medical-image-translation

# Contents
- [Dataset & Structure](#dataset--structure)
  - [MR](MR)
    - [Brain](#brats17)
    - [Breast](#breast)
  - [CT](#CT)
    - [Uterus Ovary](#uterus-ovary)
    - [Adrenal](#adrenal)
    - [Bladder Kidney](#bladder-kidney)
    - [Lung](#lung)
    - [Stomach Colon Liver Pancreas](#stomach-colon-liver-pancreas)
- [backbone](#backbone)
- [baseline](#baseline)
    - [flow-matching](#flow-matching)
    - []()
    - []()
   

# Dataset & Structure
## MR
### Brain
```markdown
Brain_MR_train_val_test
├── train
│   ├── TCGA-02-0006
│   │   ├── TCGA-02-0006_1996.08.23_t1.nii.gz
│   │   ├── TCGA-02-0006_1996.08.23_t1Gd.nii.gz
│   │   ├── TCGA-02-0006_1996.08.23_t2.nii.gz
│   │   ├── TCGA-02-0006_1996.08.23_flair.nii.gz
│   │   ├── TCGA-02-0006_1996.08.23_GlistrBoost.nii.gz   （mask）
│   │   ├── TCGA-02-0006_1996.08.23_GlistrBoost_ManuallyCorrected.nii.gz （人工矫正过的mask，不是每一个都有）
│   └── ...
├── val
│   ├── TCGA-08-0350
│   │   ├── TCGA-08-0350_1998.12.15_t1.nii.gz
│   │   ├── TCGA-08-0350_1998.12.15_t1Gd.nii.gz
│   │   ├── TCGA-08-0350_1998.12.15_t2.nii.gz
│   │   ├── TCGA-08-0350_1998.12.15_flair.nii.gz
│   │   ├── TCGA-08-0350_1998.12.15__GlistrBoost.nii.gz
│   │   ├── TCGA-08-0350_1998.12.15_GlistrBoost_ManuallyCorrected.nii.gz
│   └── ...
├── test
│   ├── TCGA-02-0003
│   │   ├── TCGA-02-0003_1997.06.08_t1.nii.gz
│   │   ├── TCGA-02-0003_1997.06.08_t1Gd.nii.gz
│   │   ├── TCGA-02-0003_1997.06.08_t2.nii.gz
│   │   ├── TCGA-02-0003_1997.06.08_flair.nii.gz
│   │   ├── TCGA-02-0003_1997.06.08_GlistrBoost.nii.gz
│   │   ├── TCGA-02-0003_1997.06.08_GlistrBoost_ManuallyCorrected.nii.gz
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码（不同病例就不分文件夹了，直接都放一起，用前缀区分病例，后缀区分slice）
├── T1_T1CE
│   ├── train
│   │   ├── TCGA-02-0006_0.png
│   │   ├── TCGA-02-0006_1.png
│   │   ├── ...
│   ├── val
│   │   ├── TCGA-08-0350_0.png
│   │   ├── TCGA-08-0350_1.png 
│   │   ├── ...
│   ├── test
│   │   ├── TCGA-02-0003_0.png
│   │   ├── TCGA-02-0003_1.png
│   │   ├── ...
```
### Breast
```markdown
包含三个公开乳腺癌数据集，已经按照比例做了合并（.mat格式版本）--> 后面会数据处理成nii版本
Breast_MR_train_val_test
├── train
│   │   ├── ISPY1_1001.mat
│   │   ├── ...
│   │   ├── UCSF-BR-01.mat
│   │   ├── ...
│   │   ├── TCGA-AO-A03M.mat
│   └── ...
├── val
│   │   ├── ISPY1_1001.mat
│   │   ├── ...
│   │   ├── UCSF-BR-01.mat
│   │   ├── ...
│   │   ├── TCGA-AO-A03M.mat
│   └── ...
├── test
│   │   ├── ISPY1_1001.mat
│   │   ├── ...
│   │   ├── UCSF-BR-01.mat
│   │   ├── ...
│   │   ├── TCGA-AO-A03M.mat
│   └── ...
└── survival_evaluation.csv
nii版本，待处理
breast_train_val_test
├── train
│   │   ├── 
│   │   ├── ...
│   │   ├── 
│   │   ├── ...
│   │   ├── 
│   └── ...
├── val
│   │   ├── 
│   │   ├── ...
│   │   ├── 
│   │   ├── ...
│   │   ├── 
│   └── ...
├── test
│   │   ├── 
│   │   ├── ...
│   │   ├── 
│   │   ├── ...
│   │   ├── 
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码
BRATS17
├── T1_T1CE
│   ├── test
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── train
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── val
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
```
## CT
### Uterus Ovary
```markdown
Uterus_Ovary_CT_train_val_test
├── train
│   ├── C3N-00866
│   │   ├── 03-05-2000-NA-CT UROGRAPHY CT ABDOME-51999
│   │   │   ├── C3N-00866_2000-03-05_CT.nii
│   │   │   ├── C3N-00866_2000-03-05_CTC.nii
│   │   ├── 03-06-2001-NA-CT RENAL MASS-CT ABDOM-36960
│   │   │   ├── C3N-00866_2001-03-06_CT.nii
│   │   │   ├── C3N-00866_2001-03-06_CTC.nii
│   ├── TCGA-09-2055
│   │   ├── 04-09-1998-NA-CT Abdo UnEn-38384
│   │   │   ├── TCGA-09-2055_1998-04-09_CT.nii
│   │   │   ├── TCGA-09-2055_1998-04-09_CTC.nii
│   └── ...
└── val
│   ├── TCGA-25-2404
│   │   ├── 10-24-1986-NA-Abdomen01AbdPelvisRoutine Adult-19915
│   │   │   ├── TCGA-25-2404_1986-10-24_CT.nii
│   │   │   ├── TCGA-25-2404_1986-10-24_CTC.nii
│   └── ...
└── test
│   ├── TCGA-61-2003
│   │   ├── 01-26-1998-NA-CT ABDOMEN WITH AND WI-80554
│   │   │   ├── TCGA-61-2003_1998-01-26_CT.nii
│   │   │   ├── TCGA-61-2003_1998-01-26_CTC.nii
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码
├── T1_T1CE
│   ├── train
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── val
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── test
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
```
### Adrenal
```markdown
Adrenal_CT_train_val_test
├── train
│   ├── Adrenal_Ki67_Seg_001
│   │   ├── 08-22-2000-NA-CT ABDOMEN-56266
│   │   │   ├── Adrenal_Ki67_Seg_001_2000-08-22_CT.nii
│   │   │   ├── Adrenal_Ki67_Seg_001_2000-08-22_CTC.nii
│   └── ...
└── val
│   ├── Adrenal_Ki67_Seg_037
│   │   ├── 02-09-2009-NA-CT Abdomen  Pelvis-63206
│   │   │   ├── Adrenal_Ki67_Seg_037_2009-02-09_CT.nii
│   │   │   ├── Adrenal_Ki67_Seg_037_2009-02-09_CTC.nii
│   └── ...
└── test
│   ├── Adrenal_Ki67_Seg_042
│   │   ├── 06-19-2011-NA-CT CHEST ABDOMEN PELVIS W WO CONTRAST-34088
│   │   │   ├── Adrenal_Ki67_Seg_042_2011-06-19_CT.nii
│   │   │   ├── Adrenal_Ki67_Seg_042_2011-06-19_CTC.nii
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码
├── T1_T1CE
│   ├── train
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── val
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── test
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
```
### Bladder Kidney
```markdown
Bladder_Kidney_CT_train_val_test
├── train
│   ├── C3L-00815
│   │   ├── 12-14-2008-NA-CT ABDOMEN AND PELVIS-75669
│   │   │   ├── C3L-00815_2008-12-14_CT.nii
│   │   │   ├── C3L-00815_2008-12-14_CTC.nii
│   ├── KiTS-00000
│   │   ├── 06-29-2003-NA-threephaseabdomen-41748
│   │   │   ├── KiTS-00000_2003-06-29_CT.nii
│   │   │   ├── KiTS-00000_2003-06-29_CTC.nii
│   └── ...
└── val
│   ├── TCGA-CJ-5678
│   │   ├── 10-09-1993-NA-CT CAP-92267
│   │   │   ├── TCGA-CJ-5678_1993-10-09_CT.nii
│   │   │   ├── TCGA-CJ-5678_1993-10-09_CTC.nii
│   └── ...
└── test
│   ├── C3N-02263
│   │   ├── 10-06-2000-NA-AbdomenJamaBrzuszna3FazyOpoznione Adult-40401
│   │   │   ├── C3N-02263_2000-10-06_CT.nii
│   │   │   ├── C3N-02263_2000-10-06_CTC.nii
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码
├── T1_T1CE
│   ├── train
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── val
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── test
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
```
### Lung
```markdown
Lung_CT_train_val_test
├── train
│   ├── CMB-LCA-MSB-00939
│   │   ├── 02-20-1960-NA-CTCAP-17196
│   │   │   ├── CMB-LCA-MSB-00939_1960-02-20_CT.nii
│   │   │   ├── CMB-LCA-MSB-00939_1960-02-20_CTC.nii
│   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── ...
│   ├── C3L-01000
│   │   ├── 06-09-2011-NA-MSKT organov grudnoy k-18628
│   │   │   ├── C3L-01000_2011-06-09_CT.nii
│   │   │   ├── C3L-01000_2011-06-09_CTC.nii
│   └── ...
└── val
│   ├── PD-1-Lung-00011
│   │   ├── 06-18-2007-NA-CAP W CONTRAST-89422
│   │   │   ├── PD-1-Lung-00011_2007-06-18_CT.nii
│   │   │   ├── PD-1-Lung-00011_2007-06-18_CTC.nii
│   └── ...
└── test
│   ├── Lung_Dx-A0104
│   │   ├── 04-23-2011-NA-ch.3d ao.cta-41811
│   │   │   ├── Lung_Dx-A0104_2011-04-23_CT.nii
│   │   │   ├── Lung_Dx-A0104_2011-04-23_CTC.nii
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码
├── T1_T1CE
│   ├── train
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── val
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── test
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
```
### Stomach Colon Liver Pancreas
```markdown
Stomach_Colon_Liver_Pancreas_CT_train_val_test
├── train
│   ├── C3L-00625
│   │   ├── 09-28-2003-NA-CT ABDOMEN NONENH  ENHANCEDAB-70477
│   │   │   ├── C3L-00625_2003-09-28_CT.nii
│   │   │   ├── C3L-00625_2003-09-28_CTC.nii
│   ├── HCC_001
│   │   ├── 04-21-2000-NA-CT ABDPEL WC-49771
│   │   │   ├── HCC_001_2000-04-21_CT.nii
│   │   │   ├── HCC_001_2000-04-21_CTC.nii
│   │   ├── 11-30-1999-NA-CT-CAP WWO CON-00377
│   │   │   ├── HCC_001_1999-11-30_CT.nii
│   │   │   ├── HCC_001_1999-11-30_CTC.nii
│   └── ...
└── val
│   ├── MSB-02169
│   │   ├── 03-26-1959-NA-CTCAP-75858
│   │   │   ├── MSB-02169_1959-03-26_CT.nii
│   │   │   ├── MSB-02169_1959-03-26_CTC.nii
│   └── ...
└── test
│   ├── TCGA-VQ-AA6J
│   │   ├── 05-30-2000-NA-AS-94537
│   │   │   ├── TCGA-VQ-AA6J_2000-05-30_CT.nii
│   │   │   ├── TCGA-VQ-AA6J_2000-05-30_CTC.nii
│   └── ...
└── survival_evaluation.csv
处理之后(将t1和t1ce左右拼成一个新的图png)——有对应代码
├── T1_T1CE
│   ├── train
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── val
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
│   └── test
│   │   ├── .png 
│   │   ├── .png 
│   │   ├── ...
```

# backbone

# baseline

### flow-matching
| paper | 会议/期刊 | dataset | 分类 | 器官 |
|---------|---------|---------|---------|---------|
|Benchmarking GANs, Diffusion Models, and Flow Matching for T1w-to-T2w MRI Translation.[[paper](https://arxiv.org/pdf/2507.14575)][[code](https://github.com/AndreaMoschetto/medical-I2I-benchmark)]|arxiv20250719||||
|LBM: Latent Bridge Matching for Fast Image-to-Image Translation.[[paper](https://arxiv.org/pdf/2503.07535)][[code](https://github.com/gojasper/LBM)]|arxiv20250310||||
|Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality.[[paper](https://arxiv.org/pdf/2503.00266)][[code](https://github.com/milad1378yz/MOTFM)]|arxiv20250301||||
|Boosting Latent Diffusion with Flow Matching.[[paper](https://arxiv.org/pdf/2312.07360)]|arxiv20241204||||
|Optimal Flow Matching: Learning Straight Trajectories in Just One Step.[[paper](https://arxiv.org/pdf/2403.13117)][[code](https://github.com/Jhomanik/Optimal-Flow-Matching)]|arxiv20241107||||
|Diversified Flow Matching with Translation Identifiability.[[paper](https://icml.cc/virtual/2025/poster/45403#:~:text=Sagar%20Shrestha%20%C2%B7%20Xiao%20Fu&text=DDM%20was%20proposed%20to%20resolve,constraints%20on%20the%20translation%20function.)]|ICML2025||||

### diffusion
| paper | 会议/期刊 | dataset | 分类 | 器官 |
|---------|---------|---------|---------|---------|
|Introducing 3D Representation for Medical Image Volume-to-Volume Translation via Score Fusion.[[paper](https://arxiv.org/pdf/2501.07430)]|20250206||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||
|Deterministic Medical Image Translation via High-fidelity Brownian Bridges (HiFi‑BBrg).[[paper](https://arxiv.org/pdf/2503.22531)]|arxiv20250308||||

### Flow ODE
| paper | 会议/期刊 | dataset | 分类 | 器官 |
|---------|---------|---------|---------|---------|
|Bi-modality medical images synthesis by a bi-directional discrete process matching method.[[paper](https://arxiv.org/pdf/2409.03977)]|20250703 ilcr2025投稿没中||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||


### transformer
| paper | 会议/期刊 | dataset | 分类 | 器官 |
|---------|---------|---------|---------|---------|
|One Model to Synthesize Them All: Multi-contrast Multi-scale Transformer for Missing Data Imputation.[[paper](https://www.semanticscholar.org/reader/04b67ac881b8285a55be147785c5e67eb544cc99)]|TMI2023||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||

### mamba
| paper | 会议/期刊 | dataset | 分类 | 器官 |
|---------|---------|---------|---------|---------|
|ABS‑Mamba: SAM2‑Driven Bidirectional Spiral Mamba Network for Medical Image Translation.[[paper](https://arxiv.org/pdf/2505.07687)]|||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||
|.[[paper]()][[code]()]|||||




