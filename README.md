# DP-LinkNet: A convolutional network for historical document image binarization

### Abstract

Document image binarization is an important pre-processing step in document analysis and archiving. The state-of-the-art models for document image binarization are variants of encoder-decoder architectures, such as FCN (*fully convolutional network*) and U-Net. Despite their success, they still suffer from three limitations: (1) reduced feature map resolution due to consecutive strided pooling or convolutions, (2) multiple scales of target objects, and (3) reduced localization accuracy due to the built-in invariance of *deep convolutional neural networks* (DCNNs). To overcome these three challenges, we propose an improved semantic segmentation model, referred to as **DP-LinkNet**, which adopts the D-LinkNet architecture as its backbone, with the proposed *hybrid dilated convolution* (HDC) and *spatial pyramid pooling* (SPP) modules between the encoder and the decoder. Extensive experiments are conducted on recent *document image binarization competition* (DIBCO) and *handwritten document image binarization competition* (H-DIBCO) benchmark datasets. Results show that our proposed DP-LinkNet outperforms other state-of-the-art techniques by a large margin. Our implementation and the pre-trained models are available at https://github.com/beargolden/DP-LinkNet.

### Proposed Architecture: DP-LinkNet

**The proposed method won the first place in ICDAR 2019 time-quality binarization competition on photographed document images taken by Motorola Z1 and Galaxy Note4 with flash off, and the second and third places on binarization of photographed document images taken by the same mobile devices with flash on, respectively[1].**

The proposed DP-LinkNet uses LinkNet[2] and D-LinkNet[3] with a pre-trained encoder as the backbone. It consists of four main parts: the **encoder** (part A), the ***hybrid dilated convolution*** (HDC) module (part B), the ***spatial pyramid pooling*** (SPP) module (part C), and the **decoder** (part D). The encoder extracts text stroke features with deep semantic information. The HDC module expands the receptive field size and aggregates multi-scale contextual features, while the SPP module encodes the output of the HDC with multi-kernel pooling. The combination of the HDC and SPP modules will produce enriched higher-level abstract feature maps. The decoder then maps the low-resolution feature map output from the central part back to the size of the input image for pixel-by-pixel classification. Although there are several subtle and important differences, what distinguishes our proposed DP-LinkNet from the two models mentioned above is that the LinkNet[2] contains only part A and D, while D-LinkNet[3] additionally contains part B.

![](https://github.com/beargolden/DP-LinkNet/blob/main/images/DP-LinkNet-architecture.png)

Fig. 1  The proposed DP-LinkNet architecture

![](https://github.com/beargolden/DP-LinkNet/blob/main/images/HDC-module.png)

Fig. 2  Hybrid Dilated Convolution (HDC) module

![](https://github.com/beargolden/DP-LinkNet/blob/main/images/SPP-module.png)

Fig. 3  Spatial Pyramid Pooling (SPP) module

### Experimental Results

#### Ablation Study

Table 1  Ablation study on LinkNet, D-LinkNet, and the proposed DP-LinkNet

| **Architecture** | **Params** | **FM(%)** | **pFM(%)** | **PSNR(dB)** |  **DRD** | **MPM(‰)** |
| ---------------: | ---------: | --------: | ---------: | -----------: | -------: | ---------: |
|        LinkNet34 | 21,642,401 |     91.65 |      92.70 |        16.83 |     1.84 |       0.52 |
|      D-LinkNet34 | 28,736,321 |     91.94 |      93.34 |        17.05 |     1.75 |       0.44 |
|     DP-LinkNet34 | 28,738,244 | **92.81** |  **94.30** |    **17.56** | **1.53** |   **0.34** |

#### More Segmentation Experiments

Table 2  Performance evaluation results of our proposed method against the TOP 3 winners in the DIBCO or H-DIBCO annual competition (best results highlighted in bold)

| Dataset      | Method   | FM(%) | pFM(%) | PSNR(dB) | NRM(%) |    DRD | MPM(‰) |
| ------------ | -------- | ----: | -----: | -------: | -----: | -----: | -----: |
| DIBCO 2009   | Rank 1   | 91.24 |        |    18.66 |   4.31 |        |   0.55 |
|              | Rank 2   | 90.06 |        |    18.23 |   4.75 |        |   0.89 |
|              | Rank 3   | 89.34 |        |    17.79 |   5.32 |        |   1.90 |
|              | Proposed | 96.39 |        |    22.16 |   1.30 |        |   0.10 |
| H-DIBCO 2010 | Rank 1   | 91.50 |  93.58 |    19.78 |   5.98 |        |   0.49 |
|              |          | 89.70 |  95.15 |    19.15 |   8.18 |        |   0.29 |
|              | Rank 2   | 91.78 |  94.43 |    19.67 |   4.77 |        |   1.33 |
|              | Rank 3   | 89.73 |  90.11 |    18.90 |   5.78 |        |   0.41 |
|              | Proposed | 96.19 |  97.06 |    22.95 |   1.29 |        |   0.10 |
| DIBCO 2011   | Rank 1   | 80.86 |        |    16.13 |        | 104.48 |  64.43 |
|              | Rank 2   | 85.20 |        |    17.16 |        |  15.66 |   9.07 |
|              | Rank 3   | 88.74 |        |    17.84 |        |   5.36 |   8.68 |
|              | Proposed | 96.27 |        |    22.23 |        |   1.01 |   0.11 |
| H-DIBCO 2012 | Rank 1   | 89.47 |  90.18 |    21.80 |        |   3.44 |        |
|              | Rank 2   | 92.85 |  93.34 |    20.57 |        |   2.66 |        |
|              | Rank 3   | 91.54 |  93.30 |    20.14 |        |   3.05 |        |
|              | Proposed | 96.90 |  97.62 |    23.99 |        |   0.84 |        |
| DIBCO 2013   | Rank 1   | 92.12 |  94.19 |    20.68 |        |   3.10 |        |
|              | Rank 2   | 92.70 |  93.19 |    21.29 |        |   3.18 |        |
|              | Rank 3   | 91.81 |  92.67 |    20.68 |        |   4.02 |        |
|              | Proposed | 97.15 |  97.77 |    24.09 |        |   0.78 |        |
| H-DIBCO 2014 | Rank 1   | 96.88 |  97.65 |    22.66 |        |   0.90 |        |
|              | Rank 2   | 96.63 |  97.46 |    22.40 |        |   1.00 |        |
|              | Rank 3   | 93.35 |  96.05 |    19.45 |        |   2.19 |        |
|              | Proposed | 97.47 |  98.05 |    23.46 |        |   0.66 |        |
| H-DIBCO 2016 | Rank 1   | 87.61 |  91.28 |    18.11 |        |   5.21 |        |
|              | Rank 2   | 88.72 |  91.84 |    18.45 |        |   3.86 |        |
|              | Rank 3   | 88.47 |  91.71 |    18.29 |        |   3.93 |        |
|              | Proposed | 96.29 |  97.03 |    23.04 |        |   1.05 |        |
| DIBCO 2017   | Rank 1   | 91.04 |  92.86 |    18.28 |        |   3.40 |        |
|              | Rank 2   | 89.67 |  91.03 |    17.58 |        |   4.35 |        |
|              | Rank 3   | 89.42 |  91.52 |    17.61 |        |   3.56 |        |
|              | Proposed | 95.52 |  96.46 |    20.83 |        |   1.31 |        |
| H-DIBCO 2018 | Rank 1   | 88.34 |  90.24 |    19.11 |        |   4.92 |        |
|              | Rank 2   | 73.45 |  75.94 |    14.62 |        |  26.24 |        |
|              | Rank 3   | 70.01 |  74.68 |    13.58 |        |  17.45 |        |
|              | Proposed | 95.99 |  96.85 |    22.71 |        |   1.09 |        |
| DIBCO 2019   | Rank 1   | 72.88 |  72.15 |    14.48 |        |  16.24 |        |
|              | Rank 2   | 71.63 |  70.78 |    14.15 |        |  16.71 |        |
|              | Rank 3   | 70.43 |  69.84 |    15.31 |        |   8.05 |        |
|              | Proposed | 87.67 |  87.56 |    18.63 |        |   2.38 |        |

Table 3  Performance evaluation results of our proposed method against the state-of-the-art techniques on the 10 DIBCO and H-DIBCO testing datasets (best results highlighted in bold)

| **Rank** | **Method**          | **FM(%)** | **pFM(%)** | **PSNR(dB)** |  **DRD** | **Score** |
| :------: | ------------------- | --------: | ---------: | -----------: | -------: | --------: |
|    1     | Proposed DP-LinkNet | **95.13** |  **95.80** |    **22.13** | **1.19** |  **1109** |
|    2     | Bezmaternykh’s UNet |     89.29 |      90.53 |        21.32 |     3.29 |      2341 |
|    3     | Vo’s DSN            |     88.04 |      90.81 |        18.94 |     4.47 |      2946 |
|    4     | Peng’s woConvCRF    |     86.09 |      87.40 |        18.99 |     4.83 |      3216 |
|    5     | Zhao’s cGAN         |     87.45 |      88.87 |        18.81 |     5.56 |      3531 |
|    6     | Wolf’s              |     78.67 |      82.89 |        16.28 |     7.80 |      4851 |
|    7     | Sauvola’s           |     79.12 |      82.95 |        16.07 |     8.61 |      5281 |
|    8     | Bhowmik’s GiB       |     83.16 |      87.72 |        16.72 |     8.82 |      5316 |
|    9     | Gallego’s SAE       |     79.22 |      81.12 |        16.09 |     9.75 |      5910 |
|    10    | Jia’s SSP           |     85.05 |      87.24 |        17.91 |     9.74 |      6219 |
|    11    | Otsu’s              |     74.22 |      76.99 |        14.54 |    30.36 |     17116 |
|    12    | Niblack’s           |     41.12 |      41.57 |         6.67 |    91.23 |     50335 |

References

[1] R. Dueire Lins, E. Kavallieratou, E. B. Smith, R. B. Bernardino, D. M. D. Jesus, "ICDAR 2019 time-quality binarization competition," in Proceedings of the *15th IAPR International Conference on Document Analysis and Recognition (ICDAR 2019)*, Sydney, NSW, AUSTRALIA, 2019, pp. 1539-1546. doi: 10.1109/icdar.2019.00248

[2] A. Chaurasia, E. Culurciello, "LinkNet: Exploiting encoder representations for efficient semantic segmentation," in Proceedings of the *2017 IEEE Visual Communications and Image Processing (VCIP 2017)*, St. Petersburg, FL, USA, 2017, pp. 1-4. doi: 10.1109/vcip.2017.8305148

[3] L. Zhou, C. Zhang, M. Wu, "D-LinkNet: LinkNet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction," in Proceedings of the *31st Meeting of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPR 2018)*, Salt Lake City, UT, USA, 2018, pp. 192-196. doi: 10.1109/cvprw.2018.00034
