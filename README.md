# Comprehensive Sequence Rearrangement and Local Multi-scale Cross-region Attention for Radiology Report Generation 
#### *by: Yan Deng* , Wenfeng Zhang , Qibing Qin, Wei Hu, Jianming Hu, Dengwei Yan

## About
![CSR-LMCA Architecture](figure/lmca.png)
We propose a radiology report generation model based on comprehensive  sequence rearrangement with multi-scale cross-region Attention (CSR-LMCA). We enhance the model's focus on disease-related key areas by employing a Saliency-guided Discriminative Attention Mapping (SDAM), significantly improving the model's ability to identify lesion regions, suppressing background noise interference. Additionally, we introduce Sequence Reordering Mamba (SR-Mamba), and processing the rearranged long sequences through mamba allows the model to efficiently extract more discriminative features. We further propose a Local Multi-Scale Cross-Region Attention (LMCA) mechanism, which first models the attention relationships within local regions and then performs cross-region information fusion. 

## Datasets
We used two public datasets (IU X-Ray and MIMIC-CXR) in this study :
- For [Iu X-Ray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) ,  you can download the dataset and then put the files in data/iu_xray.
- For [MIMIC-CXR](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing),  you can download the dataset and then put the files in data/MIMIC-CXR.

## How to Use

1. **Prepare the Data**: Download and place the required datasets into the `dataset` folder under their corresponding subdirectories.

2. **Set Up the Environment**: Ensure all necessary dependencies are installed as required by the project.

3. **Training the Model**:
   - Navigate to the root directory of the project.
   - Run the training by executing the following command:
     ```bash
     python main_train.py
     ```
   - Run the testing by executing the following command:
     ```bash
     python main_test.py
     ```

## Python Environment
Please install the following core dependencies before running the project:
- torch==2.1.1+cu118  
- torchvision==0.16.1+cu118  
- torchaudio==2.1.1+cu118  
- transformers==4.41.2  
- timm==1.0.9  
- tokenizers==0.19.1  
- numpy==1.24.4  
- opencv-python==4.10.0.84  
- tqdm==4.66.5  
- spacy==2.3.9  
- mamba_ssm==1.1.3

