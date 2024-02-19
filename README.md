# nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
This is the code of the semester project for the course "Machine learning for 3D Geometry" of the Technical University of Munich.

The main branch contains the base nnU-Net framework. We made 5 modifications and saved each of them in their respective branch.

| Branch        | Description           |
| :-------------: |------| 
| Main      | Base nnU-Net framework without any modification. | 
| attention_nnunet      | Transformed the Vanilla U-Net into an Attention U-Net while mantaining the dynamic network creation. | 
| stacked_dilated_convolutions      | Added in each level of the Vanilla U-Net stacked dilated convolutions while mantaining the dynamic network creation.      |  
| decreasingly_dilated_convolutions      | Added in each encoder level of the Vanilla U-Net decreasingly dilated convolutions while mantaining the dynamic network creation.      |  
| nnunet_3plus | Transformed the Vanilla U-Net into an U-Net 3+ while mantaining the dynamic network creation.      |  
| model_quantization |Techniques to reduce model footprint.|  

Please refer to the original [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) repository for the installation guide.

