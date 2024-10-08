# AG-Net
 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attend-and-guide-ag-net-a-keypoints-driven/image-classification-on-caltech-256)](https://paperswithcode.com/sota/image-classification-on-caltech-256?p=attend-and-guide-ag-net-a-keypoints-driven)

This is an implementation (the first public implementation as far as I know) of AG-Net as described in the paper "Attend and Guide (AG-Net): A Keypoints-driven Attention-based Deep Network for Image Recognition" by  Asish Bera, Zachary Wharton, Yonghuai Liu, Nik Bessis, and Ardhendu Behera. I include the weights for the model which achieves 98.3% accuracy on the test data of the Caltech-256 dataset. This is the global #1 score for the caltech-256 dataset on paperswithcode.com.

I include the iPython Notebook where I derived the model to "show my work", but I also include a few python files which contain the combined code from the notebook. The files and functions are the following: utils.py, which contains the data fetching and augmentations, models.py, which contains the torch modules, train.py, which trains the model and optionally saves it, and test.py, which tests the model.

Note that I coded all of this on my own with the exception of the "Intra Self-Attention" module which I obtained from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py (also note that there it's called Self-Attention (Self_Attn module)). 
