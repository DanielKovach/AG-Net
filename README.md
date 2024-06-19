# AG-Net
This is an implementation (the first public implementation as far as I know) of AG-Net as described in the paper "Attend and Guide (AG-Net): A Keypoints-driven Attention-based Deep Network for Image Recognition" by  Asish Bera, Zachary Wharton, Yonghuai Liu, Nik Bessis, and Ardhendu Behera. I include the weights for the model which achieves 98.3% accuracy on the test data. 

I include the iPython Notebook where I derived the model to "show my work", but I also include a python file AG-net.py which contains the combined code from the notebook. AG-net.py has since been split into utils.py, which contains the data fetching and augmentations, models.py, which contains the torch modules, train.py, which trains the model and optionally saves it, and test.py, which tests the model.

Note I coded all of this on my own with the exception of the "Intra Self-Attention" module which I obtained from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py (note that there it's called Self-Attention (Self_Attn module)). 
