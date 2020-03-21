# AlexNet

## AlexNet architecture
This is a tensorflow implement of an AlexNet architecture, the same as the [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) but quite diffrent in the weight numbers from the primitive paper.
## Inference
It can be inferenced by running `run_demo.py` and can generate the `ckpt`
files, and can be viewed by either `netron` and `tensorboard`.
## Training  
There is no scripts discribe how to train yet. 
## How to trace the time the model use 
There is a [issue](https://github.com/tensorflow/tensorflow/issues/1824) in tensorflow repo describes how to use 'tf.RunOptions.FULL_TRACE'.  
However when you meet 'can't find libcupti.so' error, you could add the environmet variable 'LD_LIBRARY_PATH' in your executing terminal environmet, i.e. `export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"` that will fix this issue.  
If you wanna once and for all, then edit the `.bashrc` and augment this:
`export LD_LIBRARY_PATH="/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH"`    
that's it! "-**-"
