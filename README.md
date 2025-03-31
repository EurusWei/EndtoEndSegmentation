# EndtoEndSegmentation
Data and code sharing for examples in the paper:<br>
***"End-to-end Automated Segmentation Framework for Four-Dimensional Scanning Transmission Electron Microscopy Data"*** <br>
For each dataset, there is a Jupyter Notebook demonstrating each step of the segmentation framework. Details of all functions can be found in the `.utils` file.

The data can be downloaded in:<br> 
https://www.dropbox.com/scl/fo/zw9qpyfiiwdabn87rtsje/AJ_boGr-4295HQ-O1TD_dyc?rlkey=fnzk8wxzlu33q23w1vy7ogrei&dl=0 <br>
Note that the downloaded data needs to be uncompressed.<br>

The version of python is 3.8.0 and py4DSTEM is 0.12.6. The documentation of py4DSTEM can be found in https://github.com/py4dstem/py4DSTEM/releases/tag/v0.12.6. Most of the documentation is generated from the function documentation strings, which Jupyter or most editors can show you, or you can get it from Python with the question mark, like ```py4DSTEM.process.utils.plot?```. 

We included files applying our algorithm using 0.14.4 version, in the folder ```py4dstem_14version.ipynb```. We also added the file of applying the segmentation algorithm in the py4DSTEM package in the file ```example_1_py4dstemClassifier.ipynb```.


