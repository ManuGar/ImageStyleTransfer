#!/usr/bin/env python
# coding: utf-8

# # Image style transfer generator
# 
# In this notebook we are going to transform the style of a given dataset. To do this, we need two different datsets, the dataset that we want to change style and another dataset to know which style we want to obtain.
# 
# To make this change of styles, we have several options available. The user will have to choose the method they want to use to carry out this style change and the notebook takes care of the necessary operations.
# 
# 
# The available algorithms are:
# - upit
# - nst (Neural style transfer)
# - strotss
# - dualGAN
# - forkGAN
# - ganilla
# - CUT
# - fastCUT
# - dia (Deep Image Analogy)

# In[1]:


algorithm = 'dia'
dataset_name = 'messidor'
output_path = 'messidor_dia'


# In[2]:


import os
if (not os.path.exists(output_path)):
    os.makedirs(output_path)


import sys
ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")

# In[ ]:
import cv2
for k, v in os.environ.items():
     if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

import algorithms
algorithms.generate_images(algorithm, dataset_name, output_path)


# In[ ]:




