#!/usr/bin/env python
# coding: utf-8

# In[19]:
import numpy as np
from src.data.generators import create_generators

for i in range(5):
    kek = create_generators('Hips', f"s2s_fold{i}")


# In[1]:

