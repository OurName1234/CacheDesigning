#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:08:05 2024

@author: jfs
"""

import numpy as np
from scipy.stats import poisson
from scipy.special import gammainc
import math
from Def_ConstantsAndGlobalVars import   MAJORMONONONYK
from scipy.stats import skellam
from scipy.stats import norm



def locConfidence(Lang,n,C,Distk,WkDev,ConfidLevel):
    k=1;
    while(k < MAJORMONONONYK and Confidence(Distk[(Lang,n,C,k)],Distk[(Lang,n,C,k+1)],WkDev,ConfidLevel)):
        k=k+1
    return k-1

def Confidence(Lamb1,Lamb2,ErrW,ConfidenceLevel):
   #print(ErrW,ConfidenceLevel)
   z_score = norm.ppf(1 - (1 - ConfidenceLevel) / 2)
   return ErrW >= z_score * np.sqrt(Lamb1 + Lamb2) /(Lamb1 - Lamb2)
  


def z_score_for_confidence_level(confidence_level):
    # Get the z-score for the one-tailed test
    return norm.ppf(confidence_level)
    
    