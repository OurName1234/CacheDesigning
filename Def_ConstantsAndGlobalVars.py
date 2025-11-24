#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:12:37 2024

@author: jfs
"""

ValidationCorporaFileName = 'Def_ValidationCorporaSizes.py'
TestCorporaFileName = 'Def_TestingCorporaSizes.py'
VocaburaryBDFileName = 'Def_Vocabulary.py'
ValidationKThresholdsBdFileName = 'Def_KThresolds.py'
ValidationCorporaKThresholdsBdFileName = 'Def_CorporaKThresolds.py'
SplineCoefsBDFileName = 'Def_SplineCoefs.py'

HIGHESTERROR= 9999999999999999
KSETSIZE = 10
KMIN = 1 #just in case
KMAX = 500 #just incase

MAJORMONONONYK = 5000

WKDEV = 0.06
CONFIDLEVEL = 0.95

MINWKDEV = 0.02
MAXWKDEV = 0.07
STEPWKDEV = 0.01
MINCONFIDLEVEL = 0.95
MAXCONFIDLEVEL = 0.975
STEPCONFIDLEVEL = 0.025

GREATESTTWOPOWER = 12
GREATESTKBEFOREJUMP = 16
MINAPPLICATIONK = 15
ENOUGH_V_ERROR = 0.01
POSFORSCALE = 6

INITCACHEVALUE = 10000
DELTACACHE = 40000
CACHESTEPSNUMBER = 30
INITGLOBHITRATIO = 0.05
GLOBALCACHESIZEDEVIATION = 0.00001
GLOBHITRATIOLIMIT = 10