#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:31:07 2024

@author: jfs
"""



import time
import math
import random
import numpy as np
from mpmath import mp
from scipy.interpolate import splrep,splev
from Def_DistByLangNandK import ValidationAndTestingDistkBD
from Def_TestingCorporaSizes import TestingCorporaSizesList
from Def_ValidationCorporaSizes import ValidationCorporaSizesList
from Def_monotony import locConfidence
from Def_ConstantsAndGlobalVars import GREATESTTWOPOWER, POSFORSCALE, GREATESTKBEFOREJUMP, INITCACHEVALUE, DELTACACHE, CACHESTEPSNUMBER, INITGLOBHITRATIO, GLOBHITRATIOLIMIT, GLOBALCACHESIZEDEVIATION, MAJORMONONONYK
from Def_Vocabulary import VocBD
from Def_KThresolds import KThresholdsBD
from scipy.interpolate import SmoothBivariateSpline
import pickle
from collections import defaultdict







#for LocalMaxs WithOut BoolFilter

def LocalMaxRequiredCacheWithOutBloomF(Lang,MaxNgramSize,CorpusSize,GlobGitRatio,DicDicGHofK={},ValidCorporaList={},Voc={}, verb=False):
    DicDicGHofK={}
    for ng in range(1,MaxNgramSize+1):
        DicDicGHofK[ng]=LoadSoftGH(Lang,ng,"DepParms")
    return LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobGitRatio,ValidCorporaList={},Voc={},DicDicGHofK=DicDicGHofK,Kini=1, verb=verb)

def EvalLocalMaxRealAndPercentCacheForHitRatioVsCorpusSizeWithOutBloomF(Lang,MaxNgramSize,GlobHitRatio,InitCorpusSize,EndCorpusSize,Nsteps,verb=False):
    return EvalLocalMaxRealAndPercentCacheForHitRatioVsCorpusSize(Lang,MaxNgramSize,GlobHitRatio,InitCorpusSize,EndCorpusSize,Nsteps,Kini=1,verb=verb)

def EvalLocalMaxRealAndPercentCacheForCorpusSizeVsHitRatioWithOutBloomF(Lang,MaxNgramSize,CorpusSize,InitHR,Nsteps,verb=False):
    return EvalLocalMaxRealAndPercentCacheForCorpusSizeVsHitRatio(Lang,MaxNgramSize,CorpusSize,InitHR,Nsteps,Kini=1,verb=verb)

def LocalMaxFindBestCacheVsHitRatioPointForCorpusSizeWithOutBloomF(Lang,MaxNgramSize,CorpusSize,InitHR,DeltaHR,CacheImportance=1,verb=False):
    return LocalMaxFindBestCacheVsHitRatioPointForCorpusSize(Lang,MaxNgramSize,CorpusSize,InitHR,DeltaHR,CacheImportance=1,Kini=1,verb=verb)



#for LocalMaxs With BoolFilter

def LocalMaxRequiredCacheWithBloomF(Lang,MaxNgramSize,CorpusSize,GlobGitRatio,DicDicGHofK={},ValidCorporaList={},Voc={}, verb=False):
    DicDicGHofK={}
    for ng in range(1,MaxNgramSize+1):
        DicDicGHofK[ng]=LoadSoftGH(Lang,ng,"DepParms")
    return LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobGitRatio,ValidCorporaList={},Voc={},DicDicGHofK=DicDicGHofK,Kini=2, verb=verb)

def EvalLocalMaxRealAndPercentCacheForHitRatioVsCorpusSizeWithBloomF(Lang,MaxNgramSize,GlobHitRatio,InitCorpusSize,EndCorpusSize,Nsteps,verb=False):
    return EvalLocalMaxRealAndPercentCacheForHitRatioVsCorpusSize(Lang,MaxNgramSize,GlobHitRatio,InitCorpusSize,EndCorpusSize,Nsteps,Kini=2,verb=verb)

def EvalLocalMaxRealAndPercentCacheForCorpusSizeVsHitRatioWithBloomF(Lang,MaxNgramSize,CorpusSize,InitHR,Nsteps,verb=False):
    return EvalLocalMaxRealAndPercentCacheForCorpusSizeVsHitRatio(Lang,MaxNgramSize,CorpusSize,InitHR,Nsteps,Kini=2,verb=verb)

def LocalMaxFindBestCacheVsHitRatioPointForCorpusSizeWithBloomF(Lang,MaxNgramSize,CorpusSize,InitHR,DeltaHR,CacheImportance=1,verb=False):
    return LocalMaxFindBestCacheVsHitRatioPointForCorpusSize(Lang,MaxNgramSize,CorpusSize,InitHR,DeltaHR,CacheImportance=1,Kini=2,verb=verb)


# Function to load splines from the file
def LoadSoftGH(Lang, NgramSize,Model):
    if(Model == "DepParms"):
        SplineFile='Def_smooth_spline_results.pkl'
    else:
        SplineFile='Def_smooth_spline_results_const.pkl'
    try:
        with open(SplineFile, 'rb') as f:
            spline_results = pickle.load(f)
        if (Lang, NgramSize) in spline_results:
            return spline_results[(Lang, NgramSize)]
        else:
            raise KeyError(f"Splines for Lang={Lang}, NgramSize={NgramSize} not found.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {SplineFile} does not exist.")


#Retuurns Global cache needed and its percentage for FullCache capacity for a corpus size and for achieving a hit ratio. If Kini=2 it is for Bloofilter case; If Kini=1 it is for no Bloofilter case
def LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobGitRatio,ValidCorporaList={},Voc={},DicDicGHofK={},Kini=2, verb=False):
    if(ValidCorporaList == {}):
        ValidationCorporaSizesList(ValidCorporaList);
    if(Voc=={}):
        VocBD(Voc)
    DicForDks={i:(Dk(Lang,i,CorpusSize,1,Voc,ValidCorporaList),Dk(Lang,i,CorpusSize,2,Voc,ValidCorporaList)) for i in range(1,MaxNgramSize+1)}
    ActHitRefCache={i: (2 * sum(D1 for j,(D1,_) in  DicForDks.items() if j > i), DicForDks[i][Kini-1],Kini) for i in  DicForDks if i != MaxNgramSize}
    InitRefs={i: Refs for i,(Refs,_,_) in ActHitRefCache.items()}
    TotRefs= sum(Href for (Href,_,_) in ActHitRefCache.values())
    TotD1= sum(D1 for i,(D1,_) in DicForDks.items() if i< MaxNgramSize )
    TotD2= sum(D2 for i, (_,D2) in DicForDks.items() if i< MaxNgramSize )
    if(DicDicGHofK=={}):
        for ng in range(1,MaxNgramSize+1):
            DicDicGHofK[ng]=LoadSoftGH(Lang,ng,"DepParms")
    sai=False;k=Kini
    while(not sai):
        NgramSize=MaxNgramSize-1
        while(not sai and NgramSize >=1):
            (D1,D2)=DicForDks[NgramSize]
            (HitRefs,CacheSize)=LocalMaxMinusFirstKHitRef(Lang,NgramSize,CorpusSize,k,ActHitRefCache,InitRefs[NgramSize],D1,DicDicGHofK[NgramSize],ValidCorporaList=ValidCorporaList,Voc=Voc)
            ActHitRefCache[NgramSize]=(HitRefs,CacheSize,k)
            (ActHit,ActHitRefs)=AppZeroActHitRef(ActHitRefCache,TotRefs)
            if (ActHit > GlobGitRatio):
                ActHitRefCache[NgramSize]=(HitRefs,CacheSize,k+1)
                NgramSize -= 1
            else:
                CorrectHitRefs  = GlobGitRatio * TotRefs
                NewCache= CacheSize + (CorrectHitRefs - ActHitRefs) * D1 / InitRefs[NgramSize]
                ActHitRefCache[NgramSize]=(CorrectHitRefs - ActHitRefs,NewCache,k)
                sai=True
        k+=1
    GlobCache=sum(CacheSiz for (_,CacheSiz,_) in ActHitRefCache.values())
    if(verb):
        print("Local Max Global Cache Size for",GlobGitRatio,"HitRatio and ",CorpusSize,"Corpus Size:", GlobCache, ". Precent of the Total distinct n-grams:",GlobCache/TotD1)
    return  (GlobCache,ActHitRefCache,GlobCache/TotD2,GlobCache/TotD1)



#
def FirstDkRatioForCorpora(Lang,NgramSize,k,CorpusSizeLimInf=3e7,CorpusSizeLimSup=3e12):
    ValidCorporaList={};  Distk={}; TestCorporaDic={}; Voc={}; SplineCoefs={}; 
    ValidationCorporaSizesList(ValidCorporaList); 
    ValidCorpora=ValidCorporaList[(Lang,NgramSize)]
    ValidCorpora=sorted(ValidCorpora)
    GHCorpusForScale=ValidCorpora[POSFORSCALE]
    GreatestTrainCorpus=ValidCorpora[-2]
    DicGHofK=LoadSoftGH(Lang, NgramSize,"DepParms")
    ValidationAndTestingDistkBD(Distk)
    TestingCorporaSizesList(TestCorporaDic); 
    TestCorpora=TestCorporaDic[(Lang,NgramSize)]
    VocBD(Voc)
    V=Voc[(Lang,NgramSize)]
    RepRatioEmpFileName="FirstDkRatioForCorpora_Emp_"+str(k)+"_"+str(Lang)+"_"+str(NgramSize)
    with open(RepRatioEmpFileName,'w') as file:
        GlobErr=[];
        for C in TestCorpora:
            PredD1=PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,C,1,DicGHofK,V)
            PredDk=PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,C,k,DicGHofK,V)
            PredRate=(PredD1 -PredDk )/PredD1
            EmpD1=Distk[(Lang,NgramSize,C,1)]
            EmpDk=Distk[(Lang,NgramSize,C,k)]
            EmpRate=(EmpD1 -EmpDk )/EmpD1
            st=str(np.log(C))+"\t"+str(EmpRate)
            file.write(st+"\n")
            RelErr=abs((PredRate - EmpRate )/EmpRate)
            print("Prediction for the ratio of number of first",k-1,"distinct n_grams over D for Corpus ",C,": ",PredRate,". Empirical: ",EmpRate," . (Relative Error: ", RelErr)
            GlobErr.append(RelErr)
        GlobErr=np.array(GlobErr)
        file.close()
        print("Avegare Relative Errors: ",np.mean(GlobErr))
    RepRatioPredFileName="FirstDkRatioForCorpora_Pred_"+str(k)+"_"+str(Lang)+"_"+str(NgramSize)
    with open(RepRatioPredFileName,'w') as file:
        base=1.5
        for C in [(20e6)* base**p for p in range(1,40) if (20e6)* base**p >= CorpusSizeLimInf and (20e6)* base**p <= CorpusSizeLimSup ]:
            PredD1=PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,C,1,DicGHofK,V)
            PredDk=PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,C,k,DicGHofK,V)
            PredRate=(PredD1 -PredDk )/PredD1
            st=str(np.log(C))+"\t"+str(PredRate)
            file.write(st+"\n")
        file.close()







#
def LocalMaxMinusFirstKHitRef(Lang,NgramSize,CorpusSize,k,ActHitRefCache,NgInitRef,D1,DicGHofK,ValidCorporaList={},Voc={}):
    if(ValidCorporaList == {}):
        ValidationCorporaSizesList(ValidCorporaList);
    if(Voc=={}):
        VocBD(Voc)
    ValidCorpora=ValidCorporaList[(Lang,NgramSize)]
    ValidCorpora=sorted(ValidCorpora)
    GHCorpusForScale=ValidCorpora[POSFORSCALE]
    GreatestTrainCorpus=ValidCorpora[-2]
    V=Voc[(Lang,NgramSize)]
    (HitRefs,_,_)=ActHitRefCache[NgramSize]
    dk=PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,CorpusSize,k,DicGHofK,V)
    dkp1=PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,CorpusSize,k+1,DicGHofK,V)
    HitRefs= HitRefs -  NgInitRef * (dk - dkp1) /D1
    CacheSize=dkp1
    return (HitRefs,CacheSize)
    

def AppZeroActHitRef(ActHitRefCache,TotRefs):
    Soma=0
    for NgramSize in ActHitRefCache:
        (HitRef,_,_)=ActHitRefCache[NgramSize]; Soma+= HitRef
    return (Soma/TotRefs,Soma)


def EvalLocalMaxRealAndPercentCacheForHitRatioVsCorpusSize(Lang,MaxNgramSize,GlobHitRatio,InitCorpusSize,EndCorpusSize,Nsteps,Kini=2,verb=False):
    step=np.log(EndCorpusSize)/Nsteps
    ValidCorporaList={}; Distk={}; Voc={}; VocBD(Voc)
    ValidationCorporaSizesList(ValidCorporaList); ValidationAndTestingDistkBD(Distk)
    GlobErr=[]
    TestCorporaDic={}
    TestingCorporaSizesList(TestCorporaDic);
    TestCorpora=TestCorporaDic[(Lang,1)]
    DicDicGHofK={}
    for ng in range(1,MaxNgramSize+1):
        DicDicGHofK[ng]=LoadSoftGH(Lang,ng,"DepParms")
    LocMaxCacheForHitRatioVsCorpusSizeFileName="LocMaxCacheForHitRatioVsCorpusSize_"+str(Lang)+"_"+str(GlobHitRatio)+"_"+str(Kini)
    EmpLocMaxCacheForHitRatioVsCorpusSizeFileName="EmpLocMaxCacheForHitRatioVsCorpusSize_"+str(Lang)+"_"+str(GlobHitRatio)+"_"+str(Kini)
    LocMaxPercentCacheForHitRatioVsCorpusSizeFileName="LocMaxPercentCacheForHitRatioVsCorpusSize_"+str(Lang)+"_"+str(GlobHitRatio)+"_"+str(Kini)
    EmpLocMaxPercentCacheForHitRatioVsCorpusSizeFileName="EmpLocMaxPercentCacheForHitRatioVsCorpusSize_"+str(Lang)+"_"+str(GlobHitRatio)+"_"+str(Kini)
    with open(LocMaxCacheForHitRatioVsCorpusSizeFileName,'w') as file1, open(EmpLocMaxPercentCacheForHitRatioVsCorpusSizeFileName,'w') as file2, open(LocMaxPercentCacheForHitRatioVsCorpusSizeFileName,'w') as file3, open(EmpLocMaxCacheForHitRatioVsCorpusSizeFileName,'w') as file4:
        for i in range(1,Nsteps +1):
            CorpusSize=InitCorpusSize + np.exp(i*step)
            (GlobCacheSize,_,_,GlobCacheSizePercent)= LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,ValidCorporaList=ValidCorporaList,Voc=Voc,DicDicGHofK=DicDicGHofK,Kini=Kini, verb=False)
            st1=str(CorpusSize)+"\t"+str(GlobCacheSize); st3=str(CorpusSize)+"\t"+str(GlobCacheSizePercent);
            file1.write(st1+"\n"); file3.write(st3+"\n");
        file1.close(); file3.close();
        for CorpusSize in TestCorpora:
            (GlobCacheSize,_,_,_)=LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,ValidCorporaList=ValidCorporaList,Voc=Voc,DicDicGHofK=DicDicGHofK,Kini=Kini, verb=False)
            (EmpGlobCacheSize,_,_,EmpGlobCacheSizePercent) = EmpLocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,TestCorporaDic, Distk,Kini=Kini)
            ErrRel=abs(GlobCacheSize - EmpGlobCacheSize)/EmpGlobCacheSize
            if(verb):
                print("Relative Error of LocalMax cache size for HitRatio",GlobHitRatio,"and corpus size",CorpusSize,":",ErrRel)
            GlobErr.append(ErrRel)
            st2=str(CorpusSize)+"\t"+str(EmpGlobCacheSizePercent)
            file2.write(st2+"\n");
            st4= str(CorpusSize)+"\t"+str(EmpGlobCacheSize); file4.write(st4+"\n");
        file2.close(); file4.close() 
    if(verb):
        print("Avg Error",np.mean(GlobErr))
        

def EmpLocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,TestCorporaDic, Distk,Kini=2):
    Idx=TestCorporaDic[(Lang,1)].index(CorpusSize)
    EmpDicForDks={i:(Distk[(Lang,i,TestCorporaDic[(Lang,i)][Idx],1)],Distk[(Lang,i,TestCorporaDic[(Lang,i)][Idx],2)]  ) for i in range(1,MaxNgramSize+1) }
    EmpActHitRefCache={i: (2 * sum(D1 for j,(D1,_) in  EmpDicForDks.items() if j > i), EmpDicForDks[i][Kini-1],Kini) for i in  EmpDicForDks if i != MaxNgramSize}
    EmpInitRefs={i: Refs for i,(Refs,_,_) in EmpActHitRefCache.items()}
    EmpTotRefs= sum(EmpHref for (EmpHref,_,_) in EmpActHitRefCache.values())
    EmpTotD1= sum(D1 for i,(D1,_) in EmpDicForDks.items() if i< MaxNgramSize )
    EmpTotD2= sum(D2 for i, (_,D2) in EmpDicForDks.items() if i< MaxNgramSize )
    sai=False;k=Kini
    while(not sai):
        NgramSize=MaxNgramSize - 1
        while(not sai and NgramSize >=1):
            (D1,D2)=EmpDicForDks[NgramSize]
            (EmpHitRefs,EmpCacheSize)=EmpLocalMaxMinusFirstKHitRef(Lang,NgramSize,TestCorporaDic[(Lang,NgramSize)][Idx],k,EmpActHitRefCache,EmpInitRefs[NgramSize],D1,Distk)
            EmpActHitRefCache[NgramSize]=(EmpHitRefs,EmpCacheSize,k)
            (EmpActHit,EmpActHitRefs)=EmpAppZeroActHitRef(EmpActHitRefCache,EmpTotRefs)
            if (EmpActHit > GlobHitRatio):
                EmpActHitRefCache[NgramSize]=(EmpHitRefs,EmpCacheSize,k+1)
                NgramSize -= 1
            else:
                CorrectHitRefs  = GlobHitRatio * EmpTotRefs
                EmpNewCache= EmpCacheSize +(CorrectHitRefs -  EmpActHitRefs  ) * D1 / EmpInitRefs[NgramSize]
                EmpActHitRefCache[NgramSize]=(CorrectHitRefs,EmpNewCache,k)
                sai=True
        k+=1
    EmpGlobCache=sum(EmpCacheSiz for (_,EmpCacheSiz,_) in EmpActHitRefCache.values())
    return  (EmpGlobCache,EmpActHitRefCache,EmpGlobCache/EmpTotD2,EmpGlobCache/EmpTotD1)
    

def EmpLocalMaxMinusFirstKHitRef(Lang,NgramSize,CorpusSize,k,EmpActHitRefCache,EmpNgInitRefNg,D1,Distk):
    (EmpHitRefs,_,_)=EmpActHitRefCache[NgramSize]
    EmpDk=Distk[(Lang,NgramSize,CorpusSize,k)]
    EmpDkp1=Distk[(Lang,NgramSize,CorpusSize,k+1)]
    EmpHitRefs= EmpHitRefs - EmpNgInitRefNg * (EmpDk -  EmpDkp1)/D1
    EmpCacheSize=EmpDkp1
    return (EmpHitRefs,EmpCacheSize)


def EmpAppZeroActHitRef(EmpActHitRefCache,EmpTotRefs):
    EmpSoma=0
    for NgramSize in EmpActHitRefCache:
        (EmpHitRef,_,_)=EmpActHitRefCache[NgramSize]; EmpSoma+= EmpHitRef
    return (EmpSoma / EmpTotRefs,EmpSoma) 
 


#OK 
def EvalLocalMaxRealAndPercentCacheForCorpusSizeVsHitRatio(Lang,MaxNgramSize,CorpusSize,InitHR,Nsteps,Kini=2,verb=False):
    jump=0.4 * 10/Nsteps
    ValidCorporaList={}; Distk={}; Voc={}; VocBD(Voc)
    ValidationCorporaSizesList(ValidCorporaList); ValidationAndTestingDistkBD(Distk)
    GlobErr=[]
    TestCorporaDic={}
    TestingCorporaSizesList(TestCorporaDic);
    TestCorpora=TestCorporaDic[(Lang,1)]
    DicDicGHofK={}
    for ng in range(1,MaxNgramSize+1):
        DicDicGHofK[ng]=LoadSoftGH(Lang,ng,"DepParms")
    LocalMaxCacheForCorpusSizeVsHitRatioFileName="LocalMaxCacheForCorpusSizeVsHitRatio_"+str(Lang)+"_"+str(CorpusSize)+"_"+str(Kini)
    EmpLocalMaxCacheForCorpusSizeVsHitRatioFileName="EmpLocalMaxCacheForCorpusSizeVsHitRatio_"+str(Lang)+"_"+str(CorpusSize)+"_"+str(Kini)
    LocalMaxPercentCacheForCorpusSizeVsHitRatioFileName="LocalMaxPercentCacheForCorpusSizeVsHitRatio_"+str(Lang)+"_"+str(CorpusSize)+"_"+str(Kini)
    EmpLocalMaxPercentCacheForCorpusSizeVsHitRatioFileName="EmpLocalMaxPercentCacheForCorpusSizeVsHitRatio_"+str(Lang)+"_"+str(CorpusSize)+"_"+str(Kini)
    with open(LocalMaxCacheForCorpusSizeVsHitRatioFileName,'w') as file1, open(EmpLocalMaxPercentCacheForCorpusSizeVsHitRatioFileName,'w') as file2, open(LocalMaxPercentCacheForCorpusSizeVsHitRatioFileName,'w') as file3, open(EmpLocalMaxCacheForCorpusSizeVsHitRatioFileName,'w') as file4:
        for i in range(0,Nsteps +1):
            GlobHitRatio=InitHR + (1- InitHR)*(1- np.exp(- jump * i))
            (GlobCacheSize,_,_,GlobCacheSizePercent)= LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,ValidCorporaList=ValidCorporaList,Voc=Voc,DicDicGHofK=DicDicGHofK,Kini=Kini, verb=False)
            st1=str(GlobHitRatio)+"\t"+str(GlobCacheSize); st3=str(GlobHitRatio)+"\t"+str(GlobCacheSizePercent);
            file1.write(st1+"\n"); file3.write(st3+"\n");
            (EmpGlobCacheSize,_,_,EmpGlobCacheSizePercent) = EmpLocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,TestCorporaDic, Distk, Kini=Kini)
            ErrRel=abs(GlobCacheSize - EmpGlobCacheSize)/EmpGlobCacheSize
            if(verb):
                print("Relative Error of cache size for HitRatio",GlobHitRatio,"and corpus size",CorpusSize,":",ErrRel)
            GlobErr.append(ErrRel)
            st2=str(GlobHitRatio)+"\t"+str(EmpGlobCacheSizePercent)
            file2.write(st2+"\n");
            st4= str(GlobHitRatio)+"\t"+str(EmpGlobCacheSize); file4.write(st4+"\n");
        file2.close(); file4.close()
    if(verb):
        GlobErr=np.array(GlobErr)
        print("Mean Error:",np.mean(GlobErr),";","Mean Square Root Error:",np.mean(GlobErr**2)**0.5)

        

#ok
def LocalMaxFindBestCacheVsHitRatioPointForCorpusSize(Lang,MaxNgramSize,CorpusSize,InitHR,DeltaHR,CacheImportance=1,Kini=2,verb=False):
    ValidCorporaList={}; Distk={}; Voc={}; VocBD(Voc)
    ValidationCorporaSizesList(ValidCorporaList); ValidationAndTestingDistkBD(Distk)
    GlobErr=[]
    TestCorporaDic={}
    TestingCorporaSizesList(TestCorporaDic);
    TestCorpora=TestCorporaDic[(Lang,1)]
    DicDicGHofK={}
    for ng in range(1,MaxNgramSize+1):
        DicDicGHofK[ng]=LoadSoftGH(Lang,ng,"DepParms")
    LocalMaxFindBestCacheVsHitRatioPointForCorpusSizeFileName="LocalMaxFindBestCacheVsHitRatioPointForCorpusSize_"+str(Lang)+"_"+str(CorpusSize)+"_"+str(CacheImportance)+"_"+str(Kini)
    LocalMaxEvolForFindBestCacheVsHitRatioPointForCorpusSizeFileName="LocalMaxEvolForFindBestCacheVsHitRatioPointForCorpusSize_"+str(Lang)+"_"+str(CorpusSize)+"_"+str(CacheImportance)+"_"+str(Kini)
    with open(LocalMaxFindBestCacheVsHitRatioPointForCorpusSizeFileName,'w') as file1,open(LocalMaxEvolForFindBestCacheVsHitRatioPointForCorpusSizeFileName,'w') as file2:
        GlobHitRatio= InitHR; MinDerivRatio=999999999999
        (GlobCacheSize,_,_,_)= LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio,ValidCorporaList=ValidCorporaList,Voc=Voc,DicDicGHofK=DicDicGHofK,Kini=Kini, verb=False)
        while (GlobHitRatio <=1-DeltaHR):
            (NextGlobCacheSize,_,_,CacheOverD1)= LocalMaxRequiredCache(Lang,MaxNgramSize,CorpusSize,GlobHitRatio + DeltaHR,ValidCorporaList=ValidCorporaList,Voc=Voc,DicDicGHofK=DicDicGHofK,Kini=Kini, verb=False)
            DerivRatio=(DeltaHR/(GlobHitRatio + DeltaHR)) / (( (NextGlobCacheSize - GlobCacheSize)/NextGlobCacheSize )**CacheImportance)
            st2=str(GlobCacheSize)+"\t"+str(GlobHitRatio)
            file2.write(st2+"\n");
            if(DerivRatio < MinDerivRatio and DerivRatio > 0 ):
               MinDerivRatio = DerivRatio
               OptCacheSize=NextGlobCacheSize
               OptHr= GlobHitRatio + DeltaHR
               OptCacheOverD1=CacheOverD1
            GlobHitRatio+= DeltaHR
            GlobCacheSize=NextGlobCacheSize
        if(verb):
            print("Most efficient HitRatio Vs Cache size for LocalMax application, for ",CorpusSize,"corpus size",":",OptHr,"(Hit Ratio);",OptCacheSize,"(Cache Size)","corresponding to",OptCacheOverD1,"of cache size for 100% Hit ratio ")
        st1=str(OptCacheSize)+"\t"+str(OptHr)
        file1.write(st1+"\n");
    file1.close(); file2.close()
   

def PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,CorpusForScale,C,k,DicGHofK,V):
    if(C > GreatestTrainCorpus ):
        (PredGk,PredHk)=GetGH(Lang,NgramSize,DicGHofK,k,CorpusForScale )
    else:
        (PredGk,PredHk)=GetGH(Lang,NgramSize,DicGHofK,k,C)
    return CalcD(V,PredGk,PredHk,C)

    
def GetGH(Lang,NgramSize,DicGHofK,k,CorpusSize):
    (SplineG,SplineH)=DicGHofK[k]
    Gpred = SplineG(CorpusSize)
    Hpred = np.exp(-1 * SplineH( CorpusSize))
    return (Gpred,Hpred)


def CalcD(V,Ka,Kb,C):
    return V*1.0/(1 + (Kb * C)**-Ka)


def Dk(Lang,NgramSize,CorpusSize,k,Voc,ValidCorporaList):
    ValidCorpora=ValidCorporaList[(Lang,NgramSize)]
    ValidCorpora=sorted(ValidCorpora)
    GHCorpusForScale=ValidCorpora[POSFORSCALE]
    GreatestTrainCorpus=ValidCorpora[-2]
    DicGHofK=LoadSoftGH(Lang, NgramSize,"DepParms")
    V=Voc[(Lang,NgramSize)]
    return PredictDkScale(Lang,NgramSize,GreatestTrainCorpus,GHCorpusForScale,CorpusSize,k,DicGHofK,V)
    


###########


















       







         
  
  


        






            

#ok 


      



               
        

 
            
     




     
    

        
        
    
    
         
   
    
             
        
    
    





    



    

        
        
        

    
            
    
       
 
        
        

    









#




    

			

	
