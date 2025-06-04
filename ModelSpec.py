#import KK as KK
#import MEM as MEM

import numpy as np
import pandas as pd
from PreprocSpec import SpecAnalyzer, evalclassif, MeansensiPlot
#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import Phase_Retrieval_Package.KK as KK
import Phase_Retrieval_Package.MEM as MEM

def SpectralPreproc(wn, data, datatype = None, wncalib = None, wncalib_param = None
                    , Intcalib = None, Intcalib_param = None
                    , despike_m = None, despike_thresh = None, crop_range = None
                    , snip_iter = None, snip_w = None, smooth_w = None, smooth_poly_order = None
                    , silent_reg = None, norm_method = None, phase_retrieval=None, nb_poles=None):
    """
    Parameters
    ----------
    wn : numpy array
        Wavenumber axis.
    data : numpy ndarray
        Spectra in rows.
    datatype : string, optional
        Type of data including 'Raman', 'IR', 'LSPR', 'Fluor', 'CARS'. The default is None.
    wncalib : logical, optional
        True if wavenumber calibration is required to be implemented. The default is None.
    wncalib_param : dict, optional
        Dictionary of parameters for wavenumber calibration. The default is None.
    Intcalib : logical, optional
        Perform intensity calibration in case of CARS data. The default is None.
    Intcalib_param : dict, optional
        Dictionary of the parameters for intensity calibration. The default is None.
    despike_m : int, optional
        Kernel size for despiking. The default is None.
    despike_thresh : int, optional
        Threshold value in case of despiking. The default is None.
    crop_range : list, optional
        List of 2 values to crop data. The default is None.
    snip_iter : int, optional
        Number of iteration in case of SNIP. The default is None.
    snip_w : int, optional
        The window size in case of SNIP. The default is None.
    smooth_w : int, optional
        The window size for smoothing. The default is None.
    smooth_poly_order : int, optional
        The order of the polynomial for smoothing. The default is None.
    silent_reg : list, optional
        List of 2 values to remove outside. The default is None.
    norm_method : string, optional
        The normalization method to implement including 'vector', 'l1', 'max'. The default is None.
    phase_retrieval : string, optional
        Implement phase retrieval in case of CARS data and here indicate 'KK' or 'MEM'. The default is None.
    nb_poles : int, optional
        The number of poles used in case of 'MEM'. The default is None.

    Returns
    -------
    spec_idx : list
        list of indices representing the spectra that kept in the analysis after applying quality check.
    preproc_spc : numpy ndarray
        Return the preprocessed spectra.

    """
    preproc_fn = SpecAnalyzer(wn,data)
    if despike_m is not None:
        preproc_spc = preproc_fn.despike(despike_m, despike_thresh)
        pd.DataFrame(preproc_spc).to_csv("Despiked_Spectra.csv", index_label= 'Wavenumber',
                              header=[wn[i] for i in range(len(wn))])
        preproc_fn.plot_mean_sd('Despiked')
        
    if wncalib is not None:
        if datatype =='CARS':
            kk = KK.KramersKronig(pad_factor=len(wn))
            outRec = kk.calculate(wncalib_param['wavenumberstandard'])
            imagpart_kk = outRec.imag
            wncalib_param['wavenumberstandard'] = imagpart_kk
            calib_object = preproc_fn.make_calibration(wncalib_param)
            out_calib = preproc_fn.apply_calibration(data, calib_object, wn, white_true_new=None)
            preproc_spc = out_calib['mat_new']
            wn = out_calib['x_new']
            pd.DataFrame(preproc_spc).to_csv("wnCalibrated_Spectra.csv", index_label= 'Wavenumber',
                              header=[wn[i] for i in range(len(wn))])
            preproc_fn.plot_mean_sd('Wavenumber calibrated')
            
            if Intcalib is not None:
                unif_CARS, preproc_spc = preproc_fn.CARSIntCalib(Intcalib_param)
                pd.DataFrame(preproc_spc).to_csv("IntCalibrated_Spectra.csv", index_label= 'Wavenumber'
                                                 , header=[wn[i] for i in range(len(wn))])
                preproc_fn.plot_mean_sd('Intensity calibrated')
                
            if phase_retrieval=='KK':
               kk = KK.KramersKronig(pad_factor=len(wn))
               for i in range(data.shape[0]):
                   outRec = kk.calculate(preproc_spc[i,])
                   preproc_spc[i,] = outRec.imag    
            elif phase_retrieval=='MEM':
                for i in range(data.shape[0]):
                    MEMfn = MEM.buildMEM(preproc_spc[i,], nb_poles)
                    outMEM = MEM.applyMEM(MEMfn, nu_0=[0])
                    Window = MEMfn['w']
                    preproc_spc[i,] = outMEM['imag'][..., Window == 1]       
        else:
            calib_object = preproc_fn.make_calibration(wncalib_param)
            out_calib = preproc_fn.apply_calibration(data, calib_object, wn, white_true_new=None)
            preproc_spc = out_calib['mat_new']
            wn = out_calib['x_new']
            pd.DataFrame(preproc_spc).to_csv("wnCalibrated_Spectra.csv", index_label= 'Wavenumber',
                              header=[wn[i] for i in range(len(wn))])
            preproc_fn.plot_mean_sd('Wavenumber calibrated')
    
    if crop_range is not None:
        preproc_spc = preproc_fn.crop_spectrum(crop_range)
        wn = wn[np.where((wn > crop_range[0]) & (wn < crop_range[1]))[0]]
        pd.DataFrame(preproc_spc).to_csv("Crop_Spectra.csv", index_label= 'Wavenumber',
                                      header=[wn[i] for i in range(len(wn))])
        preproc_fn.plot_mean_sd('Cropped')
        
    if snip_iter is not None: 
        preproc_spc = preproc_fn.snip(snip_iter, snip_w)
        pd.DataFrame(preproc_spc).to_csv("Baseline_corr_Spectra.csv", index_label= 'Wavenumber',
                                      header=[wn[i] for i in range(len(wn))])
        preproc_fn.plot_mean_sd('Baseline corrected')
    if smooth_w is not None:    
        preproc_spc = preproc_fn.smooth_spectra(smooth_w, smooth_poly_order)
        pd.DataFrame(preproc_spc).to_csv("Smoothed_Spectra.csv", index_label= 'Wavenumber',
                                      header=[wn[i] for i in range(len(wn))])
        preproc_fn.plot_mean_sd('Smoothed')
        
    spec_idx, preproc_spc = preproc_fn.quality_check()
    pd.DataFrame(preproc_spc).to_csv("Qualitycontr_Spectra.csv", index_label= 'Wavenumber',
                                  header=[wn[i] for i in range(len(wn))])
    preproc_fn.plot_mean_sd('Quality Control step')
    
    if silent_reg is not None:    
        preproc_spc = preproc_fn.remove_silentreg(silent_reg)
        #removesilent_spc[np.where((wn > silent_reg[0]) & (wn < silent_reg[1]))[0]]=0
        pd.DataFrame(preproc_spc).to_csv("Remove_silent_Spectra.csv", index_label= 'Wavenumber',
                                      header=[wn[i] for i in range(len(wn))])
        preproc_fn.plot_mean_sd('Remove silent region')
    if norm_method is not None:     
        preproc_spc = preproc_fn.normal_spectra('vector')
        pd.DataFrame(preproc_spc).to_csv("Normalized_Spectra.csv", index_label= 'Wavenumber',
                                      header=[wn[i] for i in range(len(wn))])
        preproc_fn.plot_mean_sd('Preprocessed')
    return spec_idx, preproc_spc

def PCAClassif(data, label, batch, nFolds, CV, K, classmethd = 'lda'):
    """
    Parameters
    ----------
    data : numpy ndarray
        Input data.
    label : numpy ndarray
        Label to be used in the classification analysis.
    batch : numpy ndarray, optional
        Array of batch in case of 'LOO' cross-validation. The default is None.
    nFolds : int
        Number of folds in case of 'kfold' cross-validation.
    CV : string
        The type of cross-validation to use including 'kfold' or 'LOO'.
    K : int
        Number of components for the PCA analysis.
    classmethd : string, optional
        The classsification method to use either 'lda' or 'svm'. The default is 'lda'.

    Returns
    -------
    PRED : numpy ndarray
        Array of prediction values.

    """
    PRED = np.empty([data.shape[0], K] ,dtype='object')
    if CV == 'KFOLD':
        Folds = np.random.choice(range(1, nFolds+1), data.shape[0], replace=True)
    elif CV == 'LOO':
        Folds = len(np.unique(batch))
        Folds = batch
    
    for i in range(nFolds):
        Testset = data[np.where(Folds==i+1)]
        Labtrain =  label[np.where(Folds!=i+1)]
        Trainset = data[np.where(Folds!=i+1)]
    
        # PCA function
        # scaler = StandardScaler(with_std=False)
        # scaler.fit(Trainset)
        # Trainset = scaler.transform(Trainset)
        # Testset = scaler.transform(Testset)
        
        pcafn = PCA(n_components = K)    
        #pcafn.fit(Trainset)
        PCtrain = pcafn.fit_transform(Trainset)
        #PCtest = pcafn.transform(Testset)
        
        for j in range(K):
            selPC = PCtrain[:,range(j+1)]
            if classmethd == 'lda':
                ldafn = LinearDiscriminantAnalysis()
                ldafn.fit(selPC, Labtrain)
                PRED[np.where(Folds==i+1),j] = ldafn.predict(Testset[:,range(j+1)])
            elif classmethd == 'svm':
                svmfn = svm.SVC(kernel='rbf')
                svmfn.fit(selPC, Labtrain)
                PRED[np.where(Folds==i+1),j] = svmfn.predict(Testset[:,range(j+1)])
                
    PRED = np.array(PRED, dtype=np.float64)
    EVAL = evalclassif(label, PRED)
    MeansensiPlot(EVAL, K)
    
    return PRED
    