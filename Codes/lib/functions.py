def roc_auc(robust,robust_adver):
        import numpy as np
        from scipy import stats
        nb = robust.shape[0]    # number of images
        
        # Cumulative histograms
        res = stats.cumfreq(robust, numbins=nb, defaultreallimits=(0, 1))
        res_adver = stats.cumfreq(robust_adver, numbins=nb, defaultreallimits=(0, 1))
        
        # True and false positives
        vpos=res_adver[0]/nb
        fpos=res[0]/nb

        first_element = np.array([0])
        vpos = np.append(first_element,vpos)
        fpos = np.append(first_element,fpos)      

        # Compute ROC_AUC with the integral
        roc_auc=0
        for i in range(1,nb+1):
            roc_auc=roc_auc+(fpos[i]-fpos[i-1])*(vpos[i]+vpos[i-1])/2    
        
        return roc_auc

def min_error(robust,robust_adver): # Minimum error of classification (false positives + false negatives) divided by total number of images
        import numpy as np
        from scipy import stats 
        nb = robust.shape[0] # number of images
        # Histogramas cumulativos)
        res = stats.cumfreq(robust, numbins=nb, defaultreallimits=(0, 1))
        res_adver = stats.cumfreq(robust_adver, numbins=nb, defaultreallimits=(0, 1))
        false_clasif=nb-res_adver[0]+res[0] # error de clasificacion como funcon del umbral
        i=np.argsort(false_clasif)[0] # se busca el menor
        bin_size = res.binsize
        best_threshold = (i+1) * bin_size # this goes between bin_size and 1
        return ( false_clasif[i]/(2*nb), best_threshold)

def float_to_str(f):
    ret = "{:.10f}".format(f).rstrip("0").replace(".","_")
    if ret[-1] == "_":
        ret += "0"
    return ret