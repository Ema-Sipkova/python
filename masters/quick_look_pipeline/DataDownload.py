#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Usage:
#
#     $ python3 FrekSpec_Multiproc.py list
#

import sys
import time

from multiprocessing import Pool
from datetime import datetime

import lightkurve as lk
from lightkurve import search_targetpixelfile
from lightkurve.periodogram import LombScarglePeriodogram

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import matplotlib.image as img


def fce_execute(input_filename):
    # plonkove procesy, jen aby to bezelo
    result = "%s %s X Y Z" % (datetime.utcnow(), input_filename[0])    
    begin_dt = datetime.utcnow()

    # umele zatizeni CPU, aby bylo v 'top' mozno sledovat, ze bezi dany pocet jader
    while (datetime.utcnow() - begin_dt).total_seconds() < 0.05:
        pass
    

    
    t0 = time.process_time()

    # =============== Noise at f+-1c/d =========
    def fnoise (A, res):     
        m = A[:,1].argmax()
        fs = A[ m, 0]
        nla = 0.0                
        nls = 0.0
        if (fs - A[0,0]) > 1:
            nla = np.average(A[int(m-res):int(m+res),1]) # noise as average amplitude
            nls = np.std(A[int(m-res):int(m+res),1]) # noise as stdeva
        else:
            nla = np.average(A[0:int(m+res),1])
            nls = np.std(A[0:int(m+res),1])
        return A[m,1], nla, nls     
    
    
    
    # ============== Phase function =============
    def fphase ( m0 , f, A ):
        n = len(A)
        B = np.zeros([n,2]); B[:,1] = A[:,1]
        for i in range(n):
            B[i,0] = (A[i,0]-m0)*f - np.floor((A[i,0]-m0)*f) 
        C=np.zeros([n,2]); C[:,0]=B[:,0]-1; C[:,1]=B[:,1]        
        D = np.concatenate((B,C), axis=0); D=D[D[:,0].argsort()]
        return D
    
    
    # ========= Position of gaps in the data =====
    def findgaps(times, mingap = .5):
        ret=[]
        t=times[0]
        for i in range(1, len(times)):
            if times[i]-t >= mingap:
                ret.append(i)
            t=times[i]
        ret.append(len(times))
        return ret
        
    # ============================= Main function ==================================
    
    vstup = str(input_filename[0])
    
    name = 'TIC '+vstup     # nazev
    auth = 'TESS-SPOC'      # rutina
    TS = lk.search_lightcurve(name, author=auth)
    if len(TS)!=0:
        Tspoc_data = TS.download_all(); 
        Tspoc_stitched = Tspoc_data.stitch()
        Tspoc_stitched = Tspoc_stitched.remove_nans()
        
        # time correction
        tt = np.ascontiguousarray(Tspoc_stitched.time.value + 2457000.0, dtype=np.float64)
        
        # mean magnitude
        tmflux = 1.0
        tm = np.ascontiguousarray(-2.5*np.log10(Tspoc_stitched.flux/tmflux), dtype=np.float64)
        tmerr = np.ascontiguousarray(2.5*Tspoc_stitched.flux_err/Tspoc_stitched.flux/np.log(10), dtype=np.float64)
        
        TS = np.zeros([len(tt),3], dtype=np.float64)
        TS[:,0] = tt; TS[:,1] = tm; TS[:,2] = tmerr
        
        #######################################################
        # Saving the light curve
        np.savetxt(vstup+'_'+auth+'.dat', TS)
        
        # ============== TPF download  ===================
        datalist = search_targetpixelfile(name)
        # pokud se nepodari stahnout tpf, pouzije se obrazek ze souboru
        if len(datalist)==0:
            im = img.imread('smile.png')
        else:
            #reading png image file  
            tes = datalist[0].download()
            plt.ioff()
            tes.plot(aperture_mask=tes.pipeline_mask)
            plt.savefig(vstup+'_tpf.png', bbox_inches='tight')
            im = img.imread(vstup+'_tpf.png')            
            plt.close('all')
    
        
        d = TS
        hjd = np.average(d[:,0])
        d[:,0] -= np.average(d[:,0])
        d[:,1] -= np.average(d[:,1])
        gap_size = 0.3                     
        gap = findgaps(d[:,0],gap_size)
        n = len(d) 
        
        zoom = 1000 # scalling for mmag
        #zoom = 1e6  # scalling for power mmag
           
        ts = d[-1,0]-d[0,0]         # casovy rozsah dat
        fres = (1.0/ts)             # Frequency resolution
        rms = np.std(d[:,1])*zoom   # rms   
    
        # Definice rozsahu spekter a samplingu; frend je nejvyssi frekvence
        frstart = 0.01; frend = 100; frsamp = int(6*frend/fres) 
        #wend = 1.0; wsamp =  int(6*wend/fres)                         
        
        
        # ======== Frequency spectrum ================
        fr = np.linspace(frstart, frend, frsamp); nfr = len(fr)
        LS = LombScarglePeriodogram.from_lightcurve(Tspoc_stitched, frequency=fr)
        amp=np.array(LS.power)                              # Amplitude of the peaks
        amp = np.nan_to_num(amp)
        
        resolution = np.floor(frsamp/frend)                 # Frequency resolution
        FR = np.zeros([nfr,2]); FR[:,0] = fr; FR[:,1] = amp # Matrix with the frequency spectrum
        
        # =============== Subsamples of the frequency spectrum in GDOR and DSCT regimes ===============
        GD = FR[ FR[:,0] <= 5.0, :]
        DS = FR[ (FR[:,0] >= 5.0) & (FR[:,0] < frend), :]

        
        # ====================================================
        
        maximum, noise, noise_std = fnoise(FR, resolution)
                
        # *****************************************************************************
        # ==========================================================================
        # ============ Plotting of light curves ===================
        fig = plt.figure(figsize=(20,15))
        gs = GridSpec(3, 6, figure=fig)
        plt.ioff()
               
        # The whole view on the data
        ax1 = fig.add_subplot(gs[0, :4])
        
        ax1.set_title('TIC'+vstup+'  Pts:'+str(n)+\
                      ',  TS:'+str(np.round(ts,1))+\
                      ' d,  Res:'+str(np.round(fres,3))+\
                      ' c/d HJD:'+str(np.round(hjd,5))+\
                      ' Fmax:'+str(np.round(FR[ FR[:,1].argmax(), 0], 3))+\
                      ' Amp:'+str(np.round(maximum/noise,3)), fontsize = 16)

        ax1.scatter(d[:,0], d[:,1]*zoom, marker='o', c='navy', s=6)
        xl1 = d[0,0]-ts/20
        xl2 = np.max(d[:,0]) + ts/20
        ax1.set_xlim(xl1, xl2)
        ax1.set_ylim(-5*rms, 5*rms)
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_ylabel(r'mmag', fontsize=20)
        ax1.set_xlabel(r'HJD', fontsize=20)
        ax1.tick_params(axis='x', labelsize=14); ax1.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        
        # ============= tpf plot ============
        ax2 = fig.add_subplot(gs[0, 4:]) 
        ax2.imshow(im) 
        ax2.axis('off')
        plt.tight_layout()
        
        
        
        # ========= Frequency of GDOR regime
        fshift = GD[ GD[:,1].argmax(), 0]   # fac je skalovani amplitud fr okna, fshift je hodnota piku s maximalni amplitudou ve frekvencnim spektru
        
        maximumGD, noiseGD, noise_std = fnoise(GD, resolution)
         
        ax3 = fig.add_subplot(gs[1, :4])
        ax3.axvline(fshift, color="lightskyblue", linestyle=":", lw=2)
        ax3.plot(GD[:,0], GD[:,1]*zoom, 'k-', alpha=1)      # frekcvencni spektrum
        plt.tight_layout()
        
        ax3.axhline(4*noiseGD*zoom, color="r", linestyle="--", lw=2)#, label="4.0*SN average")
        
        if maximumGD > 8*noiseGD:
            yl = 1.2*maximumGD*zoom
        else:
            yl = 12*noiseGD*zoom
        xl = 5
        ax3.set_ylim(0, yl)
        ax3.set_xlim(0, xl)
        ax3.text(0.8*xl, 0.9*yl, r'f$_{\rm max}$ = ' + str(np.round(fshift,3)) + r' c/d', fontsize=16)
        ax3.text(0.8*xl, 0.8*yl, r'SNR$_{\rm max}$ = ' + str(np.round(maximumGD/noiseGD,1)), fontsize=16)
        ax3.tick_params(axis='x', labelsize=14); ax3.tick_params(axis='y', labelsize=14)
        ax3.set_ylabel(r'mmag', fontsize=20)
        
        # ======= Phase gDor =================
        ax4 = fig.add_subplot(gs[1, 4:])       
        FAZE = fphase ( np.max(GD[:,1]) , fshift, d )        
        ax4.scatter(FAZE[:,0], FAZE[:,1]*zoom, marker='o', c='navy', s=3)
        ax4.set_ylim(-5*rms, 5*rms)
        ax4.set_ylim(ax4.get_ylim()[::-1])
        ax4.tick_params(axis='x', labelsize=14); ax4.tick_params(axis='y', labelsize=14)        
        plt.tight_layout()
        
        
        
        # =========================================================
        # ========= Frequency spectrum of the DSCT regime
        fshift = DS[ DS[:,1].argmax(), 0]   # fac je skalovani amplitud fr okna, fshift je hodnota piku s maximalni amplitudou ve frekvencnim spektru
        
        maximumDS, noiseDS, noise_std = fnoise(DS, resolution)
        
        ax5 = fig.add_subplot(gs[2, :4])
        ax5.axvline(fshift, color="lightskyblue", linestyle=":", lw=2)
        ax5.plot(FR[:,0], FR[:,1]*zoom, 'k-', alpha=1)               
        
        ax5.axhline(4*noiseDS*zoom, color="r", linestyle="--", lw=2, label="4.0*SN average")

        plt.tight_layout()
        
        if maximumDS > 8*noiseDS:
            yl = 1.2*maximumDS*zoom
        else:
            yl = 12*noiseDS*zoom
        
        xl2 = fshift+20.0

        if fshift <= 10.0:
            xl1 = DS[ DS[:,1].argmax(), 0] - 5.0
        else: 
            xl1 = DS[ DS[:,1].argmax(), 0] - 10.0

        ax5.set_ylim(0, yl)
        ax5.set_xlim(xl1, xl2)
        ax5.text(0.8*xl2, 0.9*yl, r'f$_{\rm max}$ = ' + str(np.round(fshift,3)) + r' c/d', fontsize=16)
        ax5.text(0.8*xl2, 0.8*yl, r'SNR$_{\rm max}$ = ' + str(np.round(maximumDS/noiseDS,1)), fontsize=16)
        ax5.text(0.8*xl2, 4.6*noiseDS*zoom, r'SNR$_{\rm at~fmax\pm1}$ = ' + str(4.0), fontsize=16, color='r')
        ax5.tick_params(axis='x', labelsize=14); ax5.tick_params(axis='y', labelsize=14)
        ax5.set_ylabel(r'mmag', fontsize=20)
        ax5.set_xlabel(r'Frequency (c/d)', fontsize=20)
        plt.tight_layout()
        
        # ======= Phase curve for dSct regime =================
        ax6 = fig.add_subplot(gs[2, 4:])       
        FAZE = fphase ( np.max(DS[:,1]) , fshift, d )        
        ax6.scatter(FAZE[:,0], FAZE[:,1]*zoom, marker='o', c='navy', s=3)
        ax6.set_ylim(-5*rms, 5*rms)
        ax6.set_ylim(ax6.get_ylim()[::-1])
        ax6.tick_params(axis='x', labelsize=14); ax6.tick_params(axis='y', labelsize=14)
        ax6.set_xlabel(r'Phase', fontsize=20)
        plt.tight_layout()
        
        # Ulozeni obrazku
        plt.savefig(vstup+'_'+auth+'.png', bbox_inches='tight')

        # Vypsani udaju o datasetu
        # TIC sektors pocetBodu casovyRozsah gdor amp snr dsct amp snr
        out = str(vstup)+","+str(int(len(gap)/2))+","+str(len(d))+","+str(np.round(ts,3))+","+str(np.round(GD[ GD[:,1].argmax(), 0],5))+","+str(np.round(GD[ GD[:,1].argmax(), 1],5))+","+str(np.round(maximumGD/noiseGD,3))+","+str(np.round(DS[ DS[:,1].argmax(), 0],5))+","+str(np.round(DS[ DS[:,1].argmax(), 1],5))+","+str(np.round(maximumDS/noiseDS,3))+","+str(hjd)
        
        print(out)
        
        with open(input_filename[1], 'a') as file:
            file.write(out + '\n')
        
        plt.close('all')

    else:
        print(vstup+" ######### ")
    
    return result

def main():

    if len(sys.argv) != 3:
        print("Usage: %s STARS_LST" % sys.argv[0])
        sys.exit()

    stars_lst = sys.argv[1]
    output_filename = sys.argv[2]

    with open(stars_lst) as fo:
        lines = fo.readlines()

    input_filenames = []
    for line in lines:
        input_filenames.append((line.strip(), output_filename))

    # zaridi, ze vzdy pobezi najednou maximalne 20 procesu - zalezi na poctu jader
    pool = Pool(processes=20)
    results = pool.map(fce_execute, input_filenames)

if __name__ == '__main__':
    main()

# python DataDownload.py list.txt params.csv
# list.txt => only TIC numbers
# params.csv => output file