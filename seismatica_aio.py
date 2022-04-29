#%% Imports
import sys
import numpy as np
import pandas as pd
import obspy
import scipy.spatial.distance as ssd
from scipy.stats.mstats import zscore
from matplotlib import pylab as plt
from matplotlib.dates import DateFormatter
import datetime
from multiprocessing import Pool

def dists(x, y, n_chunks):
    # Distances for choice: braycurtis canberra cityblock euclidean sqeuclidean minkowski
    d = len(x) // n_chunks
    dsts = []
    for i in range(n_chunks):
        dst = ssd.euclidean(x[i*d:(i+1)*d], y[i*d:(i+1)*d])
        dsts.append(dst)
    return(dsts)

def similarity(data):
    ent = []
    set_inv = data[3]
    n_chunks = data[4]
    control_id = data[5]
    
    for wave in data[:3]:
        df = np.diff(wave)**2
        df[df==0] = 1
        s = np.sum(df)
        ent.append(-df/s * np.log(df/s))

    E = ent[0] + ent[1] + ent[2]

    # Result test pattern
    H_test = np.cumsum(E)
    
    # Matrix D building and normalization
    D = zscore(np.array(set_inv + [H_test]), axis = 0)
    
    # Distances calculation
    diags = []
    for j in range(control_id + 1):
        dst = dists(D[j,:], D[-1,:], n_chunks)
        diags.append(dst)
    res_matrix = np.array(diags)
    
    # Minimal distances from test to control patterns counts
    experts_count = len(dst)
    control_rate = 0
    for k in range(experts_count):
        if np.argmin(res_matrix[:,k]) == control_id:
            control_rate += 1

    return(control_rate / experts_count)
    
#%% Main script
if __name__ == '__main__':
    
    cfg_name = sys.argv[1]
    # cfg_name = 'seismatica_1.cfg' # Manual set cfg for debug
    cfg = {}
    cfg_lines = []
    
    try:
        with open(cfg_name, 'r') as f:
            cfg_lines = f.readlines()
        f.close()
    except:
       print('Config file not found')
       sys.exit(0)
    
    for line in cfg_lines:
        if line[0] != '#':
            rec = line.split('=')
            cfg[rec[0]] = rec[1].replace('\n', '')
    
    #%% Build control pattern
    t1, t2 = cfg['control_time'].split(' ')
    x1 = t1.split(':')
    x2 = t2.split(':')
    waveforms_c = []
    for line in cfg['control'].split(' '):
        mseed = obspy.read(line, format = 'mseed')
        sinfo = mseed[0].stats
        sdata = mseed[0].data
        st = sinfo.starttime.datetime.replace(hour = int(x1[0]), minute = int(x1[1]), second = int(x1[2]), microsecond = 0) 
        et = sinfo.starttime.datetime.replace(hour = int(x2[0]), minute = int(x2[1]), second = int(x2[2]), microsecond = 0)
        
        mseed = obspy.read(line, starttime=obspy.UTCDateTime(st), endtime=obspy.UTCDateTime(et), format = 'mseed')
        waveforms_c.append([sinfo['channel'], mseed[0].data])
    
    t_c = pd.date_range(st, et, periods = len(mseed[0].data)).to_pydatetime()
    
    diffs = []
    ent = []
    
    for wave in waveforms_c:
        df = np.diff(wave[1])**2
        df[df==0] = 1
        diffs.append(df)
        s = np.sum(df)
        ent.append(-df/s * np.log(df/s))
    
    E = ent[0] + ent[1] + ent[2]

    # Control pattern
    H_cont = np.cumsum(E)
    N = len(H_cont)
    
    ### Build invented patterns for matrix D
    set_inv = []
    H_max = 3 * np.log(N)
    H_sign = np.cumsum(3 * -np.ones(N)/N * np.log(np.ones(N)/N))
    
    ### Set range of multiplicator a = 0.7...2.1 and generate invented patterns
    for a in np.linspace(H_max / N * 0.7, H_max / N * 2.1, int(cfg['n_invented'])):
        patt_inv = np.arange(N) * a
        patt_inv[patt_inv > H_max] = H_max
        set_inv.append(patt_inv)
    
    ### Append to invented patterns set the control pattern
    set_inv.append(H_sign)
    set_inv.append(H_cont)
    control_id = len(set_inv) - 1

    ### Detection preparation
    t1, t2 = cfg['seismogram_time'].split(' ')
    x1 = t1.split(':')
    x2 = t2.split(':')
    waveforms = []
    for line in cfg['seismogram'].split(' '):
        mseed = obspy.read(line, format = 'mseed')
        sinfo = mseed[0].stats
        sdata = mseed[0].data
        st = sinfo.starttime.datetime.replace(hour = int(x1[0]), minute = int(x1[1]), second = int(x1[2]), microsecond = 0) 
        et = sinfo.starttime.datetime.replace(hour = int(x2[0]), minute = int(x2[1]), second = int(x2[2]), microsecond = 0)
        
        mseed = obspy.read(line, starttime=obspy.UTCDateTime(st), endtime=obspy.UTCDateTime(et), format = 'mseed')
        waveforms.append([sinfo['channel'], mseed[0].data])
    
    st_wave = st
    scan_step = int(sinfo['sampling_rate'] * float(cfg['scan_step']))
    step_count = int((len(waveforms[0][1]) - N - 1) / scan_step)

    n_chunks = int(cfg['n_chunks'])
    
    ### Calculations
    if 'mode' in list(cfg):
        mode = cfg['mode']
    else:
        mode = 'gil'
    
    print('Run in <%s> mode ...'%mode)
    start = datetime.datetime.now()

    frames = [[waveforms[0][1][i*scan_step:i*scan_step+N+1], waveforms[1][1][i*scan_step:i*scan_step+N+1], waveforms[2][1][i*scan_step:i*scan_step+N+1], set_inv, n_chunks, control_id] for i in range(step_count)]
    
    if mode == 'mpi':
        with Pool(processes = 3) as poolx:
            control_prob = poolx.map(similarity, frames)
            poolx.close()
            poolx.join()
    else:
        control_prob = list(map(similarity, frames))
        
    stop = datetime.datetime.now()    
    print('Job is done for', stop - start)
    
    ### Plotting
    date_form = DateFormatter("%H:%M:%S")
    tw = pd.date_range(st, et, periods = len(mseed[0].data)).to_pydatetime()
    tp = st + np.arange(step_count) * datetime.timedelta(seconds = float(cfg['scan_step']))
    tickstep = 10 # int(cfg['n_chunks'])
    ticks = st + np.arange(int((et-st).total_seconds() / tickstep) + 1) * datetime.timedelta(seconds = tickstep)
    ylbl1 = ['X ', 'Y ', 'Z ']
    ylbl2 = [r'$E_X$', r'$E_Y$', r'$E_Z$']
    
    if cfg['plot_control'] == 'True':
        plt.figure(figsize = (12, 12), dpi = 300)
        for i in range(1, len(waveforms)+1):
            ax = plt.subplot(3, 2, (i-1) * 2 + 1)
            plt.plot(t_c, waveforms_c[i-1][1] / max([abs(max(waveforms_c[i-1][1])), abs(min(waveforms_c[i-1][1]))]), 'k', alpha = 0.75)
            plt.ylabel(ylbl1[i-1], fontsize = 16, rotation = 0)
            plt.xticks(fontsize = 12, rotation = 90)
            plt.yticks(fontsize = 12)
            plt.title(waveforms[i-1][0], fontsize = 16)
            plt.grid(True)
        plt.xlabel('Time', fontsize = 16)
        # plt.show()

        # plt.figure()
        for i in range(1, len(ent)+1):
            plt.subplot(3, 2, i * 2)
            plt.plot(ent[i-1], 'k', alpha = 0.75)
            plt.ylabel(ylbl2[i-1], fontsize = 16, rotation = 0, horizontalalignment='right')
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.grid(True)
        plt.xlabel(r'Counts $\it{i}$', fontsize = 16)
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(E, 'k', alpha = 0.75)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        # plt.xlabel(r'Counts $\it{i}$')
        plt.ylabel(r'$\bf{H} $', fontsize = 16, rotation = 0, horizontalalignment='right')
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(H_cont, 'k', alpha = 0.75)
        plt.xlabel(r'Counts $\it{i}$', fontsize = 16)
        plt.ylabel(r'$\bf{C} $', fontsize = 16, rotation = 0, horizontalalignment='right')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)    
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    if cfg['plot_invent'] == 'True':
        plt.figure()
        # plt.subplot(1, 2, 1)
        plt.plot(H_cont, '-g', lw = 3, alpha = 0.8)
        plt.plot(H_test, '-.b', lw = 2, alpha = 0.8)
        plt.plot(H_sign, '--k', lw = 2)
        
        for df in set_inv:
            plt.plot(df, '--k', lw = 1)
            
        plt.xticks(np.arange(11)*N//10, np.arange(11)*N//10, fontsize = 16)
        plt.xlabel(r'Counts $\it{i}$', fontsize = 18)
        plt.ylabel(r'$\bf{C}$', rotation = 0, fontsize = 16, horizontalalignment='right')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12) 
        plt.legend(['Control pattern', 'Test pattern', 'Singular pattern', 'All invented patterns'], fontsize = 14)
        # plt.legend(['Контрольный шаблон', 'Тестовый шаблон', 'Вырожденный шаблон', 'Все модельные шаблоны'], fontsize = 14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure()
        # plt.subplot(1, 2, 2)
        plt.plot(D[-2,:], '-g', lw = 3, alpha = 0.8)
        plt.plot(D[-1,:], '-.b', lw = 2, alpha = 0.8)
        plt.plot(D[-3,:], '--k', lw = 2)
        
        for i in range(len(D) - 3):
            plt.plot(D[i,:], '--k', lw = 1)
            
        plt.xticks(np.arange(11)*N//10, np.arange(11)*N//10, fontsize = 16)
        plt.xlabel(r'Counts $\it{i}$', fontsize = 16)
        plt.ylabel(r'$\bf{D}$', rotation = 0, fontsize = 16, horizontalalignment='right')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12) 
        plt.legend(['Control pattern', 'Test pattern', 'Singular pattern', 'All invented patterns'], fontsize = 14)
        # plt.legend(['Контрольный шаблон', 'Тестовый шаблон', 'Вырожденный шаблон', 'Все модельные шаблоны'], fontsize = 14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    if cfg['plot_scan'] == 'True':
        if len(tp) < 500:
            plt.figure()
            for i in range(len(waveforms)):
                ax = plt.subplot(4, 1, i + 1)
                plt.rcParams['axes.xmargin'] = 0
                plt.plot(tw, waveforms[i][1] / max([abs(max(waveforms_c[i-1][1])), abs(min(waveforms_c[i-1][1]))]), 'k', alpha = 0.75)
                
                if cfg['plot_windows'] == 'True':
                    
                    winds = cfg['timestamps'].split(' ')
                    scales = np.random.randint(5, 10, size = len(winds)) / 10
                    for w, s in zip(winds, scales):
                        
                        x1 = w.split(':')
                        st = sinfo.starttime.datetime.replace(hour = int(x1[0]), minute = int(x1[1]), second = int(x1[2]), microsecond = 0)
                        et = st + datetime.timedelta(seconds = int(N / sinfo['sampling_rate']))
                        mx = max(waveforms[i][1])
                        plt.plot([st, et], [min(waveforms[i][1])*s, min(waveforms[i][1])*s], '--k', lw = 1)
                        plt.plot([et, et], [min(waveforms[i][1])*s, max(waveforms[i][1])*s], '--k', lw = 1)
                        plt.plot([st, et], [max(waveforms[i][1])*s, max(waveforms[i][1])*s], '--k', lw = 1)
                        plt.plot([st, st], [min(waveforms[i][1])*s, max(waveforms[i][1])*s], '--k', lw = 1)
                        plt.text(st, max(waveforms[i][1])*s, w, size=12, ha="left", va="top", bbox=dict(boxstyle="roundtooth", ec='k', fc='w',))
                
                plt.yticks(fontsize = 14)    
                plt.xticks(ticks, fontsize = 10)
                plt.setp(ax.get_xticklabels(), visible = False)
                plt.ylabel(ylbl1[i-1], fontsize = 16, rotation = 0)
                ax.xaxis.set_major_formatter(date_form)
                plt.grid(True)
                
            ax = plt.subplot(4, 1, 4)
            plt.rcParams['axes.xmargin'] = 0
            plt.plot(tp, control_prob, '--k', lw = 1)
            plt.plot(tp, control_prob, 'o', markersize = 6, markeredgecolor = 'black', linewidth = 1, color = 'orange', alpha = 0.7)
            plt.ylim([0, 1.1])
            plt.xticks(ticks, fontsize = 14, rotation = 90)
            plt.yticks(np.linspace(0, 1, 5), fontsize = 14)
            plt.xlabel('Time', fontsize = 16)
            plt.ylabel('p', rotation = 0, fontsize = 16, horizontalalignment='right')
            ax.xaxis.set_major_formatter(date_form)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            #%%
            plt.figure()
            for i in range(len(waveforms)):
                ax = plt.subplot(4, 1, i + 1)
                plt.rcParams['axes.xmargin'] = 0
                plt.plot(tw, waveforms[i][1] / max([abs(max(waveforms_c[i-1][1])), abs(min(waveforms_c[i-1][1]))]), 'k', alpha = 0.75)
                
                plt.yticks(fontsize = 14)    
                # plt.xticks(ticks, fontsize = 10)
                # plt.setp(ax.get_xticklabels(), visible = False)
                plt.ylabel(ylbl1[i], fontsize = 16, rotation = 0)
                ax.xaxis.set_major_formatter(date_form)
                plt.grid(True)
            
            events = []
            num = 0
            for i in range(len(control_prob)):
                if control_prob[i] > 0.35:
                    num += 1
                    events.append([tp[i], control_prob[i], num])
                    
            ax = plt.subplot(4, 1, 4)
            plt.rcParams['axes.xmargin'] = 0
            plt.plot(tp, control_prob, '-r', lw = 1, alpha = 0.75)
            # plt.plot(tp, control_prob, 'o', markersize = 6, markeredgecolor = 'black', linewidth = 1, color = 'orange', alpha = 0.7)
            # for event in events:
            #     plt.text(event[0], event[1], str(event[2]), fontsize = 16, ha = 'center', va = 'bottom', bbox = dict(boxstyle="circle,pad=0.3", fc="w", ec="k", lw=1))
            
            plt.ylim([0, 1.1])
            # plt.xticks(ticks, fontsize = 14, rotation = 90)
            plt.yticks(np.linspace(0, 1, 5), fontsize = 14)
            plt.xlabel('Time', fontsize = 16)
            plt.ylabel('p', rotation = 0, fontsize = 16, horizontalalignment='right')
            ax.xaxis.set_major_formatter(date_form)
            plt.grid(True)
            plt.tight_layout()
            plt.show()            