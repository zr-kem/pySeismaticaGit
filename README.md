# pySeismaticaGit
Seismic events detection and classification tool preview.

# Requremnets
Python 3.8+
    sys numpy scipy pandas datetime obspy matplotlib

    Packages instalation:
    Linux:
        pip3 install [list_of_packages] --user
    Windows:
        python -m pip install [list_of_packages]

# Script using
    Linux:
        python3 seismatica_aio.py config_file_name
    Windows:
        python seismatica_aio.py config_file_name

# Config file options
## Full path to miniseed seismogram for control pattern, directions are space separated
control=AN.BRCR.81.EHE.D.2013.004 AN.BRCR.81.EHN.D.2013.004 AN.BRCR.81.EHZ.D.2013.004
## Time window of control pattern splash
control_time=08:50:48 08:52:08
## Full path to miniseed seismogram for test pattern, directions are space separated
seismogram=AN.BRCR.81.EHE.D.2013.004 AN.BRCR.81.EHN.D.2013.004 AN.BRCR.81.EHZ.D.2013.004
## Time window of test patterns, must be wider than 'control_time'
seismogram_time=08:49:48 08:53:08
## Scanning step for 'seismogram_time', usually is 1 second
scan_step=1
## Number of invented patterns
n_invented=20
## Number of chunks
n_chunks=20
## Plotting options
### Plot control pattern: source waveform, transformations, final shape
plot_control=False
### Plot invented patterns - source and normalized shapes. Depricated!
plot_invent=False
### Plot similarity evolution step-by-step for a number of time windows
plot_scan=True
### Plot sample time windows on 'seismogram' at desired 'timestamps'
plot_windows=True
timestamps=08:49:49 08:50:48 08:51:45
### Run in multiprocessor mode - set mode to 'mpi' or none
mode=mpi
