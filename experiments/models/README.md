# Probabilistic model experiments

This folder contains the code used for Table 1 of the paper. Each folder contains implementations of the models in various languages. Note that some files may not exist (for example, if a smaller version of the same baseline times out, it is unnecessary to run the larger version). 

Each folder contains the model programs for their respective entry in the table, and a script `test.py`. For each folder, `test.py` is a Python script that will automatically run each benchmark in the folder, and write the results to a output csv file. `test.py` in each case contains a variable `TESTS`, a list indicating which tests should be run (by default, all of them), and a variable `OUTPUT_FILENAME`: the name of the output csv file to be written to. You may want to modify these (for example, remove some entries from `TESTS` that have significant runtime to observe more runs of those with shorter runtime).

More specific details for each folder are given below:

## `dicejl`
This folder contains the models used in the 'Dice.jl (binary)' column of Table 1. The script for this folder assumes you have already done the installation/setup of Alea.jl according to the instructions in the primary README.

The models in this folder are timed internally using the Julia BenchmarkTools utility, and so `test.py` does not provide tunable parameters for the timing.

## `dicejl_onehot`
This folder contains the models used in the 'Dice.jl (one-hot)' column of Table 1. The script in this folder is the same as the one for `dicejl` in general, and so has the same notes.

## `webppl`
This folder contains the models used in the 'WebPPL' column of Table 1. This script assumes you have already installed WebPPL, and contains a modifiable variable `WEBPPL_COMMAND`. This is by default `webppl`, but can be modified to a `/path/to/webppl` depending on your installation. This also assumes you have installed `webppl-timeit`, which is used for timing. 

The WebPPL script contains two parameters for our trials. TIMEOUT is the timeout in seconds (default 3600s = 1h) for the trials; if any run of a baseline exceeds this timeout, the script will output 'x' for said baseline. RUNS is the number of runs to take the median over (default 5). You may want to modify this; lower runtime experiments should use a larger number of runs. 

## `psi`
This folder contains the models used in the 'Psi' and 'Psi (DP)' columns of Table 1. This script assumes you have already installed Psi, and contains a modifiable variable `PSI_COMMAND`: by default `psi`, and can as before be changed to `/path/to/psi` depending on your installation. Psi is timed using a system `time` utility - by default `/usr/bin/time`, and this script may have errors if a different `time` is used. 

The script `test.py` in `psi` runs both the Psi and Psi (DP) experiments by default. 

The Psi script, like the WebPPL script, contains two parameters for our trials. TIMEOUT is the timeout in seconds (default 3600s = 1h) for the trials; if any run of a baseline exceeds this timeout, the script will output 'x' for said baseline. RUNS is the number of runs to take the median over (default 5).

