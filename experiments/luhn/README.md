# Luhn model

This folder contains the code used for Figure 3 of the paper. Each subfolder contains runnable implementations of the Luhn model for IDs ranging from 2-10 digits.

Instructions for using the example code are given below. These assume you have already installed Dice.jl, WebPPL, and Psi according to the instructions in `../INSTALL.md`"

## Instructions

### Dice.jl

The `dicejl` directory consists of the following files:
- `luhn6_example.jl` is an example Luhn file with 6 ID digits; it can be run using the following command. It reports the time taken to run this program.

```bash
julia --project luhn6_example.jl
```

- `luhn.jl` is a Luhn template program which can take a varying argument `n`, representing the number of digits in the ID. As an example, the following command runs the Luhn program for 6 digits. `n` can range from 2-10.

```bash
julia --project luhn.jl 6
```

Now to replicate the results in Figure 3, run the following commands. It writes the results to a file `dicejl_luhn.txt` that can be used later to create the plots

```bash
./run.sh
```

### WebPPL

The `webppl` folder consists of the following files:

- `luhn6_example.wppl` is an example Luhn file with 6 ID digits. To run it, use the following command

```bash
webppl luhn6_example.wppl --require webppl-timeit
```

- `luhn.wppl` is a template file as above, taking a named argument `n` representing the number of digits in the id. It can be run using the following command. As above, `n` can range from 2-10.

```bash
webppl luhn.wppl --require webppl-timeit -- --n 6
```

- `run.sh` is again a shell script calling `luhn.wppl` with arguments 2-10, and writing the results to a file `webppl_luhn.txt`.

### Psi

The `psi` folder contains 4 files:

- `luhn4_example.psi` is an example Luhn file with 4 ID digits. This file can be run with the following command:

```bash
psi luhn4_example.psi
psi --dp luhn4_example.psi
``` 

As there is no built-in timing utility, we used the Bash `time` command to measure time for Psi. 

- `gen_psi_luhn.py` is a Python script which takes a single parameter `n` and outputs a Psi Luhn program corresponding to `n` ID digits. For example, to generate a Psi program equivalent to `luhn4_example.psi`, call 

```bash
python gen_psi_luhn.py 4 > output.psi
```

- `run.sh` and `run_dp.sh` are two shell scripts which iterate over ID lengths 2-10 to replicate the results in Figure 3. They correspond to Psi and Psi (DP), and output to files `psi_luhn.txt` and `psi_dp_luhn.txt`, respectively. 

```bash
./run.sh
./run_dp.sh
```

Note that for WebPPL and Psi, the provided scripts only run a single trial for each experiment. 

## Plot the Figure

Run the following command to replicate Figure 3. This will recreate `luhn.png` using the results of the latest run.

```bash
python3 luhn.py
```






