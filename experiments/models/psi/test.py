# STILL MOSTLY UNTESTED
# REMOVED THIS WHEN DONE


import subprocess
from statistics import median
import csv

# based off of Ryan's microbenchmark test.py 
OUTPUT_FILENAME = "psi_results.csv"

# command to run psi
# replace with path if necessary
PSI_COMMAND = "psi"
TIME_COMMAND = "/usr/bin/time"

TESTS = [
    "book", 
    "caesar_large",
    "caesar_medium",
    "caesar_small",
    "disease_medium",
    "disease_small",
    "fw_medium",
    "fw_small",
    "gcd_large",
    "gcd_medium",
    "gcd_small",
    "linext_large",
    "linext_medium",
    "linext_small",
    "luhn_medium",
    "luhn_small",
    "radar",
    "ranking_large",
    "ranking_medium",
    "ranking_small"
]

TIMEOUT = 3600 # 60*60 = 1 hour
RUNS = 5

def time_test(name, timeout, runs, use_dp=False):
    # for psi, we use elapsed system time 
    time_cmd= [TIME_COMMAND, "-f", "%e"]
    psi_cmd = [PSI_COMMAND, f"{test}.psi",
               "--dp" if use_dp else ""]
    
    cmd = time_cmd + psi_cmd
    try:
        times = []
        for _ in range(RUNS):
            run = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout = TIMEOUT
            )

            if run.returncode == 137:
                # out of memory
                return "x"
            times.append(float(run.stderr))
        return median(times) # note: result in s
    except subprocess.TimeoutExpired:
        # timeout
        return "x"


results = []
with open(OUTPUT_FILENAME, "w") as output_file:
    output_file.write("test,psi(s),psi_dp(s)\n")

for test in TESTS:
    result_no_dp = time_test(test, TIMEOUT, RUNS)
    result_dp = time_test(test, TIMEOUT, RUNS, use_dp=True)
    with open(OUTPUT_FILENAME, "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([test, result_no_dp, result_dp])





