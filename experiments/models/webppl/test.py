import subprocess
from statistics import median
import csv

# based off of Ryan's microbenchmark test.py 
OUTPUT_FILENAME = "webppl_results.csv"

# command to run webppl
# replace with path if necessary
WEBPPL_COMMAND = "webppl"

TESTS = [
    "book", 
    "caesar_large",
    "caesar_medium",
    "caesar_small",
    "disease_large",
    "disease_medium",
    "disease_small",
    "fw_large",
    "fw_medium",
    "fw_small",
    # "gcd_large",
    # "gcd_medium",
    # "gcd_small",
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

def time_test(name, timeout, runs):
    cmd = [WEBPPL_COMMAND, f"{test}.wppl", "--require", "webppl-timeit"]
    try:
        times = []
        for _ in range(RUNS):
            run = subprocess.run(
                cmd, 
                capture_output = True,
                text=True,
                timeout = TIMEOUT
            )

            if run.returncode == 137:
                # out of memory
                return "x"
            
            times.append(int(run.stdout))
        return median(times) # note: result in ms
    except subprocess.TimeoutExpired:
        # timeout
        return "x"


results = []
with open(OUTPUT_FILENAME, "w") as output_file:
    output_file.write("test,webppl(ms)\n")

for test in TESTS:
    result = time_test(test, TIMEOUT, RUNS)
    
    with open(OUTPUT_FILENAME, "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([test, result])



    




