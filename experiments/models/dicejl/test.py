import subprocess
from statistics import median
import csv

# based off of Ryan's microbenchmark test.py 
OUTPUT_FILENAME = "dicejl-results.csv"

TESTS = [
    "book", 
    "caesar_large",
    "caesar_medium",
    "caesar_small",
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
    "ranking_small",
    "triangle_large",
    "triangle_medium",
    "triangle_small",
    "tugofwar"
]

def time_test(name):
    cmd = ["julia", "--project", f"{test}.jl"]
    
    run = subprocess.run(
        cmd, 
        capture_output = True,
        text=True,
    )

    if run.returncode == 137:
        # out of memory
        return "x"
        
    # the Julia BenchmarkTools utility already takes multiple trials
    # result in s
    print(run.stdout)
    print(run.stderr)
    result = float(run.stdout)
    if result > 3600:
        return "x" # timeout
    
    return float(run.stdout)


results = []
with open(OUTPUT_FILENAME, "w") as output_file:
    output_file.write("test,dicejl(s)\n")

for test in TESTS:
    result = time_test(test)
    
    with open(OUTPUT_FILENAME, "a") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([test, result])



    




