import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

x = [2, 3, 4, 5, 6, 7, 8, 9]

dice = []
with open('dicejl/dicejl_luhn.txt', 'r') as f:
    for i in range(2, 10):
        dice.append(float(f.readline()))

webppl = []
with open('webppl/webppl_luhn.txt', 'r') as f:
    for i in range(2, 9):
        webppl.append(float(f.readline())/1000)

psi = []
with open('psi/psi_luhn.txt', 'r') as f:
    for i in range(2, 4):
        psi.append(float(f.readline()))

dp = []
with open('psi/psi_dp_luhn.txt', 'r') as f:
    for i in range(2, 6):
        dp.append(float(f.readline()))
    



# webppl = [0.0048, 0.0228, 0.1288, 0.6068, 5.241, 59.4592, 622.5, 5582.102]
# dp = [0.0762, 1.0858, 16.904, 225.823]
# psi = [0.4122, 56.632, 7966.656]
# dice = [0.0119, 0.0106, 0.0161, 0.018, 0.0212, 0.0291, 0.0372, 0.0528]

fig = plt.figure(figsize=(6,3))

for num in range(1, 2):
    print(num)
    (x, imp_klds, col_klds, gibbs_klds, dice) = (x, webppl, dp, psi, dice)
    ax = fig.add_subplot(1,1,num)
    ax.plot(range(len(imp_klds)),imp_klds,label="WebPPL",linewidth=4, linestyle='dashed')
    ax.plot(range(len(col_klds)),col_klds,label="Psi (DP)",linewidth=4, linestyle='dotted')
    ax.plot(range(len(gibbs_klds)),gibbs_klds,label="Psi",linewidth=4, linestyle="dashdot")
    ax.plot(range(len(dice)),dice,label="Alea.jl",linewidth=4)
    ax.set_xlabel("# Digits",fontsize=24)
    ax.set_yscale('log')
    ax.set_title("Runtime for varying ID digits", fontsize=24, y = -0.40, pad = -14)
    # ax.set_figwidth(2)
    # ax.set_figheight(2)
    if(num==1):
        ax.set_ylabel("Runtime(s)",fontsize=24)
    # ax.set_yticks([-1,-5,-10])
    ax.tick_params(axis='both', which='major', labelsize=24)
    if(num==1):
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.78), ncol=2, fontsize=24)    
    plt.subplots_adjust(wspace=0.3)
	
fig.savefig('luhn.png',bbox_inches='tight',dpi=200)