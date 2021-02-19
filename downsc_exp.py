import numpy as np
import sys
from ip3_ryr_ca1_spine_class_file import *

[gN, nRyR, nip3, tau_r, vsp] = sys.argv[1].split("_")
[gN, nRyR, nip3, tau_r, vsp] = [float(gN), int(nRyR), int(nip3), float(tau_r), float(vsp)]

sp1 = ca1_spine(gN, nRyR, nip3, tau_r, vsp) ## Er-bearing spine
sp0 = ca1_spine(gN, 0, 0, 100, vsp) ## Er- spine

sol = sp1.do_rdp(1,10)
sol0 = sp0.do_rdp(1,10)

####### extracting CaM, ryr open probab, ip3 open probab ########
camTr = [np.sum(sol[i,13:20]) for i in range(sol.shape[0])]
camTr0 = [np.sum(sol0[i,13:20]) for i in range(sol0.shape[0])]
o1o2 = [(sol[i,-4] + sol[i,-5]) for i in range(sol.shape[0])]

#################### saving into a file #####################
timeCoursef = "downsc_rdp_r{}_ip3{}_tau{}_f{}_vinit{}.csv".format(nRyR, nip3, tau_r, 1, round(vsp,2))
timeCoursef0 = "downsc_rdp_r{}_ip3{}_f{}_vinit{}.csv".format(0, 0, 1, round(vsp,2))
dvf = "downsc_dv_v.csv"

## Order: w, ca, caer, cam, o1+o2, ip3, h -- ER bearing spine
## Order: w, ca, caer, cam -- ER - spine

for i in range(sol.shape[0]):
    with open(timeCoursef, "a+") as f:
        f.write("{},{},{},{},{},{},{} \n".format(float(sol[i,-17]), float(sol[i,-12]), float(sol[i,-13]), float(camTr[i]), float(o1o2[i]), float(sol[i,22]), float(sol[i,-14])))
        
for i in range(sol0.shape[0]):
    with open(timeCoursef, "a+") as f:
        f.write("{},{},{},{} \n".format(float(sol0[i,-17]), float(sol0[i,-12]), float(sol0[i,-13]), float(camTr0[i])))

## Order: vinit, vf, vinit0, vf0              
with open(dvf, "a+") as f:
    f.write("{},{},{},{} \n".format(float(vsp * sol[0,-17]), float(vsp * sol[-1,-17]), float(vsp * sol0[0,-17]), float(vsp * sol0[-1,-17])))

                