##############################################
# copyright@Zhenhai 20210923
# data processing code for Chip 4 DC transport measurement results
# feel free to use


import Labber
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

moveToOrigin = True
dataselect = -1

# load data
################################################
data_address = 'A:/Labber/20210920_shadowmon_Chip4_DC/2021/09/Data_0922'
logfile = Labber.LogFile(data_address + '/DC_measurement_sweepGate_upWards.hdf5')
Keithley = logfile.getData('keith - SMUA Source Value (Voltage)')
YKGW = logfile.getData('YOKOGAWA_GS200 - Voltage')
YKGW = YKGW - 303 * 10**-6 # 303 * 10**-6 is the dc offset
LI1 = logfile.getData('lock_in_1 - Value')
LI2 = logfile.getData('lock_in_2 - Value')
Agilent = logfile.getData('Agilent_34410A - Voltage')
Agilent = Agilent + 1.393 # there is a DC offset from pre-amp, around 1.393 V


# data processing
################################################
amp_factor = 10**3  # amplification factor of SR 560
Basel_factor = 10**5 # amplification factor of Basel
R_YKGW = 100 * 10**3  # unit Ohm
Idc = YKGW / R_YKGW  # unit Ampere
Idc = Idc * 10**9 #unit nA
Vdc = Agilent / amp_factor # unit Volt
Vdc = Vdc * 10**3 #unit mV
C_electron = 1.602 * 10**-19
h = 6.626 * 10**-34

dV = np.real(LI1) / amp_factor
dI = LI2 / Basel_factor
G = dI / dV
G = G / (2*C_electron**2/h) # unit 2e^2/h
R = dV / dI # unit Ohm
R = R / 1000 # unit kOhm


#start from origin point and data selecting
if moveToOrigin:
	index = np.argmin(np.abs(YKGW[0]))
	Vdc = Vdc[:, index:]
	Idc = Idc[:, index:]
	G = G[:, index:]
	R = R[:, index:]

if dataselect == -1:
	pass
else:
	Vdc = Vdc[:, :dataselect]
	Idc = Idc[:, :dataselect]
	G = G[:, :dataselect]
	R = R[:, :dataselect]


# MAR analysis
###########################################
# eV = 2delta / n_MAR
#delta_Al_wire `= 210 ueV = 0.21meV
# four dips are 0.001610, 0.00734, 0.014095, 0.027525, 0.03497 mV
def MAR(Rn, delta, offset):
	return 2.0*delta * Rn + offset

dips = np.asarray([0.001610, 0.00734, 0.014095, 0.027525, 0.03497]) # unit mV
R_n = 1.0 / np.arange(5, 0, -1)
pfit, pcov = curve_fit(MAR, R_n, dips) # unit meV
plt.figure()
fit_n = np.arange(0, 1.0, 0.01)
fit_V = MAR(fit_n, *pfit)
plt.plot(R_n, dips, 'o')
plt.plot(fit_n, fit_V, 'b')
plt.show()





# plotting
###########################################
plt.figure()
for i in [60]:
	plt.subplot(131)
	plt.plot(Vdc[i], G[i], label='gate {}'.format(np.round(Keithley[i, 0], 2)))
	plt.xlabel('Vdc[mV]')
	plt.ylabel('G[2e^2/h]')
	plt.legend()
	plt.subplot(132)
	plt.plot(Idc[i], R[i])
	plt.xlabel('Idc[nA]')
	plt.ylabel('R[kOhm]')
	plt.subplot(133)
	plt.plot(Idc[i, :-100], Vdc[i, :-100])
	plt.xlabel('Idc[nA]')
	plt.ylabel('Vdc[mV]')
plt.show()

