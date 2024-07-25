# Metropolis Algorithm

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# main parameters to be varied/created when running different tests
T1 = 2.5
T2 = 2.5
p1 = 0.2 # if either of these are set to 1: initial state completely spin down
p2 = 0.8 # set to 0: initial state completely spin up
seed1 = 42
seed2 = 314159
sweeps = 400
```

```python
# Ising model with metropolis kinetics.

# creating initial random arrays
maxDim = 50
vol = maxDim ** 2
one_over_vol = 1.0 / vol
one_over_sweeps = 1.0 / sweeps
N = sweeps * vol

rs1 = np.random.RandomState(seed1)
rs2 = np.random.RandomState(seed2)
spins1 = np.zeros((maxDim, maxDim))
spins2 = np.zeros((maxDim, maxDim))
for i in range(0, maxDim):
  for j in range(0, maxDim):
    if rs1.random() < p1:
      spins1[i,j] = -1
    else:
      spins1[i,j] = 1
    if rs2.random() < p2:
      spins2[i,j] = -1
    else:
      spins2[i,j] = 1


# function to compute change in energy of potential spin flip
def pot_dEfunction(spins, i, j):
  left = spins[i, (j - 1) % spins.shape[1]]
  right = spins[i, (j + 1) % spins.shape[1]]
  top = spins[(i - 1) % spins.shape[0], j]
  bottom = spins[(i + 1) % spins.shape[0], j]
  return 2 * spins[i, j] * (left + right + top + bottom)

# function to calculate Hamiltonian of whole system
def hamiltonian(spins):
  Ham = 0
  for i in range(spins.shape[0]):
    for j in range(spins.shape[1]):
      left = spins[i, (j - 1) % spins.shape[1]]
      right = spins[i, (j + 1) % spins.shape[1]]
      top = spins[(i - 1) % spins.shape[0], j]
      bottom = spins[(i + 1) % spins.shape[0], j]
      Ham += -spins[i, j] * (left + right + top + bottom)
  return Ham / 2

# metropolis algorithm
def metropolis(spins, T, N, randomState):
  E_o = hamiltonian(spins)
  E = E_o
  mag_o = np.sum(spins)
  mag = mag_o

  energy_axis = np.zeros(sweeps)
  energy_axis[0] = E_o
  mag_axis = np.zeros(sweeps)
  mag_axis[0] = mag_o
  time_axis = np.arange(sweeps)      # separate time axis for each simulation

  for sweep in range(sweeps):
    # start of metropolis iterations
    for iteration in range(vol):
      i = randomState.choice(range(spins.shape[0]))   # randomly select row
      j = randomState.choice(range(spins.shape[1]))   # randomly select column
      pot_dE = pot_dEfunction(spins, i, j)
      if pot_dE < 0:                                # swap check
        spins[i, j] = -spins[i, j]
        mag += 2 * spins[i, j]
        E += pot_dE
      elif randomState.random() < np.exp(-pot_dE / T):  # "unique aspect"
        spins[i, j] = -spins[i, j]
        mag += 2 * spins[i, j]
        E += pot_dE
    # end of metropolis iterations
    # (therefore the following happens every 'sweep')

    energy_axis[sweep] = E * one_over_vol   # per site
    mag_axis[sweep] = mag * one_over_vol    # per site

  return time_axis, energy_axis, mag_axis # spec_heat_arr, mag_susc_arr
```

```python
# applying metropolis algorithm to two systems differing by
# initial state & seed

time_axis1, energy_axis1, mag_axis1 = metropolis(spins1, T1, N, rs1)
time_axis2, energy_axis2, mag_axis2 = metropolis(spins2, T2, N, rs2)

# displaying first 2D Ising model
print()
plt.figure(figsize=(4, 3))
plt.imshow(spins1, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title("1st Monte Carlo simulation of a 2D Ising model \
using the Metropolis algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature: {T1}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed1}', color='black', fontsize=10)
plt.show()

# displaying second 2D Ising model
print()
plt.figure(figsize=(4, 3))
plt.imshow(spins2, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title("2nd Monte Carlo simulation of a 2D Ising model \
using the Metropolis algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature: {T2}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed2}', color='black', fontsize=10)
plt.show()

# plotting graphs of (time vs energy) & (time vs magnetisation)
# for both simulations
print()
plt.plot(time_axis1, energy_axis1, label='Energy1')
plt.plot(time_axis2, energy_axis2, label='Energy2')
plt.plot(time_axis1, mag_axis1, label='magnetisation1')
plt.plot(time_axis2, mag_axis2, label='magnetisation2')
plt.xlabel('Time')
plt.ylabel('Energy/Magnetisation')
plt.title(f'Energy and Magnetisation vs Time at Temperatures {T1} and {T2}')
plt.legend()
plt.grid(True)
plt.show()

# note: graph in report, for results of above code, uses: p1 = 0.2, p2 = 0.8
```


```python
# phase transition for an Ising model with metropolis kinetics.

T3 = np.arange(0.2, 5.2, 0.2) # ie T3 is an array of values from 0.5 to 5.0
p3 = 0.5                      # in steps of 0.2
seed3 = 100

rs3 = np.random.RandomState(seed3)
spins3 = np.zeros((maxDim, maxDim))
for i in range(0, maxDim):
  for j in range(0, maxDim):
    if rs3.random() < p3:
      #could be np.random.RandomState(seed3).random() < p3:
      spins3[i,j] = -1
    else:
      spins3[i,j] = 1

# a 3rd ising model has now been randomly generated.

def specific_heat_fn(avg_xsq, avg_x, T, vol):
  c = (1/(T**2)) * vol * (avg_xsq - avg_x**2)
  return c

def magnetic_susc_fn(avg_xsq, avg_x, T, vol):
  chi = (1/T) * vol * (avg_xsq - avg_x**2)
  return chi

# functions defined along with equations in report
```

```python
vals_taken = 30
import csv
file_name = "simulation_results.csv"

with open(file_name, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['T', 'E', 'm', 'c', 'X'])
  for i in T3:
    # for every temperature, we calculate one value for energy,
    # magnetisation, specific heat, and magnetic susc

    #calculate arrays, then turn into single average value
    time_axis3, energy_axis3, mag_axis3 = metropolis(spins3, i, N, rs3)

    avg_E = np.mean(energy_axis3[-vals_taken:]) # single value obtained
    sq_sum = 0
    for j in energy_axis3[-vals_taken:]:
      sq_sum += j**2
    avg_Esq = sq_sum / vals_taken

    avg_mag = np.mean(mag_axis3[-vals_taken:])  # single value obtained
    sq_sum = 0
    for k in mag_axis3[-vals_taken:]:
      sq_sum += k**2
    avg_magsq = sq_sum / vals_taken

    c = specific_heat_fn(avg_Esq, avg_E, i, vol)      # single value obtained
    chi = magnetic_susc_fn(avg_magsq, avg_mag, i, vol)   # single value obtained

    # POTENTIAL ISSUE IDENTIFIED WITHIN CODE
    # equilibrium of any one quantity will not occur at the same point for
    # different temperatures, (as is incorrectly prdeicted by setting the
    # vals_taken to be constant)
    # therefore vals_taken should not be a constant
    # i predict that a solution to this would involve calculating EQUILIBRATION
    # TIME for each simulation; this is beyond the scope of this project and
    # hence will not be implemented

    # The following code displays our ising model at each integer value of temp

    if i % 1.0 == 0:
      print()
      plt.figure(figsize=(4, 3))
      plt.imshow(spins3, cmap='coolwarm', interpolation='nearest')
      plt.colorbar().remove()
      plt.xticks([])
      plt.yticks([])
      plt.title(f"Monte Carlo simulation ({int(i)}) of a 2D COP Ising model \
      using the Met algorithm")
      plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
      plt.text(-45, 5, f'Temperature: {i}', color='black', fontsize=10)
      plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
      plt.text(-45, 11, f'Seed: {seed3}', color='black', fontsize=10)
      plt.show()

    # write single values into row, corresponding to its T
    # time axis not needed
    writer.writerow([i, avg_E, avg_mag, c, chi])

# at this point we have obtained results for average energy, average
# magnetisation, specific heat and magnetic susceptibility, for different values
# of temperature
# and written these results into file_name = "simulation_results.csv"



# the following code reads this csv file and then plots four graphs

import pandas as pd

df = pd.read_csv('simulation_results.csv')

# writer.writerow(['T', 'E', 'm', 'c', 'X'])


plt.plot(df['T'], df['E'], marker = 'o', label = 'Energy')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title('Temperature vs. Energy')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df['T'], df['m'], marker = 'o', label = 'magnetisation')
plt.xlabel('Temperature')
plt.ylabel('magnetisation')
plt.title('Temperature vs. magnetisation')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df['T'], df['c'], marker = 'o', label = 'specific heat')
plt.xlabel('Temperature')
plt.ylabel('specific heat')
plt.title('Temperature vs. specific heat')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df['T'], df['X'], marker = 'o', label = 'magnetic susceptibility')
plt.xlabel('Temperature')
plt.ylabel('magnetic susceptibility')
plt.title('Temperature vs. magnetic susceptibility')
plt.grid(True)
plt.legend()
plt.show()
```

# Glauber Algorithm

```python
# Ising model with glauber kinetics.


def glauber(spins, T, N, randomState):
  E_o = hamiltonian(spins)
  E = E_o
  mag_o = np.sum(spins)
  mag = mag_o

  energy_axis = np.zeros(sweeps)
  energy_axis[0] = E_o
  mag_axis = np.zeros(sweeps)
  mag_axis[0] = mag_o
  time_axis = np.arange(sweeps)      # separate time axis for each simulation

  for sweep in range(sweeps):
    # start of glauber iterations
    for iteration in range(vol):
      i = randomState.choice(range(spins.shape[0]))   # randomly select row
      j = randomState.choice(range(spins.shape[1]))   # randomly select column
      pot_dE = pot_dEfunction(spins, i, j)
      if pot_dE < 0:                                # swap check
        spins[i, j] = -spins[i, j]
        mag += 2 * spins[i, j]
        E += pot_dE
      elif randomState.random() < (np.exp(-pot_dE/T)/(1 + np.exp(-pot_dE/T))):
        spins[i, j] = -spins[i, j]              # "unique aspect" ^^^
        mag += 2 * spins[i, j]
        E += pot_dE
    # end of Glauber iterations
    # (therefore the following happens every 'sweep')

    energy_axis[sweep] = E * one_over_vol   # per site
    mag_axis[sweep] = mag * one_over_vol    # per site

  return time_axis, energy_axis, mag_axis
```

```python
# we want to be able to compare the two algorithms and thus we use the
# exact same two intial states as before

# therefore we do not need to redefine any new temperatures, seeds, random
# states, or choice probability
# all we have to redefine is two spin arrays, because the spins1 and
# spins2 have already been messed with and used in the metropolis demo

spins4 = np.zeros((maxDim, maxDim))
spins5 = np.zeros((maxDim, maxDim))
for i in range(0, maxDim):
  for j in range(0, maxDim):
    if rs1.random() < p1:
      spins4[i,j] = -1
    else:
      spins4[i,j] = 1
    if rs2.random() < p2:
      spins5[i,j] = -1
    else:
      spins5[i,j] = 1

time_axis4, energy_axis4, mag_axis4 = glauber(spins4, T1, N, rs1)
time_axis5, energy_axis5, mag_axis5 = glauber(spins5, T2, N, rs2)



# displaying first 2D Ising model
plt.figure(figsize=(4, 3))
plt.imshow(spins4, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title("1st Monte Carlo simulation of a 2D Ising model \
using the Glauber algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature: {T1}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed1}', color='black', fontsize=10)
plt.show()

# displaying second 2D Ising model
print()
plt.figure(figsize=(4, 3))
plt.imshow(spins5, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title("2nd Monte Carlo simulation of a 2D Ising model \
using the Glauber algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature: {T2}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed2}', color='black', fontsize=10)
plt.show()

# plotting graphs of time vs energy & magnetisation for both simulations
print()
plt.plot(time_axis4, energy_axis4, label='Energy4')
plt.plot(time_axis5, energy_axis5, label='Energy5')
plt.plot(time_axis4, mag_axis4, label='magnetisation4')
plt.plot(time_axis5, mag_axis5, label='magnetisation5')
plt.xlabel('Time')
plt.ylabel('Energy/Magnetisation')
plt.title(f'Energy and Magnetisation vs Time at Temperatures {T1} and {T2}')
plt.legend()
plt.grid(True)
plt.show()
```

```python
# phase transition for an Ising model with glauber kinetics.

# we will be using the same setup as the phase xsn demo with metrop
# therefore all we need redefine is the spins array

spins6 = np.zeros((maxDim, maxDim))
for i in range(0, maxDim):
  for j in range(0, maxDim):
    if rs3.random() < p3:
      spins6[i,j] = -1
    else:
      spins6[i,j] = 1

file_name2 = "simulation_results2.csv"

with open(file_name2, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['T', 'E', 'm', 'c', 'X'])
  for i in T3:
    # for every temperature, we calculate one value for energy, magnetisation, specific heat, and magnetic susc

    #calculate arrays
    time_axis6, energy_axis6, mag_axis6 = glauber(spins6, i, N, rs3)

    # turn arrays into single values
    avg_E = np.mean(energy_axis6[-vals_taken:])
    sq_sum = 0
    for j in energy_axis6[-vals_taken:]:
      sq_sum += j**2
    avg_Esq = sq_sum / vals_taken

    avg_mag = np.mean(mag_axis6[-vals_taken:])
    sq_sum = 0
    for k in mag_axis6[-vals_taken:]:
      sq_sum += k**2
    avg_magsq = sq_sum / vals_taken


    c = specific_heat_fn(avg_Esq, avg_E, i, vol)
    chi = magnetic_susc_fn(avg_magsq, avg_mag, i, vol)

    # write single values into row, corresponding to its T
    # time axis not needed

    writer.writerow([i, avg_E, avg_mag, c, chi])

import pandas as pd

df2 = pd.read_csv('simulation_results2.csv')

# writer.writerow(['T', 'E', 'm', 'c', 'X'])


plt.plot(df2['T'], df2['E'], marker = 'o', label = 'Energy')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title('Temperature vs. Energy')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df2['T'], df2['m'], marker = 'o', label = 'magnetisation')
plt.xlabel('Temperature')
plt.ylabel('magnetisation')
plt.title('Temperature vs. magnetisation')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df2['T'], df2['c'], marker = 'o', label = 'specific heat')
plt.xlabel('Temperature')
plt.ylabel('specific heat')
plt.title('Temperature vs. specific heat')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df2['T'], df2['X'], marker = 'o', label = 'magnetic susceptibility')
plt.xlabel('Temperature')
plt.ylabel('magnetic susceptibility')
plt.title('Temperature vs. magnetic susceptibility')
plt.grid(True)
plt.legend()
plt.show()
```
# Kawasaki Algorithm

```python
# COP Ising model with kawasaki kinetics

# note to self
# if (i1, j1) == (i2, j2):    # if the indexes represent the same location
# if spins[i1,j1] == spins[i2,j2]:  # if the indexes are the same value



# function to calculate dE
def pot_dE_fn_Kaw(spins, i1, j1, i2, j2):
  left_p = spins[i1,(j1-1) % spins.shape[1]]
  right_p = spins[i1,(j1+1) % spins.shape[1]]
  top_p = spins[(i1-1) % spins.shape[0],j1]
  bottom_p = spins[(i1+1) % spins.shape[0],j1]

  left_q = spins[i2,(j2-1) % spins.shape[1]]
  right_q = spins[i2,(j2+1) % spins.shape[1]]
  top_q = spins[(i2-1) % spins.shape[0],j2]
  bottom_q = spins[(i2+1) % spins.shape[0],j2]

  # the following checks are to make sure of p =/ j and q =/ i.
  # we simply set to zero so that they are not counted within the sum later.
  # note we will only be checking the former, as the latter can be
  # considered a result of the former.

  # another advantage of this method is that we are not changing the actual
  # index's value to 0, we are just changing how it will affect the value of dE

  if (i1,(j1-1) % spins.shape[1]) == (i2,j2):  # leftp = j
    left_p = 0
    right_q = 0
  if (i1,(j1+1) % spins.shape[1]) == (i2,j2):
    right_p = 0
    left_q = 0
  if ((i1-1) % spins.shape[0],j1) == (i2,j2):
    top_p = 0
    bottom_q = 0
  if ((i1+1) % spins.shape[0],j1) == (i2,j2):
    bottom_p = 0
    top_q = 0

  dE = 2 * ( spins[i1,j1] * (left_p + right_p + top_p + bottom_p) + \
              spins[i2,j2] * (left_q + right_q + top_q + bottom_q))

  return dE


# function to carry out the kawasaki algorithm
def kawasaki(spins, T, N, randomState):
  E = hamiltonian(spins)
  energy_axis = np.zeros(sweeps)
  energy_axis[0] = E
  time_axis = np.arange(sweeps)      # separate time axis for each simulation
  for sweep in range(sweeps):
    for iteration in range(vol):
      i1 = randomState.choice(range(spins.shape[0]))
      j1 = randomState.choice(range(spins.shape[1])) # randomly selected site1
      i2 = randomState.choice(range(spins.shape[0]))
      j2 = randomState.choice(range(spins.shape[1])) # randomly selected site2

      if spins[i1,j1] == spins[i2,j2] or (i1,j1) == (i2,j2):
        dE = 0.0
        E += dE
      else:
        dE = pot_dE_fn_Kaw(spins, i1, j1, i2, j2)
        if dE < 0:
          spins[i1, j1] = -spins[i1, j1]
          spins[i2, j2] = -spins[i2, j2]
          E += dE
        elif randomState.random() < (1 / (1 + np.exp(dE / T))):
          spins[i1, j1] = -spins[i1, j1]
          spins[i2, j2] = -spins[i2, j2]
          E += dE
    energy_axis[sweep] = E * one_over_vol   # per site
  return time_axis, energy_axis

# at this point we have defined the kawasaki algorithm itself, and the function
# for finding potential energy change of a spin-exchange (which is of course
# then used within kawasaki)

# creating initial arrays
T7 = 2.5
T8 = 2.5
p7 = 0.5
p8 = 0.5
seed7 = 123
seed8 = 100
rs7 = np.random.RandomState(seed7)
rs8 = np.random.RandomState(seed8)
spins7 = np.zeros((maxDim, maxDim))
spins8 = np.zeros((maxDim, maxDim))
for i in range(0, maxDim):
  for j in range(0, maxDim):
    if rs7.random() < p7:
      spins7[i,j] = -1
    else:
      spins7[i,j] = 1
    if rs8.random() < p8:
      spins8[i,j] = -1
    else:
      spins8[i,j] = 1

# running the algorithm on the arrays defined
# and obtaining values for energy
time_axis7, energy_axis7 = kawasaki(spins7, T7, N, rs7)
time_axis8, energy_axis8 = kawasaki(spins8, T8, N, rs8)

# displaying first 2D COP Ising model
print()
plt.figure(figsize=(4, 3))
plt.imshow(spins7, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title("1st Monte Carlo simulation of a 2D COP Ising model \
using the Kawasaki algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature: {T7}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed7}', color='black', fontsize=10)
plt.show()

# displaying second 2D COP Ising model
print()
plt.figure(figsize=(4, 3))
plt.imshow(spins8, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title("2nd Monte Carlo simulation of a 2D COP Ising model \
using the Kawasaki algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature: {T8}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed8}', color='black', fontsize=10)
plt.show()

# plotting graphs of time vs energy
print()
plt.plot(time_axis7, energy_axis7, label='Energy7')
plt.plot(time_axis8, energy_axis8, label='Energy8')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title(f'Energy vs Time at Temperature {T8}')
plt.legend()
plt.grid(True)
plt.show()
```

```python
# phase separation for an Ising model with kawasaki kinetics.

# we will be using the same setup as the phase xsn demo with metrop
# therefore all we need redefine is the spins array

spins9 = np.zeros((maxDim, maxDim))
for i in range(0, maxDim):
  for j in range(0, maxDim):
    if rs3.random() < p3:
      spins9[i,j] = -1
    else:
      spins9[i,j] = 1

file_name3 = "simulation_results3.csv"

with open(file_name3, 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['T', 'E', 'c'])
  for i in T3:
    # for every temperature, we calculate one value for energy and specific heat

    #calculate arrays
    time_axis9, energy_axis9, = kawasaki(spins9, i, N, rs3)

    # turn arrays into single values
    avg_E = np.mean(energy_axis9[-vals_taken:])
    sq_sum = 0
    for j in energy_axis9[-vals_taken:]:
      sq_sum += j**2
    avg_Esq = sq_sum / vals_taken

    c = specific_heat_fn(avg_Esq, avg_E, i, vol)

    # write single values into row, corresponding to its T
    # time axis not needed

    # graphic of model at T = 1,2,3,4,5
    if i % 1.0 == 0:
      print()
      plt.figure(figsize=(4, 3))
      plt.imshow(spins9, cmap='coolwarm', interpolation='nearest')
      plt.colorbar().remove()
      plt.xticks([])
      plt.yticks([])
      plt.title(f"Monte Carlo simulation ({int(i)}) of a 2D COP Ising model \
      using the Kawasaki algorithm")
      plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
      plt.text(-45, 5, f'Temperature: {i}', color='black', fontsize=10)
      plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
      plt.text(-45, 11, f'Seed: {seed3}', color='black', fontsize=10)
      plt.show()

    writer.writerow([i, avg_E, c])

# we have now obtained results for energy and specific heat for different
# temperatures

import pandas as pd

df3 = pd.read_csv('simulation_results3.csv')

# writer.writerow(['T', 'E', 'c'])

print()
plt.plot(df3['T'], df3['E'], marker = 'o', label = 'Energy')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title('Temperature vs. Energy')
plt.grid(True)
plt.legend()
plt.show()

print()
plt.plot(df3['T'], df3['c'], marker = 'o', label = 'specific heat')
plt.xlabel('Temperature')
plt.ylabel('specific heat')
plt.title('Temperature vs. specific heat')
plt.grid(True)
plt.legend()
plt.show()
```

# Social Applications

```python
# kawasaki APPLICATIONS code
# for the paper: Immigration, integration and ghetto formation by
# Hildegard Meyer-Ortmanns

# the following variables are to be changed for each test
concentration = 0.2
test_T = 1.25
N = 10**7

# the following is initial random array creation with fixed
# proportion/concentration of +1s to -1s
# note migrants denoted by +1, natives denoted by -1
T_c = 2.269
T = test_T * T_c
maxDim = 50
seed = 42
rs = np.random.RandomState(seed)
num_ones = int(concentration * maxDim**2)
num_neg_ones = int(maxDim **2 - num_ones)
spins = np.concatenate([np.ones(num_ones), -np.ones(num_neg_ones)])
np.random.shuffle(spins)
spins = spins.reshape((maxDim, maxDim))

# apply kawasaki algorithm to each test's spec
time_axis, energy_axis = kawasaki(spins, T, N, rs)

# print model for each test
plt.figure(figsize=(4, 3))
plt.imshow(spins, cmap='coolwarm', interpolation='nearest')
plt.colorbar().remove()
plt.xticks([])
plt.yticks([])
plt.title(f"Monte Carlo simulation of a 2D COP Ising model \
using the Kawasaki algorithm")
plt.text(-45, 2, f'Size: {maxDim} by {maxDim}', color='black', fontsize=10)
plt.text(-45, 5, f'Temperature T/Tc: {test_T}', color='black', fontsize=10)
plt.text(-45, 8, f'Iterations: {N}', color='black', fontsize=10)
plt.text(-45, 11, f'Seed: {seed}', color='black', fontsize=10)
plt.text(-45, 14, f'Concentration: {concentration}', color='black', fontsize=10)
plt.show()

# print graph of time vs energy
print()
plt.plot(time_axis, energy_axis, label='Energy')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title(f'Energy vs Time at Temperature T/Tc: {test_T}, Concentration: {concentration}, after {N} Iterations.')
plt.legend()
plt.grid(True)
plt.show()
```