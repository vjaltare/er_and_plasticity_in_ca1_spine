<h1> Biohpysical model of an ER-bearing CA1 dendritic spine </h1>
  
This repository contains computational model of an ER-bearing dendritic spine. This model is written in [![Python 3.7](https://img.shields.io/badge/python-3.7-yellow.svg)](https://www.python.org/downloads/release/python-360/). 

<h2> Acessing the model </h2>

The `ca1_spinemodel.py` file contains the biophysical model of the CA1 dendritic spine along with specific methods which simulate standard plasticity protocols.

<h2> Initializing the spine with Ca2+ handling and ER-related parameters </h2>

- Call the constructor by declaring a parametrized object `ca1_spine(...)`.
- You can initialize the following parameters from the constructor (in order):

|Parameter/Argument|Variable|Default Value + Units|
|--|--|--|
|NMDA Conductance|`g_NMDAR`|675 pS|
|RyR cluster size|`nRyR`|30 (usually anything in 10<sup>1</sup> works) receptors|
|IP3 receptor cluster size|`n_ip3`|50 (usually anything in 10<sup>1</sup> works) receptors|
|Peak SOCC flux|`Vsoce_0`|1000 uM/s|
|ER-refilling time constant|`tau_refill_0`|1 s|
|Spine Volume|`Vspine`|0.06 um<sup>3</sup>|
|Membrane depolarization|`scDep`|0 mV|

- After initialization, the parameters `Vsoce_0` and `tau_refill_0` scale with the ratio of the spine volume to the spine surface area. In the code this scaled value is referred as `Vsoce` and `tau_refill`.

<h2> Performing experiments <i>in-silico</i> </h2>

### Rate-dependent plasticity protocol:
- The rate-dependent plasticity protocol (RDP) stimulates a spine with varying frequencies of presynaptic inputs. 
- Paremeters (in order): input frequency `f_input`, number of inputs `n_inputs`
- In a standard experiment, a 15 min 1 Hz spike train is a way to induce LTD. Thus, 900 spikes is usually used for this experiment.

### Schaffer-Collateral Place Field firing pattern
- Emulates the realistic Schaffer-Collateral place cell firing pattern as described in <a href="https://www.jneurosci.org/content/29/21/6840">Issac et al. J.Neurosci 2009</a>
- Parameters: `beta_pre`, `beta_post`, `tmax`, `f_burst_min`, `f_burst_max`, `max_inputs`
  - `beta_pre`, `beta_post`: average firing rate for pre and postsynaptic APs respectively in Hz
  - `f_burst_min`, `f_burst_max`: min and max frequency of spikes in a burst. Avg freq of each burst is sampled from a uniform distribution [`f_burst_min`, `f_burst_max`]  
  - `max_inputs`: max inputs in a burst. For each burst no. of inputs is sampled from a uniform distribution [1, `max_inputs`]

### Theta-burst protocol
Under construction

## Fantastic biophysical models of signaling proteins and where to find them...



