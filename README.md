# gr-PR — Passive Radar GNU Radio OOT Modules 

>GNU Radio OOT modules implementing a multichannel DVB-T passive radar with DoA using KrakenSDR made as a part of a bachelour thesis


![Range-Doppler plot](docs/images/Picture_rd_plot+cfar+doa+rec.png)

---

## Overview

**gr-PR** is a collection of GNU Radio Python blocks for building a
**Passive radar** receiver. Passive radar detects targets 
by using existing broadcast as Illuminators of Oppertunity to illuminate targets (**IoO**) — currently tested against **DVB-T** 

This repository was made as part of bachelour thesis 
"**GPU-accelerated passive DVB-T radar system with direction-of-arrival estimation in GNU Radio**"
It consists of diffrent implementations of these algorithms, in diffrent ways (combinations and using NumPy and CuPy)
- ECA (Extensive Cancellation Algorthim)
- CAF (Cross-Ambiguity function)
- CFAR (Constant False Alarm Rate)
- DoA (Direction of Arrival)
<!-- TODO: add a link to the thesis here -->

---

## Signal-processing pipeline

![5channel GR flowgraph](docs/images/Picture_5ch_test_flow.png)

## Hardware setup
![Schematic of the practical test setup](docs/images/Picture_hardware_setup.png)
## Testing setup
![Picture of the practical test setup](docs/images/picture_test_setup.png)
## Known errors - bottlenecks
 - The frame rate is quite bad, only 1 in 5 CPI vectors were processed (discarding rest), in current testing, main culprit is thougth to be the rd_plot+cfar+doa visual processing thread
 - The ECA blocks still leave some residuals in 0 doppler
   

