---
name: project_gr_pr
description: gr-PR GNU Radio OOT module — passive radar signal processing pipeline (CAF, DDC, RD plot blocks)
type: project
---

gr-PR is a GNU Radio OOT Python module for DVB-T passive radar.

**Why:** User is building a passive radar receiver using KrakenSDR hardware and needs custom signal processing blocks.

**How to apply:** All blocks use vector I/O (CPI-based processing). Block style: sync_block with complex64 vector inputs sized to cpi_samp. Follow existing patterns in caf.py and PR_caf.block.yml for new blocks.

Current blocks:
- **caf.py** — Cross Ambiguity Function (FFT-based, Doppler × delay output)
- **DDC.py** — Direct-path Disturbance Cancellation (ECA batch Wiener filter)
- **rd_plot.py** — Range-Doppler plot display

Flowgraph pipeline: stream_to_vector → DDC → CAF → rd_plot
