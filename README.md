# PCC-mPFC Functional Connectivity in Male Autistic Youth: An ABIDE I+II Study

Analysis code and derived data for the manuscript:

> **Li Y.** Small PCC-mPFC hypoconnectivity in male autistic youth: pooled ABIDE evidence with ABIDE I-driven effects and limited clinical relevance. *Autism Research* (under review).

## Repository Structure

```
├── code/          Analysis scripts (Python)
├── data/          Derived connectivity values (CSV)
└── figures/       Manuscript figures
```

## Data

The ABIDE I and ABIDE II raw neuroimaging data are available from the [International Neuroimaging Data-sharing Initiative](http://fcon_1000.projects.nitrc.org/indi/abide/).

This repository contains only *derived* connectivity values (PCC-mPFC Fisher Z-transformed correlation values) and analysis scripts sufficient to reproduce all reported results.

## Analysis Overview

- Seed-based PCC-mPFC functional connectivity
- Primary OLS model with site fixed effects
- 10 sensitivity specifications including LOSO, mixed-effects, multi-radius
- Connection specificity (anterior insula control)
- Clinical correlation (ADOS severity)

## Requirements

See `code/requirements.txt`.

## Contact

Yaowu Li — hanserss0007@gmail.com
