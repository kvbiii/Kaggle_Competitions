# Kaggle Competitions Notebooks Repository

Welcome to the Kaggle Competitions Notebooks Repository! This repository contains a collection of Jupyter notebooks from various Kaggle competitions.

## Overview

This repository serves as a central hub for sharing and collaborating on notebooks related to Kaggle competitions. Whether you're a beginner looking to learn from others' solutions or an experienced data scientist sharing your insights, you'll find valuable resources here.

Kill any processes that are using the GPU before running the notebook to avoid out-of-memory errors. You can do this by running the following command in your terminal:

```bash
sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
```
