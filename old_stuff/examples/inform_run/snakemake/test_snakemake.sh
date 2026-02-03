#!/bin/bash

snakemake -j1 --rerun-triggers mtime
