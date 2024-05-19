#!/usr/bin/env bash

# Sort the jobs.txt file
cat jobs.txt | sort > tmp.txt && mv tmp.txt jobs.txt

# Run the binary
nvprof ./parallel_project ./jobs.txt
