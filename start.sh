#!/usr/bin/env bash

# Sort the jobs.txt file
cat jobs.txt | sort > tmp.txt && mv tmp.txt jobs.txt

# Run the binary
./parallel_project ./jobs.txt