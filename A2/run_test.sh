#!/bin/bash

# Array of input file names
input_files=("checkers1.txt" "checkers2.txt" "checkers3.txt" "checkers4.txt")  # Add more input file names as needed

# Loop through the input file names
for input_file in "${input_files[@]}"; do
    output_file="${input_file%.txt}.out"  # Construct the output file name based on the input file name

    # Run the command
    python3 checkers.py --inputfile "$input_file" --outputfile "$output_file"
done