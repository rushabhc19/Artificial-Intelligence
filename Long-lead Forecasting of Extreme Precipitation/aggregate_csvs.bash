#!/bin/bash

# Check that data is available
if [ ! -d 'raw_data' ]; then
    echo "Error, I need 'raw_data' directory that contains CSV files"
    exit 1
fi

# Create aggreg folder if it hasn't been created yet
mkdir -p 'raw_data/aggreg'

# Loop over the variable names
for var in {'iowa','pw','t850','u300','u850','v300','v850','z1000','z300','z500'}
do
    # Do not repeat work
    if [ ! -f "raw_data/aggreg/$var.csv" ]
    then
        # Loop over files for current variable
        for file in raw_data/${var}_*
        do
            # Print current process
            echo "$file >> raw_data/aggreg/$var.csv"

            # Append file contents to the aggregation, skipping header line
            sed -e 1d $file >> raw_data/aggreg/$var.csv
        done
    fi
done