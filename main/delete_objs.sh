#!/bin/bash

# Loop through all directories in the current directory
for dir in */ ; do
    # Remove the trailing slash from the directory name
    dir="${dir%/}"

    if [[ "$dir" =~ ^[A-Q0-9] ]]; then
        echo "Deleting directory: $dir"
        rm -rf "$dir"
    fi
done