#!/bin/bash

if [ -f "leaves.zip" ]; then
    unzip leaves.zip
    echo "leaves.zip has been successfully unzipped."

    find images -type d | while read dir; do
        lower_dir=$(echo "$dir" | tr '[:upper:]' '[:lower:]')
        if [ "$dir" != "$lower_dir" ]; then
            mv "$dir" "$lower_dir"
        fi
    done

    echo "All subdirectories in the images folder have been renamed to lowercase."
else
    echo "leaves.zip file not found."
fi
