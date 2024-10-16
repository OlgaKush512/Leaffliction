#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide the path to the zip file."
    exit 1
fi

zip_file="$1"

if [ -f "$zip_file" ]; then
    unzip "$zip_file"
    echo "$zip_file has been successfully unzipped."

    find images -type d | while read dir; do
        lower_dir=$(echo "$dir" | tr '[:upper:]' '[:lower:]')
        if [ "$dir" != "$lower_dir" ]; then
            mv "$dir" "$lower_dir"
        fi
    done

    echo "All subdirectories in the images folder have been renamed to lowercase."
else
    echo "$zip_file not found."
fi
