#!/usr/bin/env python3

import os
import subprocess

# Paths
shader_dir = "resources/shaders"
output_dir = os.path.join(shader_dir, "spv")

# Supported shader extensions
shader_extensions = [".rchit", ".rmiss", ".rgen", ".rint"]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through shader files and compile them
for filename in os.listdir(shader_dir):
    # Check for supported shader extensions
    if any(filename.endswith(ext) for ext in shader_extensions):
        shader_path = os.path.join(shader_dir, filename)
        output_path = os.path.join(output_dir, f"{filename}.spv")

        try:
            # Compile the shader using glslc
            subprocess.run(
                ["glslc", shader_path, "--target-spv=spv1.6", "-o", output_path],
                check=True
            )
            print(f"Compiled: {shader_path} -> {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {shader_path}: {e}")
