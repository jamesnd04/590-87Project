#!/usr/bin/python3

import os
import sys
from pathlib import Path

def generate_source_txt(root_dir, output_file):
    root_path = Path(root_dir).resolve()
    
    if not root_path.is_dir():
        print(f"Error: {root_dir} is not a valid directory.")
        sys.exit(1)

    # Standard extensions to include; feel free to add more (e.g., .js, .cpp)
    valid_extensions = {'.py', '.java', '.c', '.h', '.html', '.css', '.txt', '.md', '.sh'}

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in root_path.rglob('*'):
            # Skip directories and the output file itself
            if file_path.is_file() and file_path.suffix in valid_extensions:
                if file_path.resolve() == Path(output_file).resolve():
                    continue

                try:
                    relative_path = file_path.relative_to(root_path)
                    
                    # Write the header
                    outfile.write(f"\n{'='*80}\n")
                    outfile.write(f"FILE: {relative_path}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    # Write the content
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")
                        
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    print(f"Done! Source code compiled into {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path> [output_filename]")
        sys.exit(1)

    target_dir = sys.argv[1]
    out_name = sys.argv[2] if len(sys.argv) > 2 else "source_compilation.txt"
    
    generate_source_txt(target_dir, out_name)
