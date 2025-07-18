#!/usr/bin/env python3
"""
Parse a Julia script for `trixi_include` calls and generate a commands file.

Usage:
    python generate_commands.py input.jl output.txt

Each line in the output will have two fields separated by a tab:
    <threads_or_X>\t<trixi_include(...)>

Lines preceded by `# TRIXIP: SEQUENTIAL` will use `1`.
All other lines emit `X`, meaning “use default threads” in the Bash script.
"""
import os
import sys
import argparse

def parse_trixi_includes(lines):
    commands = []
    seq_flag = False
    buffer = []
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        # detect sequential marker
        if stripped.startswith('#') and 'TRIXIP: SEQUENTIAL' in stripped:
            seq_flag = True
            continue

        # start of a trixi_include block
        if 'trixi_include' in stripped and '(' in stripped:
            buffer = [stripped]
            paren_count = stripped.count('(') - stripped.count(')')
            if paren_count == 0:
                tag = '1' if seq_flag else 'X'
                commands.append((tag, buffer[0]))
                seq_flag = False
                buffer = []
            continue

        # continuing a multi-line trixi_include
        if buffer:
            buffer.append(stripped)
            paren_count += stripped.count('(') - stripped.count(')')
            if paren_count == 0:
                cmd = ' '.join(buffer).rstrip(',')
                tag = '1' if seq_flag else 'X'
                commands.append((tag, cmd))
                seq_flag = False
                buffer = []
            continue

    return commands

def main():
    parser = argparse.ArgumentParser(
        description="Generate a commands file with 'X' or '1' for trixi_include calls."
    )
    parser.add_argument('input', help='Path to the Julia script (e.g. validation.jl)')
    parser.add_argument('output', help='Path to write the commands file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        lines = f.readlines()

    commands = parse_trixi_includes(lines)

    with open(args.output, 'w') as out:
        for tag, cmd in commands:
            out.write(f"{tag}\t{cmd}\n")

    print(f"Wrote {len(commands)} commands to {args.output}")

if __name__ == '__main__':
    main()
