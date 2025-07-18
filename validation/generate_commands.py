#!/usr/bin/env python3
"""
Parse a Julia script for `trixi_include` calls and generate a commands file.

Usage:
    python generate_commands.py input.jl output.txt

Each line in the output will have two fields separated by a tab:
    <threads_or_X>\t<trixi_include(...)>

Lines preceded by `# TRIXIP: SEQUENTIAL` will use `1`.
All other lines emit `X`, meaning “use default threads” in the Bash script.
Commented-out `trixi_include` calls are ignored entirely.
"""
import argparse
import sys


def parse_trixi_includes(lines):
    commands = []
    seq_flag = False
    buffer = []
    paren_count = 0

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        # detect sequential marker
        if stripped.startswith('#') and 'TRIXIP: SEQUENTIAL' in stripped:
            seq_flag = True
            continue

        # ignore all other commented lines
        if stripped.startswith('#'):
            continue

        # start of a trixi_include call
        if 'trixi_include' in stripped and '(' in stripped:
            buffer = [stripped]
            paren_count = stripped.count('(') - stripped.count(')')
            # single-line call
            if paren_count == 0:
                tag = '1' if seq_flag else 'X'
                commands.append((tag, buffer[0]))
                seq_flag = False
                buffer = []
            continue

        # continuation of multi-line trixi_include
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

        # all other lines ignored

    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Generate a commands file with 'X' or '1' for trixi_include calls."
    )
    parser.add_argument('input', help='Path to the Julia script (e.g. validation.jl)')
    parser.add_argument('output', help='Path to write the commands file')
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        sys.exit(f"Error: input file '{args.input}' not found")

    commands = parse_trixi_includes(lines)

    with open(args.output, 'w') as out:
        for tag, cmd in commands:
            out.write(f"{tag}\t{cmd}\n")

    print(f"Wrote {len(commands)} commands to {args.output}")


if __name__ == '__main__':
    main()