#!/usr/bin/env python3
"""
Parse a Julia script for `trixi_include` calls and generate a commands file.

Usage:
    python generate_commands.py input.jl output.txt

Each line in the output will have two fields separated by a tab:
    <threads>\t<trixi_include(...)>

Lines preceded by a comment `# TRIXIP: SEQUENTIAL` will use threads=1.
Otherwise, they use the default threads from the environment variable
`TRIXI_SUBMIT_THREADS` (or 4 if unset).
"""
import os
import sys
import argparse


def parse_trixi_includes(lines, default_threads):
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

        # start of a trixi_include block
        if 'trixi_include' in stripped and '(' in stripped:
            buffer = [stripped]
            paren_count = stripped.count('(') - stripped.count(')')
            # if it's a single-line call
            if paren_count == 0:
                threads = 1 if seq_flag else default_threads
                commands.append((threads, buffer[0]))
                seq_flag = False
                buffer = []
            continue

        # continuing a multi-line trixi_include
        if buffer:
            buffer.append(stripped)
            paren_count += stripped.count('(') - stripped.count(')')
            if paren_count == 0:
                # join into one line, remove trailing commas/spaces
                cmd = ' '.join(buffer).rstrip(',')
                threads = 1 if seq_flag else default_threads
                commands.append((threads, cmd))
                seq_flag = False
                buffer = []
            continue

        # otherwise ignore

    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Generate a commands file with thread counts for trixi_include calls."
    )
    parser.add_argument('input', help='Path to the Julia script (e.g. validation.jl)')
    parser.add_argument('output', help='Path to write the commands file')
    args = parser.parse_args()

    default_threads = int(os.environ.get('TRIXI_SUBMIT_THREADS', '4'))

    with open(args.input, 'r') as f:
        lines = f.readlines()

    commands = parse_trixi_includes(lines, default_threads)

    with open(args.output, 'w') as out:
        for threads, cmd in commands:
            out.write(f"{threads}\t{cmd}\n")

    print(f"Wrote {len(commands)} commands to {args.output}")


if __name__ == '__main__':
    main()
