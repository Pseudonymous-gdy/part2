#!/usr/bin/env python3
# run_tests.py

import sys
import json
import numpy as np
import importlib

def parse_input_str(s: str):
    """
    Parse the multi-line input string into M, N, K, T, A, G
    matching the logic in your classroom runner.
    """
    lines = s.splitlines()
    M, N, K, T = map(int, lines[0].split())

    A = [list(map(int, line.split())) for line in lines[1 : 1 + M]]
    A = np.array(A)

    G = [list(map(int, line.split())) for line in lines[1 + M : 1 + 2 * M]]
    G = np.array(G)

    return M, N, K, T, A, G

def main():
    # 1) Command‐line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <function_name> [tests.json]")
        sys.exit(1)
    func_name = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else "tests.json"

    # 2) Dynamically load solution.py and the requested function
    try:
        solution = importlib.import_module("solution")
    except ImportError:
        print("Error: Could not import solution.py")
        sys.exit(1)

    if not hasattr(solution, func_name):
        print(f"Error: function '{func_name}' not found in solution.py")
        sys.exit(1)
    func = getattr(solution, func_name)

    # 3) Load test cases from JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read '{json_path}': {e}")
        sys.exit(1)

    # 4) Run each test
    total = len(tests)
    passed = 0

    for idx, case in enumerate(tests, start=1):
        inp = case["input"]
        expected = case["expected_output"].strip()

        try:
            M, N, K, T, A, G = parse_input_str(inp)
            result = func(M, N, K, T, A, G)
            output = str(result).strip()
            success = (output == expected)
        except Exception as e:
            output = f"__EXCEPTION__: {e}"
            success = False

        if success:
            print(f"[✓] Test {idx} passed")
            passed += 1
        else:
            print(f"[✗] Test {idx} failed")
            print("---- Input ----")
            print(inp)
            print("---- Expected ----")
            print(expected)
            print("---- Got ----")
            print(output)

    # 5) Summary
    print(f"\n{passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
