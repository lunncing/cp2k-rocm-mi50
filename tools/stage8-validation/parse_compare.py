#!/usr/bin/env python3
"""Strict CP2K molecular comparison; timing-only reports never grant acceptance."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
NUMERIC = re.compile(rf"^{NUMBER}$")
VRAM_CAVEAT = ("Sampled VRAM is a lower bound on the true peak. Sampling may miss transient "
               "allocations; timings and memory samples do not establish molecular acceptance.")


def decimal(raw):
    if not NUMERIC.fullmatch(raw):
        raise ValueError(f"Invalid or non-finite numeric value: {raw}")
    value = Decimal(raw.replace("D", "E").replace("d", "e"))
    if not value.is_finite():
        raise ValueError(f"Non-finite numeric value: {raw}")
    return value


def quantity(raw, unit):
    return {"raw": raw, "value": str(decimal(raw)), "unit": unit}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def last_invocation(text):
    starts = list(re.finditer(r"^.*PROGRAM STARTED AT.*$", text, re.MULTILINE))
    if not starts:
        raise ValueError("No PROGRAM STARTED AT invocation boundary")
    complete = []
    for index, start in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        segment = text[start.start():stop]
        ending = re.search(r"^.*PROGRAM ENDED AT.*$", segment, re.MULTILINE)
        if ending:
            complete.append((index, start.start(), segment[:ending.end()]))
    if not complete:
        raise ValueError("No complete, normally terminated invocation")
    index, offset, segment = complete[-1]
    return segment, {"selected_invocation": index + 1, "invocations_seen": len(starts),
                     "complete_invocations": len(complete), "start_line": text[:offset].count("\n") + 1,
                     "ignored_later_incomplete_invocations": len(starts) - index - 1}


def stats(updates):
    times = [float(decimal(item["time"]["raw"])) for item in updates]
    return {"count": len(times), "sum_seconds": sum(times),
            "mean_seconds": statistics.mean(times) if times else None,
            "median_seconds": statistics.median(times) if times else None,
            "min_seconds": min(times) if times else None,
            "max_seconds": max(times) if times else None}


def parse_log(path, natoms, timing_only=False, rks=False):
    if natoms < 1:
        raise ValueError("natoms must be positive")
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    invocation, selection = last_invocation(text)
    result = {"path": str(path.resolve()), "sha256": sha256(path), "selection": selection,
              "normal_termination": True, "natoms_expected": natoms, "scf_blocks": [],
              "timings": [], "timing_unit": "seconds", "timing_column": "TOTAL TIME maximum (last column)",
              "nested_timing_policy": "Inclusive per-routine times; never sum nested timers.",
              "electron_counts": {}, "energy": None, "integrated_spin": None,
              "integrated_absolute_spin": None, "forces": [], "vram_samples": [],
              "vram_caveat": VRAM_CAVEAT}
    if rks:
        result["spin_validation"] = {"status": "not_applicable_RKS",
                                     "banner_evidence": [], "spin_states_evidence": []}
    block = None
    in_steps = False
    in_timing = False
    spin = None
    force_block = None
    force_blocks = []
    not_converged = False
    for line_number, line in enumerate(invocation.splitlines(), 1):
        stripped = line.strip()
        if rks:
            if re.search(r"\bUKS\b|spin[\s-]+unrestricted", line, re.IGNORECASE):
                raise ValueError(f"Mismatched RKS evidence: UKS/spin-unrestricted at invocation line {line_number}")
            if re.fullmatch(r"DFT\|\s+Spin restricted Kohn-Sham \(RKS\) calculation\s+RKS", stripped):
                result["spin_validation"]["banner_evidence"].append({"line": line_number, "raw": line})
            if re.match(r"DFT\|\s+Number of spin states\b", stripped):
                if not re.fullmatch(r"DFT\|\s+Number of spin states\s+1", stripped):
                    raise ValueError(f"Mismatched RKS evidence: Number of spin states must be 1: {line}")
                result["spin_validation"]["spin_states_evidence"].append({"line": line_number, "raw": line})
        if "SCF WAVEFUNCTION OPTIMIZATION" in line:
            block = {"index": len(result["scf_blocks"]) + 1, "updates": [],
                     "converged": False, "declared_steps": [], "line": line_number}
            result["scf_blocks"].append(block)
            in_steps = False
        if re.search(r"Step\s+Update method\s+Time\s+Convergence", line):
            if block is None:
                raise ValueError("SCF table without SCF invocation header")
            in_steps = True
        if in_steps and re.match(r"^\s*\d+\s+", line):
            fields = line.split()
            if len(fields) < 6 or not all(NUMERIC.fullmatch(field) for field in fields[-4:]):
                raise ValueError(f"Malformed SCF update at invocation line {line_number}: {line}")
            update = {"step": int(fields[0]), "method_raw": " ".join(fields[1:-4]),
                      "time": quantity(fields[-4], "seconds"),
                      "convergence": quantity(fields[-3], "CP2K convergence metric"),
                      "total_energy": quantity(fields[-2], "hartree"),
                      "energy_change": quantity(fields[-1], "hartree"),
                      "line": line_number, "raw": line}
            if decimal(fields[-4]) < 0 or update["step"] < 1:
                raise ValueError("Invalid SCF step/time")
            block["updates"].append(update)
        converged = re.search(r"SCF run converged in\s+(\d+)\s+steps", line)
        if converged:
            if block is None:
                raise ValueError("Convergence marker without SCF block")
            block["converged"] = True
            block["declared_steps"].append(int(converged.group(1)))
            in_steps = False
        if re.search(r"SCF run NOT converged", line, re.IGNORECASE):
            not_converged = True
            if block:
                block["converged"] = False
            in_steps = False
        match = re.fullmatch(r"Spin\s+([12])", stripped, re.IGNORECASE)
        if match:
            if rks:
                raise ValueError(f"Mismatched RKS evidence: spin-resolved electron section: {line}")
            spin = "alpha" if match.group(1) == "1" else "beta"
        match = re.fullmatch(rf"Number of electrons:\s*({NUMBER})", stripped)
        if rks and stripped.startswith("Number of electrons:"):
            result["electron_counts"]["total"] = quantity(stripped.partition(":")[2].strip(), "electrons")
        elif match and spin:
            result["electron_counts"][spin] = quantity(match.group(1), "electrons")
        match = re.fullmatch(rf"(?:Number of\s+)?(alpha|beta) electrons:\s*({NUMBER})",
                             stripped, re.IGNORECASE)
        if match:
            if rks:
                raise ValueError(f"Mismatched RKS evidence: spin-resolved electron count: {line}")
            result["electron_counts"][match.group(1).lower()] = quantity(match.group(2), "electrons")
        match = re.search(rf"Integrated (absolute )?spin density:\s*({NUMBER})\s*$", line)
        if match:
            key = "integrated_absolute_spin" if match.group(1) else "integrated_spin"
            result[key] = quantity(match.group(2), "electrons (alpha minus beta density)")
        match = re.search(rf"ENERGY\|\s+Total FORCE_EVAL.*energy \[hartree\]\s+({NUMBER})\s*$", line)
        if match:
            result["energy"] = quantity(match.group(1), "hartree")
        if "Atomic forces" in line and "FORCES|" in line or "ATOMIC FORCES in" in line:
            if force_block is not None:
                raise ValueError("New force header before preceding force block closed")
            if not any(unit in line.lower() for unit in ("hartree/bohr", "[a.u.]", "[au]")):
                raise ValueError(f"Unsupported force unit: {line}")
            force_block = {"rows": [], "line": line_number, "complete": False}
            force_blocks.append(force_block)
        elif force_block is not None:
            row = re.sub(r"^\s*FORCES\|\s*", "", line).strip()
            if re.match(r"SUM(?:\s|$)|Total atomic force", row, re.IGNORECASE):
                force_block["complete"] = True
                force_block = None
            elif re.match(r"^\d+\s+", row):
                fields = row.split()
                if len(fields) in (4, 5) and all(NUMERIC.fullmatch(value) for value in fields[1:]):
                    atom, kind, element, components = int(fields[0]), None, None, fields[1:4]
                elif len(fields) in (6, 7) and fields[1].isdigit() and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", fields[2]):
                    atom, kind, element, components = int(fields[0]), int(fields[1]), fields[2], fields[3:6]
                else:
                    raise ValueError(f"Malformed force row at invocation line {line_number}: {line}")
                force_block["rows"].append({"atom": atom, "kind": kind, "element": element,
                                           "components": [quantity(value, "hartree/bohr") for value in components],
                                           "raw": line})
        if "SELF TIME" in line and "TOTAL TIME" in line:
            in_timing = True
        elif in_timing:
            fields = line.split()
            if len(fields) == 7 and fields[1].isdigit() and all(NUMERIC.fullmatch(value) for value in fields[2:]):
                if any(decimal(value) < 0 for value in fields[3:]):
                    raise ValueError("Negative CP2K timer")
                result["timings"].append({"routine": fields[0], "calls": int(fields[1]),
                                          "asd_raw": fields[2],
                                          "self_average_seconds": quantity(fields[3], "seconds"),
                                          "self_maximum_seconds": quantity(fields[4], "seconds"),
                                          "total_average_seconds": quantity(fields[5], "seconds"),
                                          "total_maximum_seconds": quantity(fields[6], "seconds"), "raw": line})
            elif result["timings"] and (not stripped or set(stripped) <= {"-"}):
                in_timing = False
        if "[PW_GPU]" in line and ("device memory" in line or "samples" in line):
            result["vram_samples"].append(line.strip())
    if rks:
        if not result["spin_validation"]["banner_evidence"]:
            raise ValueError("Missing RKS evidence: DFT| Spin restricted Kohn-Sham (RKS) calculation ... RKS")
        if not result["spin_validation"]["spin_states_evidence"]:
            raise ValueError("Missing RKS evidence: DFT| Number of spin states ... 1")
    if force_block is not None:
        raise ValueError("Incomplete final atomic-force block")
    if force_blocks:
        result["forces"] = force_blocks[-1]["rows"]
        result["force_block_count"] = len(force_blocks)
        identities = [row["atom"] for row in result["forces"]]
        if len(identities) != natoms or sorted(identities) != list(range(1, natoms + 1)):
            raise ValueError(f"Force IDs must be unique and exactly 1..{natoms}; got {identities}")
        result["forces"].sort(key=lambda row: row["atom"])
        if result["scf_blocks"] and force_blocks[-1]["line"] < result["scf_blocks"][-1]["line"]:
            raise ValueError("Final force block precedes the final SCF calculation")
    if not result["scf_blocks"] or any(not item["updates"] for item in result["scf_blocks"]):
        raise ValueError("Missing SCF update steps/times")
    result["scf_converged"] = not not_converged and all(item["converged"] for item in result["scf_blocks"])
    updates = [update for item in result["scf_blocks"] for update in item["updates"]]
    result["scf_updates"] = updates
    result["cold_start"] = {"definition": "First printed update of this invocation only",
                            "updates": updates[:1], "statistics": stats(updates[:1])}
    result["steady_state"] = {"definition": "All subsequent printed updates; not an equilibrium claim",
                              "updates": updates[1:], "statistics": stats(updates[1:])}
    for item in result["scf_blocks"]:
        previous_step = 0
        for update in item["updates"]:
            if update["step"] != previous_step + 1 and update["step"] != 1:
                raise ValueError("Missing or out-of-order SCF update rows")
            previous_step = update["step"]
        if item["converged"] and (not item["declared_steps"] or item["declared_steps"][-1] != previous_step):
            raise ValueError("Printed SCF update count does not match the convergence marker")
        item["statistics"] = stats(item["updates"])
    if not result["timings"] or not any(item["routine"] == "CP2K" for item in result["timings"]):
        raise ValueError("Missing CP2K TOTAL TIME table")
    result["total_time_seconds"] = [item for item in result["timings"] if item["routine"] == "CP2K"][-1][
        "total_maximum_seconds"]
    if not timing_only:
        missing = []
        for name in (("energy", "forces") if rks else ("energy", "integrated_spin", "forces")):
            if not result[name]:
                missing.append(name)
        for name in (("total",) if rks else ("alpha", "beta")):
            if name not in result["electron_counts"]:
                missing.append(f"{name} electron count")
        if missing:
            raise ValueError("Missing molecular data: " + ", ".join(missing))
        if not result["scf_converged"]:
            raise ValueError("SCF not converged; --timing-only is required for diagnostic sweeps")
    return result


def difference(left, right, threshold):
    delta = decimal(right["raw"]) - decimal(left["raw"])
    return {"reference": left, "candidate": right, "delta": str(delta), "absolute_delta": str(abs(delta)),
            "threshold": str(threshold), "unit": left["unit"], "within_threshold": abs(delta) <= threshold}


def compare(left, right, force_atol, energy_atol, spin_atol, timing_only=False, rks=False):
    output = {"schema": 1, "mode": "timing-only" if timing_only else "molecular",
              "molecular_acceptance": False, "accepted": False, "reference": left, "candidate": right,
              "vram_caveat": VRAM_CAVEAT}
    if rks:
        for name, log in (("reference", left), ("candidate", right)):
            validation = log.get("spin_validation", {})
            if (validation.get("status") != "not_applicable_RKS" or not validation.get("banner_evidence")
                    or not validation.get("spin_states_evidence")):
                raise ValueError(f"Missing RKS evidence in {name}; parse with rks=True")
        output["spin_validation"] = {"status": "not_applicable_RKS",
                                      "reference": left["spin_validation"], "candidate": right["spin_validation"]}
    if timing_only:
        output["diagnostic_only"] = True
        output["reason"] = "No molecular acceptance is evaluated, including for non-converged ELPA sweeps."
        return output
    if any(value is None or not value.is_finite() or value < 0 for value in (force_atol, energy_atol, spin_atol)):
        raise ValueError("All three finite, nonnegative explicit thresholds are required")
    output["thresholds"] = {"force_hartree_per_bohr": str(force_atol),
                            "energy_hartree": str(energy_atol), "spin_electrons": str(spin_atol)}
    output["energy"] = difference(left["energy"], right["energy"], energy_atol)
    if not rks:
        output["integrated_spin"] = difference(left["integrated_spin"], right["integrated_spin"], spin_atol)
        if bool(left["integrated_absolute_spin"]) != bool(right["integrated_absolute_spin"]):
            raise ValueError("Integrated absolute-spin output is present in only one log")
        if left["integrated_absolute_spin"]:
            output["integrated_absolute_spin"] = difference(
                left["integrated_absolute_spin"], right["integrated_absolute_spin"], spin_atol)
    output["electron_counts"] = {name: difference(left["electron_counts"][name], right["electron_counts"][name],
                                                  Decimal(0)) for name in (("total",) if rks else ("alpha", "beta"))}
    if len(left["forces"]) != len(right["forces"]):
        raise ValueError("Different force counts")
    force_differences = []
    for reference, candidate in zip(left["forces"], right["forces"]):
        if reference["atom"] != candidate["atom"]:
            raise ValueError("Force atom IDs differ")
        for name in ("element", "kind"):
            if reference[name] is not None and candidate[name] is not None and reference[name] != candidate[name]:
                raise ValueError(f"Force atom {reference['atom']} {name} differs")
        for axis, reference_value, candidate_value in zip("xyz", reference["components"], candidate["components"]):
            force_differences.append({"atom": reference["atom"], "axis": axis,
                                      **difference(reference_value, candidate_value, force_atol)})
    output["force_components"] = force_differences
    output["maximum_absolute_force_delta"] = str(max(Decimal(item["absolute_delta"]) for item in force_differences))
    checks = [output["energy"], *output["electron_counts"].values(), *force_differences]
    if not rks:
        checks.append(output["integrated_spin"])
    if "integrated_absolute_spin" in output:
        checks.append(output["integrated_absolute_spin"])
    output["accepted"] = left["scf_converged"] and right["scf_converged"] and all(item["within_threshold"] for item in checks)
    output["molecular_acceptance"] = output["accepted"]
    return output


def threshold(value):
    try:
        result = decimal(value)
        if result < 0:
            raise ValueError("Negative threshold")
        return result
    except (InvalidOperation, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def attach_record(log, record_path):
    if record_path is None:
        return None
    path = Path(record_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    files = record.get("recorded_files", [])
    if not files or not any(item.get("role") == "input" for item in files):
        raise ValueError(f"Record lacks exact input hashes: {path}")
    for item in files:
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError(f"Invalid input/restart hash in {path}")
    return {"record_path": str(path.resolve()), "record_sha256": sha256(path), "log_sha256": log["sha256"],
            "recorded_files": files, "record_exit": record.get("exit"),
            "binding": "User-supplied record association; hashes describe pre-run files, not inferred restart use."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", nargs="?")
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("--natoms", type=int)
    parser.add_argument("--force-atol", type=threshold, help="Absolute tolerance, hartree/bohr; required for molecular mode")
    parser.add_argument("--energy-atol", type=threshold, help="Absolute tolerance, hartree; required for molecular mode")
    parser.add_argument("--spin-atol", type=threshold, help="Absolute tolerance, electrons; required for molecular mode")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--rks", action="store_true",
                        help="Require explicit closed-shell RKS evidence; compare total electrons, spin not applicable")
    parser.add_argument("--reference-record")
    parser.add_argument("--candidate-record")
    parser.add_argument("--output", type=Path, help="Create JSON file exclusively; never overwrite")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        from test_tools import run_parser_tests
        return run_parser_tests()
    if not args.reference or not args.candidate or not args.natoms or args.natoms < 1:
        parser.error("reference, candidate and positive --natoms are required")
    if not args.timing_only and any(value is None for value in (args.force_atol, args.energy_atol, args.spin_atol)):
        parser.error("Molecular mode requires --force-atol, --energy-atol and --spin-atol")
    try:
        left = parse_log(args.reference, args.natoms, args.timing_only, rks=args.rks)
        right = parse_log(args.candidate, args.natoms, args.timing_only, rks=args.rks)
        output = compare(left, right, args.force_atol, args.energy_atol, args.spin_atol, args.timing_only, rks=args.rks)
        output["provenance"] = {"reference": attach_record(left, args.reference_record),
                                "candidate": attach_record(right, args.candidate_record),
                                "caveat": "Molecular acceptance compares logged observables only. "
                                "Input/restart hashes belong to run records; no identical-input claim is inferred."}
        status = 0 if args.timing_only or output["accepted"] else 1
    except (OSError, ValueError, InvalidOperation) as error:
        output = {"schema": 1, "accepted": False, "molecular_acceptance": False, "error": str(error)}
        status = 2
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        print(serialized, end="", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
