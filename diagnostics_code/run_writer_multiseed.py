import json
import subprocess
from pathlib import Path

import numpy as np


SEEDS = [
    2090,
    2091,
    2092,
    2093,
    2094,
]

CHECKPOINT = (
    "outputs/"
    "retrieval_gradient_test/"
    "checkpoint_best.pt"
)

BASE_OUTPUT = Path(
    "outputs/writer_multiseed"
)

SCRIPT = "writer_summary_ablation.py"


def run_seed(seed):

    output_dir = (
        BASE_OUTPUT
        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        output_dir
        / "run.log"
    )

    command = [
        "python",
        SCRIPT,

        "--checkpoint",
        CHECKPOINT,

        "--train-examples",
        "4000",

        "--validation-examples",
        "500",

        "--test-examples",
        "500",

        "--epochs",
        "15",

        "--batch-size",
        "128",

        "--writer-learning-rate",
        "1e-4",

        "--summary-learning-rate",
        "3e-4",

        "--reader-learning-rate",
        "3e-4",

        "--residual-learning-rate",
        "1e-3",

        "--seed",
        str(seed),

        "--output-dir",
        str(output_dir),
    ]

    print()
    print("=" * 100)
    print(
        f"RUNNING SEED {seed}"
    )
    print("=" * 100)

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:

            print(
                line,
                end="",
            )

            log_file.write(
                line
            )

        return_code = (
            process.wait()
        )

    if return_code != 0:

        raise RuntimeError(
            f"Seed {seed} failed "
            f"with exit code "
            f"{return_code}"
        )

    result_path = (
        output_dir
        / "writer_summary_ablation_results.json"
    )

    if not result_path.exists():

        raise FileNotFoundError(
            f"Missing result file: "
            f"{result_path}"
        )

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    return results


def summarize_variant(
    all_results,
    variant,
):

    matched = []
    mismatched = []
    query_only = []
    nll_gap = []
    stored_l2 = []
    source_l2 = []

    per_seed = []

    for seed, result in (
        all_results.items()
    ):

        test = (
            result[
                variant
            ]["test"]
        )

        row = {
            "seed": seed,

            "matched": (
                test[
                    "matched_accuracy"
                ]
            ),

            "mismatched": (
                test[
                    "mismatched_accuracy"
                ]
            ),

            "query_only": (
                test[
                    "query_only_accuracy"
                ]
            ),

            "nll_gap": (
                test[
                    "nll_gap"
                ]
            ),

            "source_l2": (
                test[
                    "source_geometry"
                ][
                    "relative_l2"
                ]
            ),

            "stored_l2": (
                test[
                    "stored_geometry"
                ][
                    "relative_l2"
                ]
            ),
        }

        per_seed.append(
            row
        )

        matched.append(
            row["matched"]
        )

        mismatched.append(
            row["mismatched"]
        )

        query_only.append(
            row["query_only"]
        )

        nll_gap.append(
            row["nll_gap"]
        )

        source_l2.append(
            row["source_l2"]
        )

        stored_l2.append(
            row["stored_l2"]
        )

    def stats(values):

        values = np.asarray(
            values,
            dtype=float,
        )

        return {
            "mean": float(
                values.mean()
            ),

            "std": float(
                values.std(
                    ddof=1
                )
            ),

            "min": float(
                values.min()
            ),

            "max": float(
                values.max()
            ),
        }

    return {
        "per_seed": per_seed,

        "matched": (
            stats(matched)
        ),

        "mismatched": (
            stats(mismatched)
        ),

        "query_only": (
            stats(query_only)
        ),

        "nll_gap": (
            stats(nll_gap)
        ),

        "source_l2": (
            stats(source_l2)
        ),

        "stored_l2": (
            stats(stored_l2)
        ),
    }


def print_variant(
    name,
    summary,
):

    print()
    print("=" * 100)
    print(
        name.upper()
    )
    print("=" * 100)

    print(
        f"{'SEED':<10}"
        f"{'MATCH':>12}"
        f"{'MISMATCH':>14}"
        f"{'QUERY':>12}"
        f"{'NLL GAP':>14}"
        f"{'SOURCE L2':>14}"
        f"{'STORED L2':>14}"
    )

    print("-" * 90)

    for row in (
        summary[
            "per_seed"
        ]
    ):

        print(
            f"{row['seed']:<10}"
            f"{row['matched']:>11.2f}%"
            f"{row['mismatched']:>13.2f}%"
            f"{row['query_only']:>11.2f}%"
            f"{row['nll_gap']:>14.4f}"
            f"{row['source_l2']:>14.4f}"
            f"{row['stored_l2']:>14.4f}"
        )

    print()
    print(
        "MATCHED ACCURACY:"
    )

    print(
        f"{summary['matched']['mean']:.2f}"
        f" ± "
        f"{summary['matched']['std']:.2f}%"
    )

    print(
        "MISMATCHED ACCURACY:"
    )

    print(
        f"{summary['mismatched']['mean']:.2f}"
        f" ± "
        f"{summary['mismatched']['std']:.2f}%"
    )

    print(
        "NLL GAP:"
    )

    print(
        f"{summary['nll_gap']['mean']:.4f}"
        f" ± "
        f"{summary['nll_gap']['std']:.4f}"
    )

    print(
        "STORED RELATIVE L2:"
    )

    print(
        f"{summary['stored_l2']['mean']:.4f}"
        f" ± "
        f"{summary['stored_l2']['std']:.4f}"
    )


def main():

    BASE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = {}

    for seed in SEEDS:

        result = run_seed(
            seed
        )

        all_results[
            str(seed)
        ] = result

    original_summary = (
        summarize_variant(
            all_results,
            "original_writer",
        )
    )

    direct_summary = (
        summarize_variant(
            all_results,
            "direct_summary",
        )
    )

    print_variant(
        "Original Writer",
        original_summary,
    )

    print_variant(
        "Direct Summary",
        direct_summary,
    )

    print()
    print("=" * 100)
    print(
        "FINAL STABILITY COMPARISON"
    )
    print("=" * 100)

    print(
        f"{'VARIANT':<24}"
        f"{'MATCHED':>18}"
        f"{'MISMATCHED':>18}"
        f"{'NLL GAP':>20}"
    )

    print("-" * 82)

    print(
        f"{'Original Writer':<24}"
        f"{original_summary['matched']['mean']:>9.2f}"
        f" ± "
        f"{original_summary['matched']['std']:<6.2f}"
        f"{original_summary['mismatched']['mean']:>8.2f}"
        f" ± "
        f"{original_summary['mismatched']['std']:<6.2f}"
        f"{original_summary['nll_gap']['mean']:>10.4f}"
        f" ± "
        f"{original_summary['nll_gap']['std']:<7.4f}"
    )

    print(
        f"{'Direct Summary':<24}"
        f"{direct_summary['matched']['mean']:>9.2f}"
        f" ± "
        f"{direct_summary['matched']['std']:<6.2f}"
        f"{direct_summary['mismatched']['mean']:>8.2f}"
        f" ± "
        f"{direct_summary['mismatched']['std']:<6.2f}"
        f"{direct_summary['nll_gap']['mean']:>10.4f}"
        f" ± "
        f"{direct_summary['nll_gap']['std']:<7.4f}"
    )

    print()
    print("=" * 100)
    print(
        "AUTOMATIC INTERPRETATION"
    )
    print("=" * 100)

    writer_mean = (
        original_summary[
            "matched"
        ]["mean"]
    )

    writer_std = (
        original_summary[
            "matched"
        ]["std"]
    )

    direct_mean = (
        direct_summary[
            "matched"
        ]["mean"]
    )

    difference = (
        direct_mean
        - writer_mean
    )

    if (
        writer_mean >= 75
        and
        writer_std <= 8
    ):

        print(
            "STABLE PASS:"
        )

        print(
            "CandidateWriter consistently learns "
            "a useful latent VALUE representation."
        )

        print(
            "The original checkpoint failure is "
            "better explained by poor training / "
            "objective alignment than by an "
            "incapable writer architecture."
        )

    elif (
        writer_mean >= 65
    ):

        print(
            "PARTIAL / MODERATELY STABLE:"
        )

        print(
            "CandidateWriter is clearly learnable, "
            "but training remains variable."
        )

        print(
            "Next step should focus on training "
            "stability rather than replacing "
            "the writer architecture."
        )

    else:

        print(
            "UNSTABLE / WEAK:"
        )

        print(
            "CandidateWriter can occasionally work "
            "but is not reliably trainable under "
            "the current objective."
        )

        print(
            "Training stabilization or architectural "
            "simplification is needed before "
            "integrating the reader."
        )

    print()

    print(
        f"Direct summary - writer "
        f"mean accuracy difference: "
        f"{difference:+.2f} pp"
    )

    output = {
        "seeds": SEEDS,

        "original_writer": (
            original_summary
        ),

        "direct_summary": (
            direct_summary
        ),

        "all_raw_results": (
            all_results
        ),
    }

    output_path = (
        BASE_OUTPUT
        / "multiseed_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        "Saved summary:",
        output_path,
    )


if __name__ == "__main__":
    main()