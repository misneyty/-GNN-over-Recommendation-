import argparse
from pathlib import Path

from src.data import load_dataset
from src.full_pipeline import run_full_pipeline
from src.predict import score_edges, write_submission
from src.train import train
from src.utils import (
    configure_reproducibility,
    get_device,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a recommendation model and export a submission.")
    parser.add_argument("--data-dir", type=str, default="data_file")
    parser.add_argument("--model", type=str, choices=["lightgcn", "heterognn"], default="lightgcn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--submission", type=str, default="outputs/submissions/Submission.csv")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Train HeteroGNN snapshots from scratch and run calibration.",
    )
    parser.add_argument(
        "--snapshot-epochs",
        type=str,
        default="60,80,100",
        help="Comma-separated epochs saved for the full-pipeline ensemble.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_pipeline:
        configure_reproducibility(args.seed)
    else:
        set_seed(args.seed)
    device = get_device(args.device)

    dataset = load_dataset(args.data_dir)
    if args.full_pipeline:
        checkpoint_path = args.checkpoint
        submission_path = args.submission
        if checkpoint_path == "outputs/checkpoints/best_model.pt":
            checkpoint_path = (
                "outputs/checkpoints/"
                "heterognn_from_scratch_full.pt"
            )
        if submission_path == "outputs/submissions/Submission.csv":
            submission_path = (
                "outputs/submissions/"
                "Submission_heterognn_from_scratch_full.csv"
            )
        snapshot_epochs = tuple(
            int(value.strip())
            for value in args.snapshot_epochs.split(",")
            if value.strip()
        )
        run_full_pipeline(
            dataset=dataset,
            data_dir=args.data_dir,
            checkpoint_path=checkpoint_path,
            submission_path=submission_path,
            snapshot_epochs=snapshot_epochs,
            dim=args.dim,
            layers=args.layers,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            device=device,
        )
        return

    result = train(
        dataset=dataset,
        model_name=args.model,
        epochs=args.epochs,
        dim=args.dim,
        layers=args.layers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        device=device,
    )

    save_checkpoint(
        args.checkpoint,
        model_name=args.model,
        model_state=result.model.state_dict(),
        best_threshold=result.best_threshold,
        valid_metrics=result.valid_metrics,
        num_authors=dataset.num_authors,
        num_papers=dataset.num_papers,
        calibrator=result.calibrator,
        calibrator_params=(
            result.calibrator.model.get_params()
            if result.calibrator is not None
            else None
        ),
        feature_names=(
            result.structural_features.feature_names
            if result.structural_features is not None
            else None
        ),
    )

    scores = score_edges(
        args.model,
        result.model,
        dataset.test_edges,
        dataset.num_authors,
        result.adj,
        device,
        calibrator=result.calibrator,
        structural_features=result.structural_features,
    )
    output_path = write_submission(scores, Path(args.submission), result.best_threshold)

    print(f"Validation metrics: {result.valid_metrics}")
    print(f"Best threshold: {result.best_threshold:.4f}")
    print(f"Saved checkpoint to: {args.checkpoint}")
    print(f"Saved submission to: {output_path}")


if __name__ == "__main__":
    main()
