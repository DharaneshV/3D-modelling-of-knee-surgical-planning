import argparse
import os

from src.segmentation.train_soft_tissue import train
from src.segmentation.predict_soft_tissue import run_inference_batch
from src.segmentation.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Week 3-4 MRI Soft Tissue Segmentation Pipeline")
    
    # Modes
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--predict", action="store_true", help="Run inference on test set")
    parser.add_argument("--evaluate", action="store_true", help="Run accuracy validation")
    
    # Shared args
    parser.add_argument("--data_dir", type=str, default="data/oaizib", help="Path to OAI-ZIB dataset")
    parser.add_argument("--model_type", type=str, default="swin_unetr", choices=["swin_unetr", "segresnet"])
    
    # Train args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases for quick test")
    parser.add_argument("--output_dir", type=str, default="results/models", help="Where to save model weights")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--resume", action="store_true", help="Resume training from best_model.pth if it exists")
    
    # Predict args
    parser.add_argument("--checkpoint", type=str, default="results/models/best_model.pth", help="Model weights for inference")
    parser.add_argument("--preds_dir", type=str, default="results/predictions", help="Where to save output masks")
    
    args = parser.parse_args()
    
    # If no mode is selected, default to printing help
    if not (args.train or args.predict or args.evaluate):
        parser.print_help()
        return

    if args.train:
        print("=== Starting Training ===")
        resume_ckpt = args.checkpoint if args.resume else None
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_cases=args.max_cases,
            use_amp=not args.no_amp,
            resume_checkpoint=resume_ckpt
        )
        
    if args.predict:
        print("\n=== Starting Inference ===")
        os.makedirs(args.preds_dir, exist_ok=True)
        run_inference_batch(
            data_dir=args.data_dir,
            output_dir=args.preds_dir,
            model_path=args.checkpoint,
            model_type=args.model_type,
            use_amp=not args.no_amp
        )
        
    if args.evaluate:
        print("\n=== Starting Evaluation ===")
        run_evaluation(
            data_dir=args.data_dir,
            predictions_dir=args.preds_dir,
            output_csv="results/accuracy_report.csv"
        )


if __name__ == "__main__":
    main()
