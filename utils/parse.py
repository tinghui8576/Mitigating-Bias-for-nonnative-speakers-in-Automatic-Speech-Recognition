import argparse
def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train a speech classification model with wandb logging.")
    # Define arguments
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=False, 
        default='openai/whisper-small', 
        help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
    )
    parser.add_argument(
        '--language', 
        type=str, 
        required=False, 
        default='English', 
        help='Language the model is being adapted to in Camel case.'
    )
    parser.add_argument(
        '--strategy', 
        type=str, 
        required=False, 
        choices=['lora', 'lwf'], 
        default='lora', 
        help='Finetune Strategy for Whisper model.'
    )
    parser.add_argument(
        '--num_proc', 
        type=int, 
        required=False, 
        default=2, 
        help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
    )
    parser.add_argument(
        '--train_strategy', 
        type=str, 
        required=False, 
        default='steps', 
        help='Training strategy. Choose between steps and epoch.'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        required=False, 
        default=1.75e-5, 
        help='Learning rate for the fine-tuning process.'
    )
    parser.add_argument(
        '--warmup', 
        type=int, 
        required=False, 
        default=20000, 
        help='Number of warmup steps.'
    )
    parser.add_argument(
        '--train_batchsize', 
        type=int, 
        required=False, 
        default=48, 
        help='Batch size during the training phase.'
    )
    parser.add_argument(
        '--eval_batchsize', 
        type=int, 
        required=False, 
        default=32, 
        help='Batch size during the evaluation phase.'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        required=False, 
        default=20, 
        help='Number of epochs to train for.'
    )
    parser.add_argument(
        '--num_steps', 
        type=int, 
        required=False, 
        default=100000, 
        help='Number of steps to train for.'
    )
    parser.add_argument(
        '--resume_from_ckpt', 
        type=str, 
        required=False, 
        default=None, 
        help='Path to a trained checkpoint to resume training from.'
    )
    parser.add_argument(
        "--save_model", 
        type=str, 
        choices=['True', 'False'], 
        default='True', 
        help="Store the training model or no")
    parser.add_argument(
        "--project_name", 
        type=str, 
        default='None', 
        help="Name of the project")
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=False, 
        default='output_model_dir', 
        help='Output directory for the checkpoints generated.'
    )
    
    parser.add_argument('--use_cuda', action='store_true', help="Flag to use GPU if available.")
    parser.add_argument('--wandb', type=str, choices=['True', 'False'], default='True', help="Use wandb logging (True or False).")
    
    
    return parser.parse_args()