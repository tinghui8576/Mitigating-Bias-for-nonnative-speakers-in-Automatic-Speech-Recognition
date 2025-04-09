
import torch
import wandb
import argparse
import utils.load as load
import utils.Model as Model
import utils
from torch.utils.data import DataLoader

def predict(model, dataloader, opt):
    
    
    
    outputs_df = model.predict(dataloader)

    # Save the DataFrame to a CSV file
    output_csv_path = f"{opt['dataset_name']}_transcriptions_{opt['model_type']}.csv"  # Specify the desired path and filename
    outputs_df.to_csv(output_csv_path, index=False)

    # Print the DataFrame or confirmation message
    print(f"Transcriptions saved to {output_csv_path}")
    return outputs_df

def evaulate(pred_df, results):

    return 
def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train a speech classification model with wandb logging.")
    
    # Define arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--save_model", type=str, choices=['True', 'False'], default='True', help="Store the training model or no")
    parser.add_argument("--project_name", type=str, default='None', help="Name of the project")
    parser.add_argument('--model_type', type=str, default='whisper-small')
    parser.add_argument('--dataset_name', default='speech-accent-archive', 
                        type=str,
                        choices=[
                            'speech-accent-archive',
                            'artie-bias-corpus', 
                        ], 
                        help="Name of the dataset")
    parser.add_argument('--sens_name', default='Gender', choices=['Gender', 'Age'])
    
    # parser.add_argument('--sens_classes', type=str, choices=['Binary', 'Multiple'], default='Binary', help='number of sensitive classes')
    
    parser.add_argument('--use_cuda', action='store_true', help="Flag to use GPU if available.")
    # parser.add_argument('--wandb', type=str, choices=['True', 'False'], default='True', help="Use wandb logging (True or False).")
    
    
    return parser.parse_args()



        

if __name__ == "__main__":
    

    opt = vars(parse_args())

    # Check if a GPU is available
    opt['device'] = torch.device('cuda' if opt['use_cuda'] and torch.cuda.is_available() else 'cpu')
 
    # if opt['wandb'] == 'True' or opt['wandb'] == True:
        
    #     wandb.init(
    #         project=opt['project_name'],
    #         name=opt['experiment'],
    #         config=opt
    #     )
    # else:
    #     wandb = None
    
    df = load.get_dataloader(opt)

    batch_size = opt['batch_size']
    test_dataset = utils.WhisperDataset(df)
    test_loader = DataLoader(test_dataset,batch_size= batch_size, collate_fn=utils.collate_fn, shuffle=True)
    model = Model.Whisper(opt)
    pred_df = predict(model, test_loader, opt)
    #evaluate(model, test_loader, opt)
    # # Train the model
    # train(model, train_loader, test_loader, opt, epochs=opt['epochs'])
    
    # if opt['wandb'] == 'True' or opt['wandb'] == True:
    #     wandb.finish()
    # else:
    #     wandb = None
    
