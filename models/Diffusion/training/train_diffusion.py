import datetime
import torch
from Diffusion.denoising_diffusion import DenoisingDiffusion

def train_diffusion(model, dataset, parameters,
    logger, device, writer, timestampStr=None, models_dir="../../workspace/models/diffusion"):

    print(f"Will train model for {model.dim}D data for {model.epochs} epochs.")
    
    # train-test-split as in WGAN (But different random)
    train_len = int(len(dataset) * 0.8)
    dev_len = len(dataset) - train_len
    train_data, _ = torch.utils.data.random_split(dataset, (train_len, dev_len))
    print(f"Training on {len(train_data)} datapoints")

    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=parameters["train_batch_size"], 
        drop_last=True
    )

    for cycle in range(parameters["cycles"]):
        logger.info(f"Start cycle {cycle}")
        logger.info(f"Model trained for {model.num_steps} iterations")
        if model.num_steps >= parameters["n_max_iterations"]:
            logger.info("Breaky cycle loop - Max iterations reached - Stop training")
            break
        
        model.train(train_loader, writer)

    logger.info("Save model")

    if not timestampStr:
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")

    model_filename = model.save_model_checkpoint(models_dir, timestampStr)
    logger.info(f"Model saved to: {model_filename}")

