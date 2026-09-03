import torch

from diffusers import DDIMScheduler, DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup 

from Diffusion.timemlp import ResMLPNet
from Diffusion.denoising_diffusion import DenoisingDiffusion


def _init_noise_scheduler(parameters, device):
    variant = parameters["noise_scheduler_variant"]
    num_train = parameters["num_train_timesteps"]
    num_sample = parameters["num_sample_timesteps"]

    if variant == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_inference_steps=num_sample, device=device)

    elif variant == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_sample, device=device)

    elif variant == "pndm":
        scheduler = PNDMScheduler(
            num_train_timesteps=num_train,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_sample, device=device)

    elif variant == "dpm":
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train,
            algorithm_type="dpmsolver++",    # common default
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_sample, device=device)

    elif variant == "euler":
        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=num_train,
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(num_sample, device=device)

    elif variant == "cosine":
        scheduler = DDPMScheduler(
            num_train_timesteps=parameters["num_train_timesteps"],
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(
            parameters["num_sample_timesteps"],
            device=device
        )

    else:
        raise ValueError(f"Unknown scheduler variant: {variant}")

    return scheduler

def create_diffusion(parameters, device):
    if parameters["model"] != "net":
        raise NotImplementedError

    embedding_size = parameters["data_dim"][2] #[1,1,128]
    diffusion_num_layers = parameters["num_res_blocks"]
    
    net = ResMLPNet(
        in_channels=embedding_size,
        time_embed_dim=embedding_size,
        model_channels=embedding_size*2,
        bottleneck_channels=embedding_size*2,
        out_channels=embedding_size,
        num_res_blocks=diffusion_num_layers,
        dropout=0,
        use_context=False,
        context_channels=embedding_size
    )

    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=parameters["learning_rate"])
    noise_scheduler = _init_noise_scheduler(parameters, device)

    denoising_diffusion = DenoisingDiffusion(
        diffusion_network=net,
        optimizer=optimizer,
        parameters=parameters,
        noise_scheduler=noise_scheduler,
        device=device
    )

    return denoising_diffusion


    