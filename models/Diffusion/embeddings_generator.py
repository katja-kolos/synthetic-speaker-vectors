import torch

from Diffusion.init_diffusion import create_diffusion

# Unconditional generation
class EmbeddingsGenerator:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path

        self.denoising_diffusion = self._load_model(self.model_path)

    def _load_model(self, path):
        diffusion_checkpoint = torch.load(path, map_location='cpu')
        # Expected structure for the checkpoint:
        # { "model": self.diffusion_network.state_dict(),
        #   "config": self.parameters, #this specifies what type of scheduler was used etc
        #   "iterations": self.num_steps }
        assert type(diffusion_checkpoint) == dict, "Expected a dict with keys: model, config, iterations"
        parameters = diffusion_checkpoint["config"]

        model = create_diffusion(parameters, self.device)
        model.diffusion_network.load_state_dict(state_dict=diffusion_checkpoint["model"])
        model.diffusion_network.eval()
        return model

    def generate_embeddings(self, n=1000):
        generated_samples = self.denoising_diffusion.sample(num_samples=n).cpu()
        return generated_samples