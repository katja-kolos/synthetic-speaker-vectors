import datetime
import os
import torch
import torch.nn.functional as F


def classifier_grad(classifier, x_t_batch, t_batch, y_batch, 
                    guidance_temperature=1.0, grad_clip=None, normalize=False):
    # computes gradient for x_t * log p(y | x_t, t)
    # returns grad of sum(log p(y | x_t, t)) wrt x_t
    # x_t: [B, dim]; t_batch: long (ints) of shape [B]; y: long (ints) of shape [B]

    x_in_batch = x_t_batch.detach().requires_grad_(True)
    # print(f"Shape x_in_batch: {x_in_batch.shape}")
    # forward pass 
    logits_batch = classifier(x_in_batch, t_batch)
    # print(f"Shape logits after applying the classifier: {logits_batch.shape}")

    # hyperparameter: temperature control; we end up generating female and male with different temperature
    if guidance_temperature != 1.0:
        logits_batch = logits_batch / guidance_temperature
    
    # log p(class | x_t, t)
    log_probs_batch = F.log_softmax(logits_batch, dim=-1) 
    # print(f"Shape log_probs: {log_probs_batch.shape}")
    # pick the target class log p(y_i | x_t_i, t_i) for each sample i in the batch
    idx = torch.arange(x_in_batch.size(0), device=x_in_batch.device)
    logp_y_batch = log_probs_batch[idx, y_batch] # [B]
    # print(f"Shape selected: {logp_y_batch.shape}")
    # SUM TO scalar -- to call autograd once
    objective = logp_y_batch.sum()
    grad = torch.autograd.grad(objective, x_in_batch, create_graph=False, retain_graph=False)[0]

    if normalize:
        grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)

    if grad_clip is not None:
        grad = grad.clamp(-grad_clip, grad_clip)

    return grad.detach(), logp_y_batch.detach() #DETACH so that callers don't build huge graphs accidentally

class DenoisingDiffusion:
    def __init__(self, diffusion_network, optimizer, noise_scheduler, parameters, device):
        self.diffusion_network = diffusion_network
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.device = device

        self.dim = parameters["data_dim"][2]
        self.epochs = parameters["epochs"]
        self.parameters = parameters

        self.num_steps = 0

    def train(self, data_loader, writer):
        global_step = 0
        avg_losses = []
        for epoch in range(self.epochs):
        
            losses = []
            for step, batch in enumerate(data_loader):
                self.num_steps += 1 #overall number of training steps -- new cycle won't start if greater than max in config

                vectors = batch[0] #[batch_size, 128]
                spk_ids = batch[1]

                vectors = vectors.to(self.device)

                # Sample noise to add to speaker vectors
                noise = torch.randn(vectors.shape, device=vectors.device)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_train_timesteps, (vectors.shape[0],), device=vectors.device
                ).long()

                print(f"vectors.shape: {vectors.shape}")
                print(f"noise.shape: {noise.shape}")
                print(f"timesteps.shape: {timesteps.shape}")

                noisy_batch = self.noise_scheduler.add_noise(vectors, noise, timesteps)
                print(f"noisy_batch.shape: {noisy_batch.shape}")
                noise_pred = self.diffusion_network(noisy_batch, timesteps)
                print(f"noise_pred.shape: {noise_pred.shape}")
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                print(f"loss.shape: {loss.shape}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                losses.append(loss)

            avg_loss = sum(losses) / len(losses)
            print(f"Avg. loss after epoch {epoch}: {avg_loss}")
            avg_losses.append((epoch, avg_loss))

    def _init_noise(self, batch_size, seed=42):
        maxs, mins = 0.5, -0.5 # init random noise in the range of GST embeddings
        noise = torch.rand(batch_size, self.dim) * (maxs - mins) + mins
        return noise
    
    # Unconditional
    def sample(self, num_samples):
        self.diffusion_network.eval()
        synthetic_embeddings = []
        while len(synthetic_embeddings) < num_samples:
            if len(synthetic_embeddings) % 400 == 0:
                print(f"Sampled {len(synthetic_embeddings)} vectors.")
            x_t = self._init_noise(self.parameters["eval_batch_size"]).to(self.device)
            samples = self._sample_from_noise(self.diffusion_network, self.noise_scheduler.timesteps, x_t, self.parameters["eval_batch_size"])
            synthetic_embeddings.extend(samples.tolist())

        synthetic_embeddings_tensor = torch.Tensor(synthetic_embeddings)
        return synthetic_embeddings_tensor.to('cpu')
    
    def _sample_from_noise(self, timemlp_model, timesteps, x_t, batch_size, sample_every_timestep=False, 
                           save_dir='../../workspace/synthetic_speaker_embeddings/Diffusion/ddim_sch/', **kwargs):
        for i, t in enumerate(timesteps):
            t_batch = torch.ones(batch_size, device=x_t.device, dtype=torch.long) * t

            pred = timemlp_model(x_t, t_batch)

            # compute the previous noise sample x_t -> x_t-1
            x_t = self.noise_scheduler.step(pred, t, x_t, **kwargs).prev_sample
            
            # ablation for intermediate timesteps
            if sample_every_timestep:
                if i % sample_every_timestep == 0:
                    file_output_dir = os.path.abspath(save_dir)
                    os.makedirs(file_output_dir, exist_ok=True)
                    output_path = os.path.join(file_output_dir, f'batch_t_{i}.pt')
                    torch.save(x_t, output_path)
                    print(f'Saved batch at timestep {i} to {output_path}')
        return x_t
    
    # Conditional
    def sample_conditional(self, classifier, y):
        # Args:
        # - classifier (saved model checkpoint): noise robust classifier at different timesteps, 
        # whose gradients will be used to guide the DDPM
        # - y: vector of labels (e.g. for gender [0, 1, 1, 0, ...])
        # Returns:
        # - torch.Tensor of speaker embeddings that correspond to the desired labels in the same order 
        self.diffusion_network.eval()
        synthetic_embeddings = []
        bs = self.parameters["eval_batch_size"]
        batch_idx = 0
        while len(synthetic_embeddings) < len(y):
            
            current_y = y[batch_idx * bs: (batch_idx+1)*bs]
            if len(synthetic_embeddings) % 400 == 0:
                print(f"Sampled {len(synthetic_embeddings)} vectors.")
            x_t = self._init_noise(len(current_y)).to(self.device)
            classifier = classifier.to(self.device)
            current_y = current_y.to(self.device)
            samples = self._sample_from_noise_classifier_guidance(self.diffusion_network, self.noise_scheduler.timesteps, x_t, len(current_y), classifier, current_y)
            synthetic_embeddings.extend(samples.tolist())
            batch_idx += 1

        synthetic_embeddings_tensor = torch.Tensor(synthetic_embeddings)
        return synthetic_embeddings_tensor.to('cpu')
    
    def _sigma_t_from_scheduler(self, scheduler, t, device):
        # t is a scalar timestep value (int) from scheduler.timestep
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)
        return sigma_t
    
    def _sample_from_noise_classifier_guidance(self, timemlp_model, timesteps, x_t, batch_size, classifier, y, guidance_scale=3.0, **step_kwargs):
        timemlp_model.eval()
        classifier.eval()
        for i, t in enumerate(timesteps):
            # print(t)
            t_batch = torch.full((batch_size, ), t, device=t.device, dtype=torch.long)
            # print(f"Shape of t_batch: {t_batch.shape}")
            # print(f"Shape of y: {y.shape}") # expected batch_size
            with torch.no_grad():
                eps = timemlp_model(x_t, t_batch)
                # print(f"Shape of eps after forward on x_t and t_batch: {eps.shape}")

            with torch.enable_grad():
                grad, logp_y = classifier_grad(classifier, x_t, t_batch, y)
                # print(f"Shape of grad: {grad.shape}")

            sigma_t = self._sigma_t_from_scheduler(self.noise_scheduler, int(t), x_t.device)
            # print(f"Shape of sigma_t: {sigma_t.shape}")

            eps_guided = eps - guidance_scale * sigma_t * grad

            x_t = self.noise_scheduler.step(eps_guided, t, x_t, **step_kwargs).prev_sample

        return x_t

    def save_model_checkpoint(self, model_path, timestampStr):
        if not timestampStr:
            dateTimeObj = datetime.datetime.now()
            timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")

        name = '%s_%s' % (timestampStr, 'diffusion')
        model_filename = os.path.join(model_path, name)
        os.makedirs(model_path, exist_ok=True)
        torch.save(
            {"model": self.diffusion_network.state_dict(),
             "config": self.parameters, #this specifies what type of scheduler was used etc
             "iterations": self.num_steps}, 
            model_filename)
        return model_filename
