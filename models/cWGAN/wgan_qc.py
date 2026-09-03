# modified from https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/blob/main/anonymization/modules/sttts/speaker_embeddings/anonymization/utils/train_gan_model.py

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cvxopt import matrix
from cvxopt import solvers
from cvxopt import sparse
from cvxopt import spmatrix
from torch.autograd import grad as torch_grad
from tqdm import tqdm


class WassersteinGanQuadraticCost:

    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer, criterion, epochs, n_max_iterations,
                 data_dimensions, batch_size, device, gamma=0.1, K=-1, milestones=[150000, 250000], lr_anneal=1.0,
                 device_ids=None):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {
            'D' : [],
            'WD': [],
            'G' : [], 
            }
        self.num_steps = 0
        self.gen_steps = 0
        self.epochs = epochs
        self.n_max_iterations = n_max_iterations
        # put in the shape of a dataset sample
        self.data_dim = data_dimensions[0] * data_dimensions[1] * data_dimensions[2]
        self.batch_size = batch_size
        self.device = device
        self.criterion = criterion
        self.mone = torch.FloatTensor([-1]).to(device)
        self.tensorboard_counter = 0

        if K <= 0:
            self.K = 1 / self.data_dim
        else:
            self.K = K
        self.Kr = np.sqrt(self.K)
        self.LAMBDA = 2 * self.Kr * gamma * 2

        if device_ids is None:
            device_ids = [self.device.index]

        self.G = nn.DataParallel(self.G.to(self.device), device_ids=device_ids)
        self.D = nn.DataParallel(self.D.to(self.device), device_ids=device_ids)

        self.schedulerD = self._build_lr_scheduler_(self.D_opt, milestones, lr_anneal)
        self.schedulerG = self._build_lr_scheduler_(self.G_opt, milestones, lr_anneal)

        # self.c, self.A, self.pStart = self._prepare_linear_programming_solver_(self.batch_size)
        # we will now have a separate solver for each gender class (note: they may be imbalanced in the batch!)
        self.lp_cache = {}

    def _get_lp_solver(self, B):
        # for conditional WGAN:
        # separate solver for each class!
        if B not in self.lp_cache:
            # self.c, self.A, self.pStart = self._prepare_linear_programming_solver_(self.batch_size)
            c, A, pStart  = self._prepare_linear_programming_solver_(B)
            self.lp_cache[B] = c, A, pStart
        return self.lp_cache[B]


    def _build_lr_scheduler_(self, optimizer, milestones, lr_anneal, last_epoch=-1):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_anneal, last_epoch=-1)
        return scheduler

    def _quadratic_wasserstein_distance_(self, real, generated):
        # print("computing _quadratic_wasserstein_distance_")
        real = real.to(self.device)
        generated = generated.to(self.device)
        num_r = real.size(0)
        num_f = generated.size(0)
        # print(f"num_r, num_f: {num_r}, {num_f}")
        # print(f"shape of real: {real.shape}")
        # print(f"shape of fake: {generated.shape}")
        real_flat = real.view(num_r, -1)
        fake_flat = generated.view(num_f, -1)
        # print(f"shape of real_flat: {real_flat.shape}")
        # print(f"shape of fake_flat: {fake_flat.shape}")

        real3D = real_flat.unsqueeze(1).expand(num_r, num_f, self.data_dim)
        fake3D = fake_flat.unsqueeze(0).expand(num_r, num_f, self.data_dim)
        # print(f"shape of real3D, fake3D: {real3D.shape}, {fake3D.shape}")
        # compute squared L2 distance
        dif = real3D - fake3D
        # print(f"shape of dif: {dif.shape}")
        dist = 0.5 * dif.pow(2).sum(2).squeeze()
        # print(f"shape of dist: {dist.shape}")

        return self.K * dist

    def _prepare_linear_programming_solver_(self, batch_size):
        # we still use it in the conditional setting as is, but now separately for each class
        A = spmatrix(1.0, range(batch_size), [0] * batch_size, (batch_size, batch_size))
        for i in range(1, batch_size):
            Ai = spmatrix(1.0, range(batch_size), [i] * batch_size, (batch_size, batch_size))
            A = sparse([A, Ai])

        D = spmatrix(-1.0, range(batch_size), range(batch_size), (batch_size, batch_size))
        DM = D
        for i in range(1, batch_size):
            DM = sparse([DM, D])

        A = sparse([[A], [DM]])

        cr = matrix([-1.0 / batch_size] * batch_size)
        cf = matrix([1.0 / batch_size] * batch_size)
        c = matrix([cr, cf])

        pStart = {}
        pStart['x'] = matrix([matrix([1.0] * batch_size), matrix([-1.0] * batch_size)])
        pStart['s'] = matrix([1.0] * (2 * batch_size))

        return c, A, pStart

    # def _linear_programming_(self, distance, batch_size):
    #     b = matrix(distance.cpu().double().detach().numpy().flatten())
    #     sol = solvers.lp(self.c, self.A, b, primalstart=self.pStart, solver='glpk',
    #                      options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
    #     offset = 0.5 * (sum(sol['x'])) / batch_size
    #     sol['x'] = sol['x'] - offset
    #     self.pStart['x'] = sol['x']
    #     self.pStart['s'] = sol['s']

    #     return sol

    def _linear_programming_conditional_(self, distance, batch_size, c, A, pStart):
        # c, A, pStart used to be on self. 
        # They are now passed as arguments to compute this separately for each class
        b = matrix(distance.cpu().double().detach().numpy().flatten())
        sol = solvers.lp(c, A, b, primalstart=pStart, solver='glpk',
                         options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
        offset = 0.5 * (sum(sol['x'])) / batch_size
        sol['x'] = sol['x'] - offset
        pStart['x'] = sol['x']
        pStart['s'] = sol['s']

        return sol # we return a separate solver for each class

    # def _approx_OT_(self, sol):
    #     # Compute the OT mapping for each fake dataset
    #     ResMat = np.array(sol['z']).reshape((self.batch_size, self.batch_size))
    #     mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(self.device)

    #     return mapping

    def _approx_OT_group_(self, sol, B):
        # B is the size of the sub-batch with only 1-labels or 0-labels
        ResMat = np.array(sol['z']).reshape((B, B))
        mapping = torch.from_numpy(np.argmax(ResMat, axis=0)).long().to(self.device)

        return mapping

    def _optimal_transport_regularization_(self, output_fake, fake, real_fake_diff):
        output_fake_grad = torch.ones(output_fake.size()).to(self.device)
        gradients = torch_grad(outputs=output_fake, inputs=fake,
                               grad_outputs=output_fake_grad,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        n = gradients.size(0)
        RegLoss = 0.5 * ((gradients.view(n, -1).norm(dim=1) / (2 * self.Kr) - self.Kr / 2 * real_fake_diff.view(n,
                                                                                                                -1).norm(
            dim=1)).pow(2)).mean()
        fake.requires_grad = False

        return RegLoss

    def _critic_deep_regression_(self, speaker_vectors, labels, opt_iterations=1):
        speaker_vectors = speaker_vectors.to(self.device)

        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        self.G.train()
        self.D.train()

        # Get generated fake dataset
        # generated_data = self.sample_generator(self.batch_size)
        generated_data = self.sample_generator(labels)
        # print(f"_critic_deep_regression_: shape of generated data: {generated_data.shape}")


        # # This is for unconditional setting:
        # # compute wasserstein distance
        # distance = self._quadratic_wasserstein_distance_(speaker_vectors, generated_data)
        # # solve linear programming problem
        # sol = self._linear_programming_(distance, self.batch_size)
        # # print(f"got sol with keys {sol.keys}")
        # # approximate optimal transport
        # mapping = self._approx_OT_(sol)
        # # print(f"got mapping through _approx_OT_: {mapping.shape}")
        # real_ordered = speaker_vectors[mapping]  # match real and fake
        # # print(f"got real_ordered: {real_ordered.shape}")

        # This is for the conditional setting:
        real_ordered = torch.empty_like(speaker_vectors)
        targets_real = []
        targets_fake = []

        classes = [0, 1] # hardcoded gender classes ; TODO: fix
        for gval in classes:
            idx = (labels == gval).nonzero(as_tuple = True)[0] # indexes of where we have the gval label
            # print(idx)
            if idx.numel() < 2: # ???
                continue # skip tiny group

            xr = speaker_vectors[idx]
            xf = generated_data[idx]
            Bc = xr.size(0)

            c, A, pStart = self._get_lp_solver(Bc)
            dist = self._quadratic_wasserstein_distance_(xr, xf)
            sol = self._linear_programming_conditional_(dist, Bc, c, A, pStart)

            mapping = self._approx_OT_group_(sol, Bc)
            real_ordered[idx] = xr[mapping]

            t = torch.from_numpy(np.array(sol['x'])).float().squeeze().to(self.device)
            targets_real.append(t[:Bc].mean()) # scalar
            targets_fake.append(t[Bc:]) # (Bc, )


        # stays the same as in unconditional
        real_fake_diff = real_ordered - generated_data
        # print(f"got real_fake_diff: {real_fake_diff.shape}")

        # # construct target (unconditional)
        # target = torch.from_numpy(np.array(sol['x'])).float()
        # # print(f"got target: {target.shape}")
        # target = target.squeeze().to(self.device)
        # # print(f"squeezed target: {target.shape}")

        for i in range(opt_iterations):
            self.D.zero_grad()  # ???
            self.D_opt.zero_grad()
            generated_data.requires_grad_()
            if generated_data.grad is not None:
                generated_data.grad.data.zero_()
            # print("Passing speaker vectors through the discriminator")
            # print("discriminator on real data")
            output_real = self.D(speaker_vectors, labels)
            # print("discriminator on fake data")
            output_fake = self.D(generated_data, labels)
            output_real, output_fake = output_real.squeeze(), output_fake.squeeze()
            output_R_mean = output_real.mean(0).view(1)
            output_F_mean = output_fake.mean(0).view(1)

            # # Unconditional: losses
            # L2LossD_real = self.criterion(output_R_mean[0], target[:self.batch_size].mean())
            # L2LossD_fake = self.criterion(output_fake, target[self.batch_size:])
            # Conditional: compute losses groupwise
            L2LossD_real = 0.0
            L2LossD_fake = 0.0
            count = 0
            for gval in classes:
                idx = (labels == gval).nonzero(as_tuple=True)[0]
                if idx.numel() < 2:
                    continue # skip tiny classes

                xr = speaker_vectors[idx]
                xf = generated_data[idx]
                Bc = xr.size(0)

                c, A, pStart = self._get_lp_solver(Bc)
                dist = self._quadratic_wasserstein_distance_(xr, xf)
                sol = self._linear_programming_conditional_(dist, Bc, c, A, pStart)
                t = torch.from_numpy(np.array(sol['x'])).float().squeeze().to(self.device)

                # print(f"discriminator on real batch's elements of class {gval}")
                out_real_g = self.D(xr, labels[idx]).squeeze()
                # print(f"discriminator on fake batch's elements of class {gval}")
                out_fake_g = self.D(xf, labels[idx]).squeeze()

                L2LossD_real += self.criterion(out_real_g.mean(), t[:Bc].mean())
                L2LossD_fake += self.criterion(out_fake_g, t[Bc:])
                count += 1

            # L2LossD = 0.5 * L2LossD_real + 0.5 * L2LossD_fake
            L2LossD = 0.5 * (L2LossD_real / count) + 0.5 * (L2LossD_fake / count)
            output_fake = self.D(generated_data, labels)

            # stays 
            reg_loss_D = self._optimal_transport_regularization_(output_fake, generated_data, real_fake_diff)

            total_loss = L2LossD + self.LAMBDA * reg_loss_D # do I add y loss here or separately?

            self.losses['D'].append(float(total_loss.data)) 

            total_loss.backward()
            self.D_opt.step()

        # this is supposed to be the wasserstein distance
        wasserstein_distance = output_R_mean - output_F_mean
        self.losses['WD'].append(float(wasserstein_distance.data))

    def _generator_train_iteration(self, batch_size, labels):
        for p in self.D.parameters():
            p.requires_grad = False  # freeze critic

        self.G.zero_grad()
        self.G_opt.zero_grad()

        if isinstance(self.G, torch.nn.parallel.DataParallel):
            # z = self.G.module.sample_latent(batch_size, self.G.module.z_dim)
            z = self.G.module.sample_latent(labels, self.G.module.z_dim)
        else:
            # z = self.G.sample_latent(batch_size, self.G.z_dim)
            z = self.G.module.sample_latent(labels, self.G.z_dim)
        z.requires_grad = True

        fake = self.G(z, labels)
        output_fake = self.D(fake, labels) # TODO!!! self.D must also say the label for each datapoint -- and we must do regression on it
        output_F_mean_after = output_fake.mean(0).view(1)

        self.losses['G'].append(float(output_F_mean_after.data))

        output_F_mean_after.backward(self.mone)
        self.G_opt.step()

        self.schedulerD.step()
        self.schedulerG.step()

    def _train_epoch(self, data_loader, writer):
        for i, data in enumerate(tqdm(data_loader)):
            speaker_vectors = data[0]
            # print(f"speaker vectors shape: {speaker_vectors.shape}")
            labels = data[1] # gender labels
            # print(f"labels type and len: {type(labels), len(labels)}")
            self.num_steps += 1
            # self.tensorboard_counter += 1
            if self.gen_steps >= self.n_max_iterations:
                return
            # print("self._critic_deep_regression_(speaker_vectors, labels)")
            self._critic_deep_regression_(speaker_vectors, labels)
            # print("self._generator_train_iteration(speaker_vectors.size(0), labels)")
            self._generator_train_iteration(speaker_vectors.size(0), labels)

            D_loss_avg = np.average(self.losses['D'])
            G_loss_avg = np.average(self.losses['G'])
            wd_avg = np.average(self.losses['WD'])
            # y_avg = np.average(self.losses['y']) # new loss for labels
            print(f"D_loss_avg: {D_loss_avg}, G_loss_avg: {G_loss_avg}, wd_avg: {wd_avg}")
            
    def train(self, data_loader, writer):
        self.G.train()
        self.D.train()

        for epoch in range(self.epochs):
            if self.gen_steps >= self.n_max_iterations:
                return
            time_start_epoch = time.time()
            self._train_epoch(data_loader, writer)

            D_loss_avg = np.average(self.losses['D'])

            time_end_epoch = time.time()
            # print(f"D_loss_avg after epoch {epoch}: {D_loss_avg}")

        return self

    def sample_generator(self, labels, nograd=False, return_intermediate=False):
        # change: list of gender labels [0,1,1,0] instead of num_samples
        # we'll pass labels the Generator to sample latent sampels
        # from those latent samples, the Generator will make normal samples 
        self.G.eval()
        if isinstance(self.G, torch.nn.parallel.DataParallel):
            latent_samples = self.G.module.sample_latent(labels, self.G.module.z_dim)
        else:
            latent_samples = self.G.sample_latent(labels, self.G.z_dim)
        latent_samples = latent_samples.to(self.device)
        # print(f"Shape of latent_samples: {latent_samples.shape}")
        if nograd:
            with torch.no_grad():
                generated_data = self.G(latent_samples, labels, return_intermediate=return_intermediate) # can we do forward on both? previously latent_spaces
        else:
            generated_data = self.G(latent_samples, labels) # can we do forward on both? previously latent_spaces
        # print(f"Shape of generated_data: {generated_data.shape}")
        self.G.train()
        if return_intermediate:
            # print(f"""return_intermediate: 
                #   generated_data[0] of shape {generated_data[0].shape}, 
                #   generated_data[1] of shape {generated_data[1].shape}, 
                #   latent_samples of shape {latent_samples.shape}""")
            return generated_data[0].detach(), generated_data[1], latent_samples
        # print("returning full generated data")
        return generated_data.detach()


    def sample(self, labels):
        generated_data = self.sample_generator(labels)
        # Remove color channel -- this is initially an architecture for images that we adapt to speaker vectors; it would make sense to remove the second channel too but we wanted to replicate closely the original WGAN setup of Lux & Meyer and only add what's strictly necessary for the conditional modification
        return generated_data.data.cpu().numpy()[:, 0, :, :]

    def save_model_checkpoint(self, model_path, model_parameters, timestampStr, dataset_mean, dataset_std):
        # dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("%d-%m-%Y-%H-%M-%S")
        name = '%s_%s' % (timestampStr, 'wgan')
        model_filename = os.path.join(model_path, name)
        torch.save({
            'generator_state_dict': self.G.state_dict(),
            'critic_state_dict': self.D.state_dict(),
            'gen_optimizer_state_dict': self.G_opt.state_dict(),
            'critic_optimizer_state_dict': self.D_opt.state_dict(),
            'model_parameters': model_parameters,
            'iterations': self.num_steps,
            'mean': dataset_mean,
            'std': dataset_std
            }, model_filename)
