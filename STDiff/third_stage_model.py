from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from STDiff.third_stage_dataset import ThirdStageDataset
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import transforms
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import os
import torch.utils.tensorboard as tb
import joblib
import copy
import pandas as pd

from transformers import CLIPProcessor, CLIPModel
from torchmetrics import FID

def get_parser():
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="path containing logdir and checkpoint",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    return parser

class ImageLogger():
    def __init__(self, save_dir, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.save_dir = save_dir
        self.writer = tb.SummaryWriter()

    def _testtube(self, images, global_step, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            self.writer.add_image(
                tag, grid,
                global_step = global_step)

    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad()
    def log_images(self, model, images, sampled_images_3stage, sampled_images):
        log={}
        log["input"] = images
        if sampled_images is not None:
            log["samples_3stage"] = sampled_images_3stage
        if sampled_images is not None:
            log["samples"] = sampled_images
        return log

    def log_img(self, model, images, sampled_images_3stage, sampled_images, wlabels, flabels, w, t, global_step, current_epoch, batch_idx, split="train"):
        with torch.no_grad():
            images = self.log_images(model, images, sampled_images_3stage, sampled_images)

        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

        self.log_local(split, images, global_step, current_epoch, batch_idx)

        root = os.path.join(self.save_dir, "images", split)
        filename = "condition_gs-{:06}_e-{:06}_b-{:06}.csv".format(global_step, current_epoch, batch_idx)
        wlbl_val = wlabels.detach().cpu().numpy()
        weather_labels = np.array(["Sunny/Clear", "Cloudy/Overcast", "Rainy", "Snowy", "Foggy/Misty", "Windy", "Stormy/Severe",
                        "Hot/Heatwave", "Cold/Cold Wave", "Mixed/Variable"])
        # weather_labels = np.repeat(weather_labels, wlbl_val.shape[0], axis=0)
        # wlbl_val = weather_labels[np.where(wlbl_val == 1)[1]]
        flbl_val = flabels.detach().cpu().numpy()
        w_val = w.detach().cpu().numpy()
        t_val = t.detach().cpu().numpy()
        flbl_val = flbl_val.reshape((flbl_val.shape[0] * flbl_val.shape[1], -1))
        w_val = w_val.reshape((w_val.shape[0] * w_val.shape[1], -1))
        t_val = t_val.reshape((t_val.shape[0] * t_val.shape[1], -1))
        # phase = "val" if split != "train" else "train"
        phase = split
        path_scaler = ""
        lbl_scaler = joblib.load(os.path.join(path_scaler, "flow_scaler_" + phase))
        w_scaler = joblib.load(os.path.join(path_scaler, "weather_scaler_" + phase))
        t_scaler = joblib.load(os.path.join(path_scaler, "time_scaler_" + phase))
        t_val = t_scaler.inverse_transform(t_val).flatten()
        time_list = [np.datetime64(int(cur), "s") for cur in t_val]
        with open(os.path.join(root, filename), "w") as f:
            f.write("w_label:{}\n".format(wlbl_val))
            f.write("f_label:{}\n".format(lbl_scaler.inverse_transform(flbl_val)))
            f.write("weather:{}\n".format(w_scaler.inverse_transform(w_val)))
            f.write("time:{}\n".format(time_list))


class Refinement(torch.nn.Module):
    def __init__(self, input_shape, device):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(input_shape, device=device), requires_grad=True)
    def forward(self, x):
        return x + self.delta

# Modify the ResNet model to return activations from the specified layers
class FeatureExtractionModel(nn.Module):
    def __init__(self, model):
        super(FeatureExtractionModel, self).__init__()
        self.model = model
        self.layers = layers = {
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        self._features = {layer: torch.empty(0) for layer in layers.values()}

        # Register hooks to capture outputs from specified layers
        dct = dict(self.model.named_children())
        for name, layer in layers.items():
            layer = dct[name]
            layer.register_forward_hook(self.save_feature(layer))

    def save_feature(self, layer):
        def hook(model, input, output):
            self._features[layer] = output.detach()

        return hook

    def forward(self, x):
        _ = self.model(x)
        return self._features

class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size), nn.ReLU(),
                                nn.Linear(output_size, output_size)).to(self.device)

    def forward(self, weather, phase="train"):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, weather.shape[0], self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, weather.shape[0], self.hidden_size, device=self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(weather, (h0, c0))

        # Decode the hidden state of the last time step
        # if phase == "train":
        out = self.fc(out.reshape(out.shape[0] * out.shape[1], -1))
        # else:
        #     out = self.fc(out[:, -1, :])
        return out


class TimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        f0 = 1  # frequency for day cycle
        f1 = f0 / 365.25  # Compute the frequency for the year cycle
        k = 1e-2  # Set the scaling constant

        # Compute the positional embeddings
        c_d1 = torch.sin(2 * math.pi * f0 * time)
        c_d2 = torch.cos(2 * math.pi * f0 * time)

        c_y1 = torch.sin(2 * math.pi * f1 * time)
        c_y2 = torch.cos(2 * math.pi * f1 * time)

        # Combine the positional embeddings
        embeddings = torch.stack((c_d1, c_d2, c_y1, c_y2), dim=1)

        return embeddings


class AdaIN(nn.Module):
    def __init__(self, in_dim, w_dim, device):
        super().__init__()
        self.in_dim = in_dim
        self.norm = nn.InstanceNorm2d(in_dim)
        self.linear = nn.Linear(w_dim, in_dim * 2).to(device)

    def forward(self, x, w):
        x = self.norm(x)
        h = self.linear(w)
        h = h.view(h.size(0), self.in_dim * 2)
        gamma, beta = h.chunk(2, 1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        out = torch.add(torch.mul(x, (1 + gamma)), beta)
        return out

class ThirdStageModel(nn.Module):
    def __init__(self, model, ckptdir=""):
        super(ThirdStageModel, self).__init__()
        self.model = model

        self.model_org = copy.deepcopy(self.model)
        self.model_org.eval()

        self.device = self.model.device
        print("device",self.device)
        self.ckptdir = ckptdir

        self.wlabels = ["Sunny/Clear", "Cloudy/Overcast", "Rainy", "Snowy", "Foggy/Misty", "Windy", "Stormy/Severe", "Hot/Heatwave", "Cold/Cold Wave", "Mixed/Variable"]
        self.num_classes = len(self.wlabels)

        model_id = "openai/clip-vit-base-patch32"
        self.clip = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        label_tokens = self.processor(text=self.wlabels, padding=True, images=None, return_tensors='pt').to(self.device)
        self.label_features=self.clip.get_text_features(**label_tokens)

        hid_dim=512
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, hid_dim, device=self.device), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(hid_dim, self.num_classes, device=self.device))
        # self.resnet = self.resnet.to(self.device)
        self.resnet = self.resnet.eval().to(self.device)
        self.fc_w = nn.Sequential(nn.Linear(self.resnet.fc.in_features, hid_dim), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(hid_dim, self.num_classes)).to(self.device)
        self.fc_f = nn.Sequential(nn.Linear(self.resnet.fc.in_features, hid_dim), nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(hid_dim, 1)).to(self.device)
        self.resnet.fc = nn.Identity()
        # self.resnet_feat = FeatureExtractionModel(self.resnet)

        mid_dim = self.model.first_stage_model.embed_dim
        w_dim = 16
        num_layers = 1
        f_dim = 1
        self.w_mlp = nn.Sequential(WeatherLSTM(w_dim, hid_dim, num_layers, mid_dim, device=self.device)
                                   # nn.Linear(mid_dim, mid_dim, device=self.device), nn.ReLU(), nn.Dropout(0.1),
                                   # nn.Linear(mid_dim, mid_dim, device=self.device)
                                   ).to(self.device)
        self.f_mlp = nn.Sequential(WeatherLSTM(f_dim, hid_dim, num_layers, mid_dim, device=self.device)
                                   # nn.Linear(mid_dim, mid_dim, device=self.device), nn.ReLU(), nn.Dropout(0.1),
                                   # nn.Linear(mid_dim, mid_dim, device=self.device)
                                   ).to(self.device)
        self.adain = AdaIN(mid_dim, mid_dim, device=self.device)
        self.combine_mlp = nn.Sequential(
                nn.Linear(2 * mid_dim, mid_dim),
                nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(mid_dim, mid_dim)
            ).to(self.device)
        # self.fadain = AdaIN(mid_dim, mid_dim, device=self.device)
        # self.t_mlp = nn.Sequential(
        #         SinusoidalPositionEmbeddings(time_emb_dim),
        #         nn.Linear(time_emb_dim, output_dim),
        #         nn.ReLU(),
        #         nn.Linear(time_emb_dim, time_emb_dim)
        #     )
        for name, module in self.named_children():
            if name in ['w_mlp', 'f_mlp', 'adain', 'combine_mlp', 'fc_f', 'fc_w']:  # Skip 'model' and 'resnet'
                print(name)
                module.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def save_checkpoint(self, model, name):
        if not os.path.exists(self.ckptdir):
            os.makedirs(self.ckptdir, exist_ok=True)
        print("Summoning checkpoint.")
        ckpt_path = os.path.join(self.ckptdir, f"last_{name}.ckpt")
        torch.save(model.state_dict(), ckpt_path)

    def load_checkpoint(self, model, name):
        print("Loading checkpoint.")
        ckpt_path = os.path.join(self.ckptdir, f"last_{name}.ckpt")
        model.load_state_dict(torch.load(ckpt_path))
        # pl_sd = torch.load(ckpt_path)#, map_location="cpu")
        # sd = pl_sd["state_dict"]
        # model.load_state_dict(sd)

    # Function to compute reconstruction loss
    def compute_reconstruction_loss(self, decoded_images, original_images):
        loss = nn.L1Loss()(decoded_images, original_images)
        return loss

    def compute_mse_loss(self, pred_labels, gt_labels):
        loss = nn.MSELoss()(pred_labels, gt_labels)
        return loss

    # Function to compute class label loss
    def compute_entropy_loss(self, pred_labels, gt_labels):
        # Implement your class label loss computation
        # This could involve using a classification loss like CrossEntropyLoss
        loss = nn.BCEWithLogitsLoss()(pred_labels, gt_labels)
        # loss = F.binary_cross_entropy_with_logits(pred_labels, gt_labels)
        return loss
    
    def vq_loss(self, x, h, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.model.first_stage_model.get_input({"img":x.permute(0, 2, 3 ,1)}, "img")
        # x = self.model.first_stage_model.get_input({"img": ((x.permute(0, 2, 3, 1).clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)}, "img")
        x_h = self.model.first_stage_model.encoder(x)
        x_h = self.model.first_stage_model.quant_conv(x_h)
        quant, qloss, (_, _, ind) = self.model.first_stage_model.quantize(h)
        # quant, _, (_,_,_) = self.model.first_stage_model.quantize(h)
        quant = self.model.first_stage_model.post_quant_conv(quant)
        xrec = self.model.first_stage_model.decoder(quant)
        # quant, qloss, (_, _, ind) = self.model.first_stage_model.quantize(x_h)
        # quant, _, (_,_,ind) = self.model.first_stage_model.quantize(h)
        # quant = self.model.first_stage_model.post_quant_conv(quant)
        # xrec = self.model.first_stage_model.decoder(quant)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.model.first_stage_model.loss(qloss, x, xrec, optimizer_idx, self.model.first_stage_model.global_step,
                                            last_layer=self.model.first_stage_model.get_last_layer(), split="train", predicted_indices=ind)
            return aeloss, x_h

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.model.first_stage_model.loss(qloss, x, xrec, optimizer_idx, self.model.first_stage_model.global_step,
                                            last_layer=self.model.first_stage_model.get_last_layer(), split="train")
            return discloss, x_h

    def configure_optimizers(self):
        learning_rate = 0.0001
        lr_g_factor = 1.0
        lr_d = learning_rate
        lr_g = lr_g_factor*learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(
                                #   list(self.model.first_stage_model.encoder.parameters())+
                                  list(self.model.first_stage_model.decoder.parameters())+
                                  list(self.model.first_stage_model.quantize.parameters())+
                                #   list(self.model.first_stage_model.quant_conv.parameters())+
                                  list(self.model.first_stage_model.post_quant_conv.parameters())+
                                  list(self.fc_f.parameters()) + list(self.fc_w.parameters())+
                                  list(self.f_mlp.parameters())+
                                  list(self.w_mlp.parameters())+
                                  list(self.combine_mlp.parameters())+
                                  list(self.adain.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.model.first_stage_model.loss.discriminator.parameters(),
        #                             lr=lr_d, betas=(0.5, 0.9))
        opt_pred = torch.optim.Adam(list(self.fc_f.parameters())+list(self.fc_w.parameters()),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.model.first_stage_model.scheduler_config is not None:
            scheduler = instantiate_from_config(self.model.first_stage_model.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_pred], scheduler
        return [opt_ae, opt_pred], []

    def f1_score(self, y_pred, y_true, threshold=0.8):
        """
        Calculate F1 Score for multilabel classification.
        Args:
        y_pred (torch.Tensor): Predictions from the model (probabilities).
        y_true (torch.Tensor): Actual labels.
        threshold (float): Threshold for converting probabilities to binary output.

        Returns:
        float: F1 Score
        """
        # Binarize predictions and labels
        y_pred = (y_pred > threshold).int()
        y_true = (y_true > threshold).int()
        
        # True positives, false positives, and false negatives
        tp = (y_true * y_pred).sum(dim=1).float()  # Element-wise multiplication for intersection
        fp = ((1 - y_true) * y_pred).sum(dim=1).float()
        fn = (y_true * (1 - y_pred)).sum(dim=1).float()
        
        # Precision, recall, and F1 for each label
        epsilon = 1e-7  # To avoid division by zero
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        # Average F1 score across all labels
        avg_f1 = f1.mean().item()
        return avg_f1, tp.mean().item()

    
    def on_val_start(self):
        # ckpt_path = os.path.join(ckptdir, "last.ckpt")
        # print(f"Loading model from {ckpt_path}")
        # pl_sd = torch.load(ckpt_path)#, map_location="cpu")
        # sd = pl_sd["state_dict"]
        # model = instantiate_from_config(config.model)
        # m, u = model.load_state_dict(sd, strict=False)
        # self.model.to(self.device)
        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # for name, module in self.named_children():
        #     module.eval()
        #     for param in module.parameters():
        #         param.requires_grad = False               
                
        # for name, param in model.named_parameters():
        #     print(f'{name}: requires_grad={param.requires_grad}')
        return

    def on_train_start(self):
        # ckpt_path = os.path.join(ckptdir, "last.ckpt")
        # print(f"Loading model from {ckpt_path}")
        # pl_sd = torch.load(ckpt_path)#, map_location="cpu")
        # sd = pl_sd["state_dict"]
        # model = instantiate_from_config(config.model)
        # m, u = model.load_state_dict(sd, strict=False)
        # self.model.to(self.device)
        for name, module in self.named_children():
            if name in ['w_mlp', 'f_mlp', 'adain', 'combine_mlp', 'fc_f', 'fc_w']:  # Skip 'model' and 'resnet'
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
            else:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False     

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.first_stage_model.decoder.train()
        self.model.first_stage_model.quantize.train()
        self.model.first_stage_model.post_quant_conv.train()
        self.model.first_stage_model.loss.discriminator.train()
        for param in self.model.first_stage_model.decoder.parameters():
            param.requires_grad = True
        for param in self.model.first_stage_model.quantize.parameters():
            param.requires_grad = True
        for param in self.model.first_stage_model.post_quant_conv.parameters():
            param.requires_grad = True
        for param in self.model.first_stage_model.loss.discriminator.parameters():
            param.requires_grad = True          
                
        # for name, param in model.named_parameters():
        #     print(f'{name}: requires_grad={param.requires_grad}')
        return
        
    # def train(self, loader, image_logger):

    #     # vae = VAE()
        
    #     # _, x, _, _, _, _ = next(iter(loader))
    #     # refine = Refinement(x.shape[1:], self.device).to(self.device)
    #     # refine.train()
    #     # optimizer = optim.Adam(list(self.model.first_stage_model.decoder.parameters()), lr=0.001, betas=(0.9, 0.999))
    #     # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    #     # self.model.first_stage_model.loss = VQLPIPSWithDiscriminator(disc_conditional = False, disc_in_channels = 3, disc_num_layers = 2,
    #     #                                                              disc_start = 1, disc_weight = 0.6, codebook_weight = 1.0, n_classes=self.model.first_stage_model.n_embed).to(self.device)
    #     self.model.first_stage_model.loss = VQLPIPSWithDiscriminator(disc_conditional = False, disc_in_channels = 3, disc_num_layers = 2,
    #                                                             disc_start = 1, disc_weight = 0.6, codebook_weight = 1.0, n_classes=self.model.first_stage_model.n_embed).to(self.device)
    #     # Initialize optimizer
    #     optimizer, schedule = self.configure_optimizers()
        
    #     optimizer_idx = 1 

    #     lambda_rec=1
    #     lambda_wlbl = 0.5
    #     lambda_flbl = 0.02
    #     lambda_percept = 0.5
    #     lambda_cls_clip = 0.05

    #     # Training loop
    #     num_epochs = 400
    #     global_step = 0
    #     for epoch in range(num_epochs):
    #         self.on_train_start()
    #         total_loss = 0
    #         for batch_idx, batch in enumerate(loader):
    #             images, latents, w, wlabels, _, flabels, t = batch
    #             images = images.to(self.device)
    #             latents = latents.to(self.device)
    #             wlabels = wlabels.to(self.device)

    #             flabels = flabels.to(self.device)
    #             w = w.to(self.device)
    #             t = t.to(self.device)

    #             sampled_images = self.model_org.first_stage_model.decode(latents.detach(), force_not_quantize=False)

    #             w_embed = self.w_mlp(w)
    #             f_embed = self.f_mlp(flabels.unsqueeze(1))
    #             # latent = self.adain(latents, w_embed)
    #             combined_features = self.combine_mlp(torch.cat((w_embed, f_embed), dim=1))
    #             latents = self.adain(latents, combined_features)

    #             # # latents = refine(latents)
    #             # # vae_loss = vae.loss(latent)
    #             # sampled_images = self.model_org.first_stage_model.decode(latents, force_not_quantize=False)
    #             # decoded_images = self.model.first_stage_model.decode(latents.detach(), force_not_quantize=False)
    #             # # sampled_images = self.model_org.decode_first_stage(latents)
    #             # # decoded_images = self.model.decode_first_stage(latent)

    #             # Compute losses
    #             # reconstruction_loss = self.compute_reconstruction_loss(decoded_images, images)
    #             optimizer_idx = 1
    #             reconstruction_loss, x_h = self.vq_loss(images, latents.detach(), optimizer_idx=optimizer_idx)
    #             # reconstruction_loss += self.compute_reconstruction_loss(decoded_images, images)

    #             loss = reconstruction_loss
    #             optimizer[optimizer_idx].zero_grad()
    #             loss.backward()
    #             optimizer[optimizer_idx].step()

    #             decoded_images = self.model.first_stage_model.decode(latents, force_not_quantize=False)

    #             resnet_out = self.resnet(decoded_images)
    #             pred_wlabels = self.fc_w(resnet_out)
    #             # print(pred_wlabels.shape,wlabels.shape)
    #             wlabel_loss = self.compute_entropy_loss(pred_wlabels, wlabels)

    #             pred_flabels = self.fc_f(resnet_out)
    #             # print(pred_flabels.shape,flabels.shape)
    #             flabel_loss = self.compute_mse_loss(pred_flabels, flabels)

    #             to_pil = transforms.ToPILImage()
    #             images_pil=[to_pil (((img.clamp(-1., 1.)+ 1.0) / 2.0 * 255).to(torch.uint8)) for img in images]
    #             inputs = self.processor(text=self.wlabels, images=images_pil, padding=True, return_tensors='pt')
    #             # inputs = self.processor(text=None, images=images_pil, padding=True, return_tensors='pt')
    #             inputs = {key: value.to(self.device) for key, value in inputs.items()}
    #             images_clip = self.clip(**inputs)
    #             # image_features = self.clip.get_image_features(**images)
    #             # pred_labels = torch.matmul(image_features,self.label_features.T)
    #             # clip_class_label_loss = self.compute_reconstruction_loss(pred_labels, wlabels)
    #             decoded_images_pil=[to_pil (((img.clamp(-1., 1.)+ 1.0) / 2.0 * 255).to(torch.uint8)) for img in decoded_images]
    #             inputs = self.processor(text=self.wlabels, images=decoded_images_pil, padding=True, return_tensors='pt')
    #             # inputs = self.processor(text=None, images=decoded_images_pil, padding=True, return_tensors='pt')
    #             inputs = {key: value.to(self.device) for key, value in inputs.items()}
    #             decoded_images_clip = self.clip(**inputs)
    #             clip_class_label_loss = self.compute_entropy_loss(decoded_images_clip.logits_per_image, torch.softmax(images_clip.logits_per_image, dim=1))
    #             # clip_class_label_loss = self.compute_reconstruction_loss(decoded_images_clip.logits_per_image, images_clip.logits_per_image)
    #             # clip_class_label_loss = self.compute_reconstruction_loss(decoded_images_clip, images_clip)
    #             # clip_class_label_loss = self.compute_entropy_loss(decoded_images_clip.logits_per_image, wlabels)

    #             # target_features = self.resnet_feat(images)
    #             # predicted_features = self.resnet_feat(decoded_images)
    #             # percept_loss = 0
    #             # for layer in target_features.keys():
    #             #     percept_loss += self.compute_mse_loss(predicted_features[layer], target_features[layer])

    #             # Total loss
    #             optimizer_idx = 0
    #             reconstruction_loss, x_h = self.vq_loss(images, latents, optimizer_idx=optimizer_idx)
    #             loss = lambda_rec * reconstruction_loss + lambda_wlbl * wlabel_loss + lambda_flbl * flabel_loss # + lambda_cls_clip * clip_class_label_loss + args.lambda_vae * vae_loss
    #             # print("rec: ",reconstruction_loss, "wlabel: ", wlabel_loss, "flabel: ", flabel_loss, "cliplabel:", clip_class_label_loss, "loss: ", loss)
    #             total_loss += loss

    #             # Backpropagation
    #             optimizer[optimizer_idx].zero_grad()
    #             loss.backward()
    #             optimizer[optimizer_idx].step()
    #             # scheduler.step()

    #             # image_logger.log_img(self.model, images, decoded_images, sampled_images, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="train")
    #             # print("**************start third_stage validation**************")
    #             # self.test(loader, image_logger)
    #             # self.on_train_start()
    #             # print("**************end third_stage validation**************")

    #         global_step += 1
    #         if epoch % 50 == 0:
    #             self.save_checkpoint(self.model)
    #             image_logger.log_img(self.model, images, decoded_images, sampled_images, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="train")
    #             print("**************start third_stage validation**************")
    #             root_data = image_logger.save_dir+"_data"
    #             self.test(torch.utils.data.DataLoader(ThirdStageDataset(root_data, split="val"), batch_size=16, shuffle=False), image_logger)
    #             # self.on_train_start()
    #             print("**************end third_stage validation**************")
    #         # Print loss
    #         print(f'Epoch {epoch}, Total Loss: {total_loss.item() / len(loader)}')

    def train(self, loader, image_logger):
        # Initialize optimizer
        optimizer, schedule = self.configure_optimizers()
        
        optimizer_idx = 1

        lambda_rec=1
        lambda_wlbl = 0.5
        lambda_flbl = 0.02
        lambda_percept = 0.5
        lambda_cls_clip = 0.05

        # Training loop
        num_epochs = 200
        global_step = 0
        for epoch in range(num_epochs):

            for name, module in self.named_children():
                if name in ['fc_f', 'fc_w']:  # Skip 'model' and 'resnet'
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
            self.model.eval()

            total_loss = 0
            for batch_idx, batch in enumerate(loader):
                images, latents, w, wlabels, _, flabels, t = batch
                images = images.to(self.device)
                latents = latents.to(self.device)
                wlabels = wlabels.to(self.device)

                flabels = flabels.to(self.device)
                w = w.to(self.device)
                t = t.to(self.device)

                decoded_images = self.model.first_stage_model.decode(latents, force_not_quantize=False)

                resnet_out = self.resnet(decoded_images)

                pred_wlabels = self.fc_w(resnet_out)
                # print(pred_flabels.shape,flabels.shape)
                wlabel_loss = self.compute_entropy_loss(pred_wlabels, wlabels)

                pred_flabels = self.fc_f(resnet_out)
                # print(pred_flabels.shape,flabels.shape)
                flabel_loss = self.compute_mse_loss(pred_flabels, flabels)

                # Total loss
                # loss = lambda_flbl * flabel_loss + lambda_wlbl * wlabel_loss
                loss = flabel_loss
                total_loss += loss

                # Backpropagation
                optimizer[optimizer_idx].zero_grad()
                loss.backward()
                optimizer[optimizer_idx].step()
                # scheduler.step()

                # image_logger.log_img(self.model, images, decoded_images, sampled_images, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="train")
                # print("**************start third_stage validation**************")
                # self.test(loader, image_logger)
                # self.on_train_start()
                # print("**************end third_stage validation**************")

            if epoch == num_epochs-1:
                self.save_checkpoint(self.fc_f, "fc_f")
                self.save_checkpoint(self.fc_w, "fc_w")

            # Print loss
            print(f'Epoch {epoch}, Total Loss: {total_loss.item() / len(loader)}')

    def test(self, loader, image_logger, save_dir, fol_name):
        global_step = 0
        epoch = 0
        total_fid = 0
        fid_val = 0
        total_acc = 0
        total_f1 = 0
        total_acc_clip = 0
        total_f1_clip = 0
        flabel_error = 0
        plabel_list = np.array([])
        label_list = np.array([])
        t_list = np.array([])
        # refine = Refinement(x.shape[1:], self.device).to(self.device)
        self.load_checkpoint(self.fc_f, "fc_f")
        self.load_checkpoint(self.fc_w, "fc_w")

        fid = FID().cuda(device=self.device)

        with torch.no_grad():
            self.on_val_start()
            for batch_idx, batch in enumerate(loader):
                images, latents, w, wlabels, _, flabels, t = batch
                images = images.to(self.device)
                latents = latents.to(self.device)
                wlabels = wlabels.to(self.device)

                flabels = flabels.to(self.device)
                w = w.to(self.device)
                t = t.to(self.device)

                # latents = refine(latents)
                decoded_images = self.model.first_stage_model.decode(latents, force_not_quantize=False)
                # sampled_images = self.model_org.decode_first_stage(latents)
                # decoded_images = self.model.decode_first_stage(latent)

                fid.update(((images.clamp(-1., 1.) + 1.0) / 2.0 * 255).type(torch.uint8).cuda(device=self.device), real=True)
                fid.update(((decoded_images.clamp(-1., 1.) + 1.0) / 2.0 * 255).type(torch.uint8).cuda(device=self.device), real=False)
                # fid_val = fid.compute()
                # total_fid+=fid_val

                to_pil = transforms.ToPILImage()
                images_pil=[to_pil (((img.clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)) for img in images]
                inputs = self.processor(text=self.wlabels, images=images_pil, padding=True, return_tensors='pt')
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                images_clip = self.clip(**inputs)
                decoded_images_pil=[to_pil (((img.clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)) for img in decoded_images]
                inputs = self.processor(text=self.wlabels, images=decoded_images_pil, padding=True, return_tensors='pt')
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                decoded_images_clip = self.clip(**inputs)

                total_acc_clip+=(torch.argmax(decoded_images_clip.logits_per_image, dim=1)==torch.argmax(images_clip.logits_per_image, dim=1)).float().mean()
                # f1, acc=self.f1_score(decoded_images_clip.logits_per_image, images_clip.logits_per_image)
                # total_f1_clip+=f1
                # total_acc_clip+=acc
                # total_acc_clip+=(((decoded_images_clip.logits_per_image>0.8).int() * wlabels.int()).sum(dim=1)/wlabels.int().sum(dim=1)).mean()
                # total_acc_clip+=self.f1_score(decoded_images_clip.logits_per_image, torch.sigmoid(images_clip.logits_per_image, dim=1))
                resnet_out = self.resnet(decoded_images)
                pred_wlabels = self.fc_w(resnet_out)
                # total_acc+=(((pred_wlabels>0.8).int() * wlabels.int()).sum(dim=1)/wlabels.int().sum(dim=1)).mean()
                # total_acc+=(torch.argmax(pred_wlabels, dim=1)==torch.argmax(wlabels, dim=1)).float().mean()
                f1, acc = self.f1_score(pred_wlabels, wlabels)
                total_f1+=f1
                total_acc+=acc

                pred_flabels = self.fc_f(resnet_out)
                # flabel_error += self.compute_mse_loss(pred_flabels, flabels)

                lbl_val = flabels.detach().cpu().numpy()
                plbl_val = pred_flabels.detach().cpu().numpy()
                t_val = t.detach().cpu().numpy()
                lbl_val = lbl_val.reshape(lbl_val.shape[0] * lbl_val.shape[1])
                plbl_val = plbl_val.reshape(plbl_val.shape[0] * plbl_val.shape[1])
                t_val = t_val.reshape(t_val.shape[0] * t_val.shape[1])

                label_list = np.append(label_list, lbl_val)
                plabel_list = np.append(plabel_list, plbl_val)
                t_list = np.append(t_list, t_val)

                flabel_error += np.linalg.norm(plabel_list-label_list, 2)

                global_step += 1
                # image_logger.log_img(self.model, images, decoded_images, None, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="test")

            total_fid=fid.compute().item()

        # phase = "val" if split != "train" else "train"
        root = save_dir # os.path.join(save_dir, "images", "test")
        filename = f"flow_preds_{fol_name}.csv"
        path_scaler = ""
        lbl_scaler = joblib.load(os.path.join(path_scaler, f"flow_scaler_test_{fol_name}"))
        t_scaler = joblib.load(os.path.join(path_scaler, f"time_scaler_test_{fol_name}"))
        t_list = t_scaler.inverse_transform(t_list.reshape((t_list.shape[0],1))).flatten()
        time_list = np.array([np.datetime64(int(cur), "s") for cur in t_list])
        dct = {}
        with open(os.path.join(root, filename), "w") as f:
            dct["pred_f_label"] = lbl_scaler.inverse_transform(plabel_list.reshape((plabel_list.shape[0],1))).flatten()
            dct["f_label"] = lbl_scaler.inverse_transform(label_list.reshape((label_list.shape[0],1))).flatten()
            dct["time"] = time_list
            df = pd.DataFrame.from_dict(dct)
            df.to_csv(f)

        print(f'Total test w label accuracy: {total_acc/len(loader)}, f1-score: {total_f1/len(loader)}')
        print(f'Total test clip accuracy: {total_acc_clip/len(loader)}')
        print(f'Total f label error: {flabel_error.item()/len(loader)}')
        print(f'Total test FID: {total_fid/len(loader)}')

        import matplotlib.pyplot as plt
        sorted_indices = np.argsort(time_list)
        sorted_time_list = time_list[sorted_indices]
        sorted_plabel_list = plabel_list[sorted_indices]
        sorted_label_list = label_list[sorted_indices]

        fig=plt.figure()
        plt.plot(sorted_time_list,sorted_plabel_list, label="Prediction",color='blue')
        plt.plot(sorted_time_list,sorted_label_list, label="Ground truth",color='red')
        plt.tick_params(axis='x', rotation=45,labelsize=10)
        plt.ylabel("Flow")
        plt.xlabel("time")
        plt.title('Error')
        plt.legend()
        plt.savefig(os.path.join(root, f'label_{fol_name}.png'), bbox_inches='tight')

        return

    # def on_train_start(self):
    #     self.model.eval()
    #     for param in self.model.parameters():
    #         param.requires_grad = False
    #     for name, module in self.named_children():
    #         if name in ['fc_f', 'fc_w']:
    #             module.train()
    #             for param in module.parameters():
    #                 param.requires_grad = True
    #         else:
    #             module.eval()
    #             for param in module.parameters():
    #                 param.requires_grad = False  
    #     return
    
    # def train(self, loader, image_logger):
         
    #     self.on_train_start()
    #     optimizer = optim.Adam(list(self.fc_f.parameters())+list(self.fc_w.parameters()), lr=0.001, betas=(0.9, 0.999))
    #     lambda_wlbl = 0.5
    #     lambda_flbl = 0.02

    #     # Training loop
    #     num_epochs = 400
    #     global_step = 0
    #     for epoch in range(num_epochs):
    #         total_loss = 0
    #         for batch_idx, batch in enumerate(loader):
    #             images, latents, w, wlabels, _, flabels, t = batch
    #             images = images.to(self.device)
    #             latents = latents.to(self.device)
    #             wlabels = wlabels.to(self.device)

    #             flabels = flabels.to(self.device)
    #             w = w.to(self.device)
    #             t = t.to(self.device)
    #             sampled_images = decoded_images = self.model.first_stage_model.decode(latents.detach(), force_not_quantize=False)

    #             resnet_out = self.resnet(decoded_images)
    #             pred_wlabels = self.fc_w(resnet_out)
    #             # print(pred_wlabels.shape,wlabels.shape)
    #             wlabel_loss = self.compute_entropy_loss(pred_wlabels, wlabels)

    #             pred_flabels = self.fc_f(resnet_out)
    #             # print(pred_flabels.shape,flabels.shape)
    #             flabel_loss = self.compute_mse_loss(pred_flabels, flabels)

    #             loss = lambda_wlbl * wlabel_loss + lambda_flbl * flabel_loss
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss

    #             # # image_logger.log_img(self.model, images, decoded_images, sampled_images, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="train")
    #             # # print("**************start third_stage validation**************")
    #             # # self.test(loader, image_logger)
    #             # # self.on_train_start()
    #             # # print("**************end third_stage validation**************")

    #         global_step += 1
    #         if epoch % 50 == 0:
    #             self.save_checkpoint(self.model)
    #             image_logger.log_img(self.model, images, decoded_images, sampled_images, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="train")
    #             print("**************start third_stage validation**************")
    #             root_data = image_logger.save_dir+"_data"
    #             self.test(torch.utils.data.DataLoader(ThirdStageDataset(root_data, split="val"), batch_size=16, shuffle=False), image_logger)
    #             self.on_train_start()
    #             print("**************end third_stage validation**************")
    #         # Print loss
    #         print(f'Epoch {epoch}, Total Loss: {total_loss.item() / len(loader)}')

    # def test(self, loader, image_logger):
    #     global_step = 0
    #     epoch = 0
    #     total_fid = 0
    #     total_acc = 0
    #     total_acc_clip = 0
    #     flabel_error = 0
    #     self.on_val_start()
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(loader):
    #             images, latents, w, wlabels, _, flabels, t = batch
    #             images = images.to(self.device)
    #             latents = latents.to(self.device)
    #             wlabels = wlabels.to(self.device)

    #             flabels = flabels.to(self.device)
    #             w = w.to(self.device)
    #             t = t.to(self.device)

    #             sampled_images=decoded_images = self.model.first_stage_model.decode(latents, force_not_quantize=False)

    #             fid = FID().cuda(device=self.device)
    #             fid.update(((images.clamp(-1., 1.) + 1.0) / 2.0 * 255).type(torch.uint8).cuda(device=self.device), real=True)
    #             fid.update(((decoded_images.clamp(-1., 1.) + 1.0) / 2.0 * 255).type(torch.uint8).cuda(device=self.device), real=False)
    #             # total_fid+=fid.compute()

    #             to_pil = transforms.ToPILImage()
    #             images_pil=[to_pil (((img.clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)) for img in images]
    #             inputs = self.processor(text=self.wlabels, images=images_pil, padding=True, return_tensors='pt')
    #             inputs = {key: value.to(self.device) for key, value in inputs.items()}
    #             images_clip = self.clip(**inputs)
    #             decoded_images_pil=[to_pil (((img.clamp(-1., 1.) + 1.0) / 2.0 * 255).to(torch.uint8)) for img in decoded_images]
    #             inputs = self.processor(text=self.wlabels, images=decoded_images_pil, padding=True, return_tensors='pt')
    #             inputs = {key: value.to(self.device) for key, value in inputs.items()}
    #             decoded_images_clip = self.clip(**inputs)

    #             total_acc_clip+=(torch.argmax(decoded_images_clip.logits_per_image, dim=1)==torch.argmax(images_clip.logits_per_image, dim=1)).float().mean()
    #             # total_acc_clip+=(((decoded_images_clip.logits_per_image>0.8).int() * wlabels.int()).sum(dim=1)/wlabels.int().sum(dim=1)).mean()
    #             # total_acc_clip+=self.f1_score(decoded_images_clip.logits_per_image, torch.sigmoid(images_clip.logits_per_image, dim=1))
    #             resnet_out = self.resnet(decoded_images)
    #             pred_wlabels = self.fc_w(resnet_out)
    #             # total_acc+=(((pred_wlabels>0.8).int() * wlabels.int()).sum(dim=1)/wlabels.int().sum(dim=1)).mean()
    #             # total_acc+=(torch.argmax(pred_wlabels, dim=1)==torch.argmax(wlabels, dim=1)).float().mean()
    #             total_acc+=self.f1_score(pred_wlabels, wlabels)

    #             pred_flabels = self.fc_f(resnet_out)
    #             flabel_error += self.compute_mse_loss(pred_flabels, flabels)

    #             global_step += 1
    #             image_logger.log_img(self.model, images, decoded_images, sampled_images, wlabels, flabels, w, t, global_step, epoch, batch_idx, split="val")

    #         total_fid=fid.compute()

    #     print(f'Total test w label accuracy: {total_acc/len(loader)}')
    #     print(f'Total test clip accuracy: {total_acc_clip/len(loader)}')
    #     print(f'Total test accuracy: {flabel_error.item()/len(loader)}')
    #     print(f'Total test FID: {total_fid/len(loader)}')
    #     return
    
    def run(self, logdir):
        print("**************start third_stage**************")
        # parser = get_parser()
        # args = parser.parse_args()

        # base_configs = sorted(glob.glob(os.path.join(args.path, "configs/*.yaml")))
        # configs = [OmegaConf.load(cfg) for cfg in base_configs]
        # config = OmegaConf.merge(*configs, cli)
        batch_frequency=16
        max_images=32
        root = os.path.join(logdir, "third_stage")
        self.ckptdir = os.path.join(root, "checkpoints")
        root_data = os.path.join(logdir, "third_stage_data")

        image_logger = ImageLogger(root, batch_frequency=batch_frequency, max_images=max_images)

        print("********** train **********")
        data_ft = ThirdStageDataset(root_data, split="train")
        loader = torch.utils.data.DataLoader(data_ft, batch_size=4, shuffle=True)
        self.train(loader, image_logger)

        print("********** test **********")
        fol_name = "ar"
        root_data = os.path.join(logdir, f"third_stage_data_{fol_name}")

        data_ft = ThirdStageDataset(root_data, split="test")
        loader = torch.utils.data.DataLoader(data_ft, batch_size=4, shuffle=False)
        self.test(loader, image_logger, root, fol_name)
