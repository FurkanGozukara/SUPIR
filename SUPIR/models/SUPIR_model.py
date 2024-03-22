import copy
import random

import torch
from pytorch_lightning import seed_everything

from SUPIR.utils.colorfix import wavelet_reconstruction, adaptive_instance_normalization
from SUPIR.utils.tilevae import VAEHook
from sgm.models.diffusion import DiffusionEngine
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
from sgm.util import instantiate_from_config
from ui_helpers import printt


class SUPIRModel(DiffusionEngine):
    def __init__(self, control_stage_config, ae_dtype='bf16', diffusion_dtype='bf16', p_p='', n_p='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        printt("Loading control model.")
        control_model = instantiate_from_config(control_stage_config)
        printt("Instantiated control model.")
        self.model.load_control_model(control_model)
        printt("Loaded control model.")
        self.first_stage_model.denoise_encoder = copy.deepcopy(self.first_stage_model.encoder)
        printt("Copied denoise encoder.")
        self.sampler_config = kwargs['sampler_config']
        self.previous_sampler_config = None  # Store the previous sampler configuration
        self.sampler = None  # Initialize sampler as None
        assert (ae_dtype in ['fp32', 'fp16', 'bf16']) and (diffusion_dtype in ['fp32', 'fp16', 'bf16'])
        if ae_dtype == 'fp32':
            ae_dtype = torch.float32
        elif ae_dtype == 'fp16':
            raise RuntimeError('fp16 cause NaN in AE')
        elif ae_dtype == 'bf16':
            ae_dtype = torch.bfloat16
        if diffusion_dtype == 'fp32':
            diffusion_dtype = torch.float32
        elif diffusion_dtype == 'fp16':
            diffusion_dtype = torch.float16
        elif diffusion_dtype == 'bf16':
            diffusion_dtype = torch.bfloat16

        self.ae_dtype = ae_dtype
        self.model.dtype = diffusion_dtype

        self.p_p = p_p
        self.n_p = n_p

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            z = self.first_stage_model.encode(x)
        z.mul_(self.scale_factor)  # In-place multiplication
        return z

    @torch.no_grad()
    def encode_first_stage_with_denoise(self, x, use_sample=True, is_stage1=False):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            h = self.first_stage_model.denoise_encoder_s1(x) if is_stage1 else self.first_stage_model.denoise_encoder(x)
            moments = self.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            z = posterior.sample() if use_sample else posterior.mode()
        z.mul_(self.scale_factor)  # In-place multiplication
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", dtype=self.ae_dtype):
            out = self.first_stage_model.decode(z)
        return out.float()

    @torch.no_grad()
    def batchify_denoise(self, x, is_stage1=False):
        """
        [N, C, H, W], [-1, 1], RGB
        """
        x = self.encode_first_stage_with_denoise(x, use_sample=False, is_stage1=is_stage1)
        return self.decode_first_stage(x)

    @torch.no_grad()
    def batchify_sample(self, x, p, p_p='default', n_p='default', num_steps=100, restoration_scale=4.0, s_churn=0,
                        s_noise=1.003, cfg_scale=4.0, seed=-1,
                        num_samples=1, control_scale=1, color_fix_type='None', use_linear_cfg=False,
                        use_linear_control_scale=False,
                        cfg_scale_start=1.0, control_scale_start=0.0, sampler_cls=None, **kwargs):
        """
        [N, C], [-1, 1], RGB
        """
        assert len(x) == len(p)
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']
        if not sampler_cls:
            sampler_cls = f"sgm.modules.diffusionmodules.sampling.RestoreDPMPP2MSampler"

        n = len(x)
        if num_samples > 1:
            assert n == 1
            n = num_samples
            x = x.repeat(n, 1, 1, 1)
            p = p * n

        if p_p == 'default':
            p_p = self.p_p
        if n_p == 'default':
            n_p = self.n_p
        new_sampler_config = {
            "target": sampler_cls,
            "params": {
                "num_steps": num_steps,
                "restore_cfg": restoration_scale,
                "s_churn": s_churn,
                "s_noise": s_noise,
                "discretization_config": {
                    "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
                },
                "guider_config": {
                    "target": "sgm.modules.diffusionmodules.guiders.LinearCFG",
                    "params": {
                        "scale": cfg_scale_start if use_linear_cfg else cfg_scale,
                        "scale_min": cfg_scale
                    }
                }
            }
        }

        # Update sampler configuration
        # Check if the sampler needs to be re-instantiated
        if self.previous_sampler_config != new_sampler_config or self.sampler is None:
            self.sampler_config = new_sampler_config
            printt("Instantiating sampler.")
            del self.sampler
            self.sampler = instantiate_from_config(self.sampler_config)
            self.previous_sampler_config = new_sampler_config
            printt("Instantiated sampler.")

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        printt("Encoding first stage with denoise...")
        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        printt("Encoded first stage with denoise...")
        x_stage1 = self.decode_first_stage(_z)
        printt("Decoded first stage...")
        z_stage1 = self.encode_first_stage(x_stage1)
        printt("Encoded first stage...")

        c, uc = self.prepare_condition(_z, p, p_p, n_p, n)

        printt("Loading denoiser.")
        denoiser = lambda input, sigma, c, control_scale: self.denoiser(
            self.model, input, sigma, c, control_scale, **kwargs
        )
        printt("Loaded denoiser.")

        noised_z = torch.randn_like(_z).to(_z.device)
        printt("Sampling...")
        _samples = self.sampler(denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1, control_scale=control_scale,
                                use_linear_control_scale=use_linear_control_scale,
                                control_scale_start=control_scale_start)
        printt("Sampled.")
        output = self.decode_first_stage(_samples)
        printt("Decoded output.")
        if color_fix_type == 'Wavelet':
            output = wavelet_reconstruction(output, x_stage1)
            printt("Wavelet reconstructed.")
        elif color_fix_type == 'AdaIn':
            output = adaptive_instance_normalization(output, x_stage1)
            printt("AdaIn reconstructed.")
        return output

    def init_tile_vae(self, encoder_tile_size=512, decoder_tile_size=64, use_fast=False):
        self.first_stage_model.denoise_encoder.original_forward = self.first_stage_model.denoise_encoder.forward
        self.first_stage_model.encoder.original_forward = self.first_stage_model.encoder.forward
        self.first_stage_model.decoder.original_forward = self.first_stage_model.decoder.forward
        self.first_stage_model.denoise_encoder.forward = VAEHook(
            self.first_stage_model.denoise_encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=use_fast, color_fix=False, to_gpu=True)
        self.first_stage_model.encoder.forward = VAEHook(
            self.first_stage_model.encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=use_fast, color_fix=False, to_gpu=True)
        self.first_stage_model.decoder.forward = VAEHook(
            self.first_stage_model.decoder, decoder_tile_size, is_decoder=True, fast_decoder=use_fast,
            fast_encoder=False, color_fix=False, to_gpu=True)

    def prepare_condition(self, _z, p, p_p, n_p, n):
        batch = {'original_size_as_tuple': torch.tensor([1024, 1024]).repeat(n, 1).to(_z.device),
                 'crop_coords_top_left': torch.tensor([0, 0]).repeat(n, 1).to(_z.device),
                 'target_size_as_tuple': torch.tensor([1024, 1024]).repeat(n, 1).to(_z.device),
                 'aesthetic_score': torch.tensor([9.0]).repeat(n, 1).to(_z.device), 'control': _z}

        batch_uc = copy.deepcopy(batch)
        batch_uc['txt'] = [n_p for _ in p]

        if not isinstance(p[0], list):
            batch['txt'] = [''.join([_p, p_p]) for _p in p]
            with torch.cuda.amp.autocast(dtype=self.ae_dtype):
                c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
        else:
            assert len(p) == 1, 'Support bs=1 only for local prompt conditioning.'
            p_tiles = p[0]
            c = []
            for i, p_tile in enumerate(p_tiles):
                batch['txt'] = [''.join([p_tile, p_p])]
                with torch.cuda.amp.autocast(dtype=self.ae_dtype):
                    if i == 0:
                        _c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
                    else:
                        _c, _ = self.conditioner.get_unconditional_conditioning(batch, None)
                c.append(_c)
        return c, uc

    def move_to(self, device):
        """Moves VAEHook's .net objects to the specified device."""
        if hasattr(self.first_stage_model, 'denoise_encoder') and isinstance(self.first_stage_model.denoise_encoder,
                                                                             VAEHook):
            self.first_stage_model.denoise_encoder.to(device)
        if hasattr(self.first_stage_model, 'encoder') and isinstance(self.first_stage_model.encoder, VAEHook):
            self.first_stage_model.encoder.to(device)
        if hasattr(self.first_stage_model, 'decoder') and isinstance(self.first_stage_model.decoder, VAEHook):
            self.first_stage_model.decoder.to(device)
