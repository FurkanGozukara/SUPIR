import torch
import sys
import gc
from types import MethodType
from torch.nn.functional import silu
from SUPIR.utils import devices, shared, sd_hijack_open_clip, sd_hijack_clip, sd_hijack_unet
from SUPIR.modules.hypernetworks import hypernetwork
import sgm.modules.attention
import sgm.modules.diffusionmodules.model
import sgm.modules.diffusionmodules.openaimodel
import sgm.modules.encoders.modules

from collections import defaultdict
originals = defaultdict(dict)
sd_unet_current_unet = None
sd_unet_original_forward = None 

attention_CrossAttention_forward = sgm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = sgm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = sgm.modules.diffusionmodules.model.AttnBlock.forward

if sys.platform == "darwin":
    from modules import mac_specific

optimizers = []
def patch(key, obj, field, replacement):
    """Replaces a function in a module or a class.

    Also stores the original function in this module, possible to be retrieved via original(key, obj, field).
    If the function is already replaced by this caller (key), an exception is raised -- use undo() before that.

    Arguments:
        key: identifying information for who is doing the replacement. You can use __name__.
        obj: the module or the class
        field: name of the function as a string
        replacement: the new function

    Returns:
        the original function
    """

    patch_key = (obj, field)
    if patch_key in originals[key]:
        raise RuntimeError(f"patch for {field} is already applied")

    original_func = getattr(obj, field)
    originals[key][patch_key] = original_func

    setattr(obj, field, replacement)

    return original_func
def create_unet_forward(original_forward):
    def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
        if sd_unet_current_unet is not None:
            return sd_unet_current_unet.forward(x, timesteps, context, *args, **kwargs)

        return original_forward(self, x, timesteps, context, *args, **kwargs)

    return UNetModel_forward

sgm_patched_forward = create_unet_forward(sgm.modules.diffusionmodules.openaimodel.UNetModel.forward)
sgm_original_forward = patch(__file__, sgm.modules.diffusionmodules.openaimodel.UNetModel, "forward", sgm_patched_forward)

def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps
    
def apply_optimizations(option=None):
    global current_optimizer

    undo_optimizations()

    if len(optimizers) == 0:
        # a script can access the model very early, and optimizations would not be filled by then
        current_optimizer = None
        return ''

    #ldm.modules.diffusionmodules.model.nonlinearity = silu
    #ldm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    sgm.modules.diffusionmodules.model.nonlinearity = silu
    sgm.modules.diffusionmodules.openaimodel.th = sd_hijack_unet.th

    if current_optimizer is not None:
        current_optimizer.undo()
        current_optimizer = None

    selection = option or shared.opts.cross_attention_optimization
    if selection == "Automatic" and len(optimizers) > 0:
        matching_optimizer = next(iter([x for x in optimizers if x.cmd_opt and getattr(shared.cmd_opts, x.cmd_opt, False)]), optimizers[0])
    else:
        matching_optimizer = next(iter([x for x in optimizers if x.title() == selection]), None)

    if selection == "None":
        matching_optimizer = None
    elif selection == "Automatic" and shared.opts.disable_opt_split_attention:
        matching_optimizer = None
    elif matching_optimizer is None:
        matching_optimizer = optimizers[0]

    if matching_optimizer is not None:
        print(f"Applying attention optimization: {matching_optimizer.name}... ", end='')
        matching_optimizer.apply()
        print("done.")
        current_optimizer = matching_optimizer
        return current_optimizer.name
    else:
        print("Disabling attention optimization")
        return ''


def undo_optimizations():
    # ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    # ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    # ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward

    sgm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    sgm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                vec = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = devices.cond_cast_unet(vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)
    
class StableDiffusionModelHijack:
    fixes = None
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    def __init__(self):
        #import modules.textual_inversion.textual_inversion

        self.extra_generation_params = {}
        self.comments = []

        #self.embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()
        #self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)

    def apply_optimizations(self, option=None):
        try:
            self.optimization_method = apply_optimizations(option)
        except Exception as e:
            print(f"applying cross attention optimization: {e}")
            undo_optimizations()

    def convert_sdxl_to_ssd(self, m):
        """Converts an SDXL model to a Segmind Stable Diffusion model (see https://huggingface.co/segmind/SSD-1B)"""

        delattr(m.model.diffusion_model.middle_block, '1')
        delattr(m.model.diffusion_model.middle_block, '2')
        for i in ['9', '8', '7', '6', '5', '4']:
            delattr(m.model.diffusion_model.input_blocks[7][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.input_blocks[8][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[0][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[1][1].transformer_blocks, i)
        delattr(m.model.diffusion_model.output_blocks[4][1].transformer_blocks, '1')
        delattr(m.model.diffusion_model.output_blocks[5][1].transformer_blocks, '1')
        devices.torch_gc()

    def hijack(self, m):
        conditioner = getattr(m, 'conditioner', None)
        # if conditioner:
        #     text_cond_models = []

        #     for i in range(len(conditioner.embedders)):
        #         embedder = conditioner.embedders[i]
        #         typename = type(embedder).__name__
        #         if typename == 'FrozenOpenCLIPEmbedder':
        #             embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self)
        #             conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(embedder, self)
        #             text_cond_models.append(conditioner.embedders[i])
        #         if typename == 'FrozenCLIPEmbedder':
        #             model_embeddings = embedder.transformer.text_model.embeddings
        #             model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        #             conditioner.embedders[i] = sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords(embedder, self)
        #             text_cond_models.append(conditioner.embedders[i])
        #         if typename == 'FrozenOpenCLIPEmbedder2':
        #             embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self, textual_inversion_key='clip_g')
        #             conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(embedder, self)
        #             text_cond_models.append(conditioner.embedders[i])

        #     if len(text_cond_models) == 1:
        #         m.cond_stage_model = text_cond_models[0]
        #     else:
        #         m.cond_stage_model = conditioner

        # if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation or type(m.cond_stage_model) == xlmr_m18.BertSeriesModelWithTransformation:
        #     model_embeddings = m.cond_stage_model.roberta.embeddings
        #     model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
        #     m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

        # if type(m.cond_stage_model) == sgm.modules.encoders.modules.FrozenCLIPEmbedder:
        #     model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        #     model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        #     m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        # elif type(m.cond_stage_model) == sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
        #     m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
        #     m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        apply_weighted_forward(m)
        if m.cond_stage_key == "edit":
            sd_hijack_unet.hijack_ddpm_edit()

        self.apply_optimizations()

        #self.clip = m.cond_stage_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)
        
        # import modules.models.diffusion.ddpm_edit

        # if isinstance(m, ldm.models.diffusion.ddpm.LatentDiffusion):
        #     sd_unet.original_forward = ldm_original_forward
        # elif isinstance(m, modules.models.diffusion.ddpm_edit.LatentDiffusion):
        #     sd_unet.original_forward = ldm_original_forward
        if isinstance(m, sgm.models.diffusion.DiffusionEngine):
            sd_unet_original_forward = sgm_original_forward
        else:
            sd_unet_original_forward = None

    def undo_hijack(self, m):
        conditioner = getattr(m, 'conditioner', None)
        if conditioner:
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                if isinstance(embedder, (sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords, sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):
                    embedder.wrapped.model.token_embedding = embedder.wrapped.model.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped
                if isinstance(embedder, sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):
                    embedder.wrapped.transformer.text_model.embeddings.token_embedding = embedder.wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped

            if hasattr(m, 'cond_stage_model'):
                delattr(m, 'cond_stage_model')

        # elif type(m.cond_stage_model) == sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:
        #     m.cond_stage_model = m.cond_stage_model.wrapped

        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped

        undo_optimizations()
        undo_weighted_forward(m)

        self.apply_circular(False)
        self.layers = None
        self.clip = None

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return

        self.circular_enabled = enable

        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []
        self.extra_generation_params = {}

    def get_prompt_lengths(self, text):
        if self.clip is None:
            return "-", "-"

        _, token_count = self.clip.process_texts([text])

        return token_count, self.clip.get_target_prompt_token_count(token_count)

    def redo_hijack(self, m):
        self.undo_hijack(m)
        self.hijack(m)


def weighted_loss(sd_model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
    loss = sd_model._old_get_loss(pred, target, mean=False)

    #Check if we have weights available
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
        sd_model._custom_loss_weight = w

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss
        sd_model.get_loss = MethodType(weighted_loss, sd_model)

        #Run the standard forward function, but with the patched 'get_loss'
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del sd_model._custom_loss_weight
        except AttributeError:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss
            del sd_model._old_get_loss

def apply_weighted_forward(sd_model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)

def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass

model_hijack = StableDiffusionModelHijack()