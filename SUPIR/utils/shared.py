import types

opts = pipe_encode = types.SimpleNamespace() 
opts.unload_models_when_training=False
opts.save_optimizer_state=False
opts.force_enable_xformers=False
opts.print_hypernet_extra=False
opts.xformers=True
opts.cross_attention_optimization="xformers"
opts.disable_opt_split_attention=False
opts.use_old_emphasis_implementation=False
opts.emphasis=None
opts.comma_padding_backtrack=20
opts.sdxl_refiner_low_aesthetic_score=2.5
opts.sdxl_refiner_high_aesthetic_score=6.0
opts.hide_ldm_prints=True
opts.fp8_storage=False
opts.opt_channelslast=False
opts.half_mode = False
opts.upcast_sampling = False

def _ldm_print(*args, **kwargs):
    if opts.hide_ldm_prints:
        return

    print(*args, **kwargs)

hypernetwork_dir = None
hypernetworks = {}
loaded_hypernetworks = []
xformers_available = False
state = None
ldm_print = _ldm_print

def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    from modules import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(hypernetwork_dir)