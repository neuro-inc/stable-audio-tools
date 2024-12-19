import argparse
import json
from jinja2 import Template

def str2bool(v):
    return v.lower() in ("true", "1", "yes")

def parse_list(v):
    # Accept a comma-separated list and convert to JSON list
    # e.g. "1,2,3" -> [1, 2, 3]
    # If it's already valid JSON like "[1,2,3]", parse that directly.
    try:
        return json.loads(v)
    except:
        return [float(x.strip()) for x in v.split(',') if x.strip()]

# Default configuration
defaults = {
    "model_type": "diffusion_cond",
    "sample_size": 352800,
    "sample_rate": 44100,
    "audio_channels": 2,
    "encoder_requires_grad": False,
    "encoder_in_channels": 2,
    "encoder_channels": 128,
    "encoder_c_mults": [1, 2, 4, 8, 16],
    "encoder_strides": [2, 4, 4, 8, 8],
    "encoder_latent_dim": 128,
    "encoder_use_snake": True,
    "decoder_out_channels": 2,
    "decoder_channels": 128,
    "decoder_c_mults": [1, 2, 4, 8, 16],
    "decoder_strides": [2, 4, 4, 8, 8],
    "decoder_latent_dim": 64,
    "decoder_use_snake": True,
    "decoder_final_tanh": False,
    "pre_latent_dim": 64,
    "pre_downsampling_ratio": 2048,
    "pre_io_channels": 2,
    "t5_model_name": "t5-base",
    "t5_max_length": 128,
    "seconds_start_min_val": 0,
    "seconds_start_max_val": 512,
    "seconds_total_min_val": 0,
    "seconds_total_max_val": 512,
    "cond_dim": 768,
    "diffusion_type": "dit",
    "diffusion_io_channels": 64,
    "diffusion_embed_dim": 1536,
    "diffusion_depth": 24,
    "diffusion_num_heads": 24,
    "diffusion_cond_token_dim": 768,
    "diffusion_global_cond_dim": 1536,
    "diffusion_project_cond_tokens": False,
    "diffusion_transformer_type": "continuous_transformer",
    "use_ema": True,
    "log_loss_info": False,
    "optimizer_type": "AdamW",
    "optimizer_lr": 5e-5,
    "optimizer_betas": [0.9, 0.999],
    "optimizer_weight_decay": 1e-3,
    "demo_every": 150,
    "demo_steps": 200,
    "num_demos": 5,
    "demo_cfg_scales": [6]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model config file.")

    # Add arguments for each parameter you'd like to override
    parser.add_argument("--model_type", type=str, help="Model type.")
    parser.add_argument("--sample_size", type=int, help="Sample size.")
    parser.add_argument("--sample_rate", type=int, help="Sample rate.")
    parser.add_argument("--audio_channels", type=int, help="Audio channels.")

    parser.add_argument("--encoder_requires_grad", type=str, help="Encoder requires_grad (true/false).")
    parser.add_argument("--encoder_in_channels", type=int, help="Encoder in_channels.")
    parser.add_argument("--encoder_channels", type=int, help="Encoder channels.")
    parser.add_argument("--encoder_c_mults", type=str, help='Encoder c_mults as JSON or comma-separated.')
    parser.add_argument("--encoder_strides", type=str, help='Encoder strides as JSON or comma-separated.')
    parser.add_argument("--encoder_latent_dim", type=int, help="Encoder latent_dim.")
    parser.add_argument("--encoder_use_snake", type=str, help="Encoder use_snake (true/false).")

    parser.add_argument("--decoder_out_channels", type=int, help="Decoder out_channels.")
    parser.add_argument("--decoder_channels", type=int, help="Decoder channels.")
    parser.add_argument("--decoder_c_mults", type=str, help='Decoder c_mults as JSON or comma-separated.')
    parser.add_argument("--decoder_strides", type=str, help='Decoder strides as JSON or comma-separated.')
    parser.add_argument("--decoder_latent_dim", type=int, help="Decoder latent_dim.")
    parser.add_argument("--decoder_use_snake", type=str, help="Decoder use_snake (true/false).")
    parser.add_argument("--decoder_final_tanh", type=str, help="Decoder final_tanh (true/false).")

    parser.add_argument("--pre_latent_dim", type=int, help="Pretransform latent_dim.")
    parser.add_argument("--pre_downsampling_ratio", type=int, help="Pretransform downsampling_ratio.")
    parser.add_argument("--pre_io_channels", type=int, help="Pretransform io_channels.")

    parser.add_argument("--t5_model_name", type=str, help="t5 model name.")
    parser.add_argument("--t5_max_length", type=int, help="t5 max_length.")

    parser.add_argument("--seconds_start_min_val", type=int, help="Seconds_start min_val.")
    parser.add_argument("--seconds_start_max_val", type=int, help="Seconds_start max_val.")
    parser.add_argument("--seconds_total_min_val", type=int, help="Seconds_total min_val.")
    parser.add_argument("--seconds_total_max_val", type=int, help="Seconds_total max_val.")
    parser.add_argument("--cond_dim", type=int, help="Conditioning cond_dim.")

    parser.add_argument("--diffusion_type", type=str, help="Diffusion type.")
    parser.add_argument("--diffusion_io_channels", type=int, help="Diffusion io_channels.")
    parser.add_argument("--diffusion_embed_dim", type=int, help="Diffusion embed_dim.")
    parser.add_argument("--diffusion_depth", type=int, help="Diffusion depth.")
    parser.add_argument("--diffusion_num_heads", type=int, help="Diffusion num_heads.")
    parser.add_argument("--diffusion_cond_token_dim", type=int, help="Diffusion cond_token_dim.")
    parser.add_argument("--diffusion_global_cond_dim", type=int, help="Diffusion global_cond_dim.")
    parser.add_argument("--diffusion_project_cond_tokens", type=str, help="Diffusion project_cond_tokens (true/false).")
    parser.add_argument("--diffusion_transformer_type", type=str, help="Diffusion transformer_type.")

    parser.add_argument("--use_ema", type=str, help="Use EMA (true/false).")
    parser.add_argument("--log_loss_info", type=str, help="Log loss info (true/false).")

    parser.add_argument("--optimizer_type", type=str, help="Optimizer type.")
    parser.add_argument("--optimizer_lr", type=float, help="Optimizer learning rate.")
    parser.add_argument("--optimizer_betas", type=str, help='Optimizer betas as JSON or comma-separated.')
    parser.add_argument("--optimizer_weight_decay", type=float, help="Optimizer weight_decay.")

    parser.add_argument("--demo_every", type=int, help="Demo every steps.")
    parser.add_argument("--demo_steps", type=int, help="Demo steps.")
    parser.add_argument("--num_demos", type=int, help="Number of demos.")
    parser.add_argument("--demo_cfg_scales", type=str, help='Demo cfg scales as JSON or comma-separated.')

    args = parser.parse_args()

    # Convert arguments if they are not None
    final_params = defaults.copy()
    for param, val in vars(args).items():
        if val is not None:
            if param in ["encoder_requires_grad", "encoder_use_snake", "decoder_use_snake", "decoder_final_tanh",
                         "diffusion_project_cond_tokens", "use_ema", "log_loss_info"]:
                # Convert to bool
                final_params[param] = str2bool(val)
            elif param in ["encoder_c_mults", "encoder_strides", "decoder_c_mults", "decoder_strides",
                           "optimizer_betas", "demo_cfg_scales"]:
                # Convert to lists
                final_params[param] = parse_list(val)
            else:
                final_params[param] = val

    # Load template
    with open("model_config_template.json.j2", 'r') as f:
        template_content = f.read()

    template = Template(template_content)
    rendered = template.render(**final_params)

    # Validate that the rendered string is valid JSON
    try:
        config = json.loads(rendered)
    except json.JSONDecodeError as e:
        print(f"Error: Generated config is not valid JSON: {e}")
        exit(1)

    # Write out the final config as JSON
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
