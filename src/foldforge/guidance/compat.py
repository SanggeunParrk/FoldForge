"""Checkpoint-facing argument adapter; all guidance math is shared."""

from .engine import GuidanceEngine


class TFGEngine(GuidanceEngine):
    def step(
        self,
        denoise_net,
        *,
        x,
        t_hat,
        c_tau,
        step_scale_eta,
        step_i,
        num_diffusion_steps,
        input_feature_dict,
        torch_generator=None,
        **model_context,
    ):

        def denoise(coords, sigma):
            return denoise_net(
                x_noisy=coords,
                t_hat_noise_level=sigma,
                input_feature_dict=input_feature_dict,
                **model_context,
            )

        return super().step(
            denoise,
            x=x,
            t_hat=t_hat,
            c_tau=c_tau,
            step_scale_eta=step_scale_eta,
            step_i=step_i,
            num_diffusion_steps=num_diffusion_steps,
            input_feature_dict=input_feature_dict,
            torch_generator=torch_generator,
        )
