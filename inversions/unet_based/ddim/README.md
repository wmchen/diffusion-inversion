# DDIM Inversion

[ICLR 2021] **Denoising Diffusion Implicit Models**

> Jiaming Song, Chenlin Meng, Stefano Ermon

> Stanford University

Denoising process:

$$
z_{t-1} = \sqrt{\frac{\alpha_{t-1}}{\alpha_{t}}} z_{t} + \left ( \sqrt{1 - \alpha_{t-1}} - \sqrt{\frac{(1-\alpha_{t})\alpha_{t-1}}{\alpha_{t}}} \right ) \epsilon_{\theta}(z_{t}, t)
$$

Inversion process:

$$
z_{t} = \sqrt{\frac{\alpha_{t}}{\alpha_{t-1}}} z_{t-1} + \left ( \sqrt{1 - \alpha_{t}} - \sqrt{\frac{(1-\alpha_{t-1})\alpha_{t}}{\alpha_{t-1}}} \right ) \epsilon_{\theta}(z_{t-1}, t-1)
$$
