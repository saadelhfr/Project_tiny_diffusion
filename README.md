This repo contains python code for the MAP583 project Tiny Diffusion. The repo is based on [Project Tiny Diffusion](https://github.com/dataflowr/Project-tiny-diffusion). In the referenced repo the authors use fourrier featur networks as described in [1](https://arxiv.org/abs/2006.10739) alongside Denoising Diffusion Probabilistic Models as described here [2](https://arxiv.org/abs/2006.11239) to generate simple probability distributions in $\mathcal{R}^{2}$ space. The approach is very straight forward : 
- First : The dataset is constructed by sampling points from a known probability distribution. 
- The sampled points are projected into a higher dimentional space through a scaled fourrier feature calculation as follows : 

$$


\text{fourier}_{\text{scaled}}(x_i) = [\sin(w_0 x_i \cdot \text{scale}), \cos(w_0 x_i \cdot \text{scale}), \sin(w_1 x_i \cdot \text{scale}), \cos(w_1 x_i \cdot \text{scale}), \ldots, \sin(w_{\frac{n}{2}-1} x_i \cdot \text{scale}), \cos(w_{\frac{n}{2}-1} x_i \cdot \text{scale})]

$$


