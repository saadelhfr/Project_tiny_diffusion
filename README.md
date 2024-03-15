This repo contains python code for the MAP583 project Tiny Diffusion. The repo is based on [Project Tiny Diffusion](https://github.com/dataflowr/Project-tiny-diffusion). In the referenced repo the authors use fourrier featur networks as described in [1](https://arxiv.org/abs/2006.10739) alongside Denoising Diffusion Probabilistic Models as described here [2](https://arxiv.org/abs/2006.11239) to generate simple probability distributions in $\mathcal{R}^{2}$ space. The approach is very straight forward : 
- The dataset is constructed by sampling points from a known probability distribution. 
- We add a noise to the sampled points through the classic DDPM approach. 

$$
\text{fourier}_{\text{scaled}}(x_i) = [\sin(w_0 x_i \cdot \text{scale}), \cos(w_0 x_i \cdot \text{scale}), \sin(w_1 x_i \cdot \text{scale}), \cos(w_1 x_i \cdot \text{scale}), \ldots, \sin(w_{\frac{n}{2}-1} x_i \cdot \text{scale}), \cos(w_{\frac{n}{2}-1} x_i \cdot \text{scale})]
$$

- The projected points are used to train a neural network to learn the previously added noise .
- The Simple objective (L2 distance between predicted and added noise) is the function used to optimise the neural network.


The goal of our project is to explore ways to improve on this approach with the goal to be able to generate MNIST like scatter plots. 

It is very simple to generate individual MNIST digits ( or rather specific images from the MNIST dataset) by simply considering the image as a probability distribution and sampling from it this is our first approach which yield the following results (example with an image of zero) : 

![MNIST_0]("readme_assets/0_Readme.png")

As we can see the resulting scatter plot stays the same for different sampling processes from the trained model. While these results appear satifying, the model is unable to sample from two different distribution upon which it was trained, for example if the model is provided with data points from the MNIST 0 and MNIST 1 it generates a superposed density where the One and the zero are both represented. We show an example with the moons and dino distributions as they offer a better visual representation of this phenomenon. 
![Dino and Moon]("readme_assets/shuffled_dino_moons.png")

How can we solve this probem ? i.e. how can we make the model capable of generating from different probability distributions without the need to explicitly define them.  



Multiple possible approaches are possible and this is a curated list of our most promising approaches : 

- [Sequence to Sequence](#sequence-to-sequence-model)
- [Attention Version 1](#version-one-of-attention-implementation)
- [Attention Version 2](#version-two-of-attention-implementation) 


### Sequence to Sequence Model

The sequence to sequence model that we used is a simple implementation of the model described in [3](https://arxiv.org/abs/1409.3215) . The model is trained to predict the noise added to the input data. However the model was not able to generate any of the distribution it was trained on furthermore the loss was stagnat throughout the training period.  


### Version One of Attention Implementation

The first version of the attention implementation is a standard scaled dot product attention. The goal of adding this module is to allow for conditionning the input vector in an order agnostic manner. The basic idea is that the modul has a History vector that can be updated or reset that is used to calculate a context vector based on the state of the $\mathcal{R}^{2}$ upon which we operate. 

The model is ought to be able to denoise new points conditionned on the state of the space, This is analogue to [4](https://arxiv.org/pdf/2207.12598.pdf) where the diffusion model is trained on conditionning vector. 

while this approach yields good results it is not different than the original model as the attention is not able to capture the state of the space.



### Version Two of Attention Implementation

In the second version we keep the same idea as the first one but we change the way we calculate the context vector. Intsead of basing the calculation solely on the state of the space we use the state of the space and the state of the input vector. To perhaps allow the model to generate different distributions by "understanding" the difference between the new noised vector and the space generated before. 

While this method was more stable than the first one we still get superposed distributions just as with the base model presented in [Project Tiny Diffusion](https://github.com/dataflowr/Project-tiny-diffusion)

