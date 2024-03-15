# Tiny diffusion

## Project Overview

Tiny Diffusion is an innovative project that explores the functionality of diffusion models. The mathematical foundations of the project highlight the advantages of a step-by-step approach to explicit probability distribution modelling. Despite the sequential nature of the operation, which can be time consuming and less adept at detecting patterns based on partial normal distributions, the project has demonstrated significant potential on various datasets.

## Mathematical foundations

The core of Tiny Diffusion lies in the forward and backward processes of diffusion models, which iteratively learn the distribution of the data through a series of transformations. This project not only addresses the inherent mathematical challenges, but also proposes innovative solutions to overcome them.

## Experiments with new datasets

We tested the model on three additional datasets, including a 3D dataset, to assess its versatility and robustness. A particular challenge was the model's failure to detect lines, which was ingeniously solved by densifying the data points with uniform and Poisson distributions.

## Diffusion on the MNIST dataset

The project took a unique approach to learning the underlying distribution of the images in the MNIST dataset, comparing it to 'points of a dinosaur' to visualise the concept. The original code was adapted from a 2D to a 28x28 dimension to accommodate the MNIST dataset, using a simple CNN structure including Conv2D, BatchNorm2D, ReLU, ConvTranspose2D and additional Conv2D layers.

## Implementation and Improvements

We implemented time-step in a number of ways to improve model performance, including comparison to the initial image, comparison to the previous image, and introduction of noise. These implementations were crucial in improving the model's noise removal capabilities without the need to generate new images from randomly noisy inputs.

## Generating numbers from prompts

One of the breakthroughs of the project was the generation of numbers from prompts. By conditioning the distribution with labels and changing the learning approach from unsupervised to semi-supervised, the model showed significant improvements. Various hyperparameters were tested to fine-tune the model, with the final settings set to 25 epochs, a learning rate of 2e-4, a hidden size of 128, and five hidden layers.

## Conclusion

Tiny Diffusion has opened up new ways of understanding and harnessing the power of diffusion models. The findings and methods developed in this project could be instrumental for further research and applications in various fields of machine learning and artificial intelligence.

## Team

- Céleste GALLIEN
- Samson GOUREVITCH
- Laurine MEIER
- Ines YAICI

Date of Presentation: 15 March 2024
