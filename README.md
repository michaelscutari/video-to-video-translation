# ECE661 GAN Project: Advanced Image-to-Image Translation

A comprehensive implementation of state-of-the-art Generative Adversarial Networks for image-to-image translation tasks, featuring multiple architectures and novel extensions for temporal consistency.

## Overview

This project implements and extends several prominent GAN architectures for image-to-image translation, demonstrating proficiency in deep learning research and engineering best practices. The implementation includes CycleGAN for unpaired translation, Pix2Pix for paired translation, and a novel RecycleGAN extension that incorporates temporal consistency for video-based applications.

## Architectures Implemented

### CycleGAN
- **Purpose**: Unpaired image-to-image translation
- **Key Features**: Cycle consistency loss, identity preservation, self-attention mechanisms
- **Applications**: Style transfer (Monet ↔ Photography), domain adaptation (Horse ↔ Zebra)

### Pix2Pix
- **Purpose**: Paired image-to-image translation with conditional GANs
- **Key Features**: U-Net generator with skip connections, PatchGAN discriminator
- **Applications**: Satellite ↔ Map generation, semantic segmentation visualization

### RecycleGAN (Novel Extension)
- **Purpose**: Temporally consistent video-to-video translation
- **Key Features**: Recurrence loss for temporal consistency, future frame prediction
- **Innovation**: Extends cycle consistency to temporal domain for video applications

## Technical Implementation

### Model Architecture Enhancements
- **Self-Attention Integration**: Global self-attention mechanisms in generator bottlenecks for improved feature correlation
- **Advanced U-Net Design**: Deep encoder-decoder architecture with instance normalization and skip connections
- **Residual Learning**: Incorporated residual blocks for stable training and improved gradient flow

### Training Infrastructure
- **Distributed Computing**: SLURM job scheduling for high-performance cluster training
- **Experiment Tracking**: Weights & Biases integration for comprehensive metric monitoring
- **Gradient Stabilization**: Gradient clipping and learning rate scheduling for stable convergence
- **Replay Buffers**: Historical sample storage to improve discriminator training stability

### Software Engineering Practices
- **Modular Design**: Separate configuration, dataset, model, and training modules
- **Reproducible Research**: Comprehensive logging, checkpointing, and configuration management
- **Data Pipeline Optimization**: Efficient data loading with parallel processing and caching

## Key Features

**Advanced Loss Functions**: Implementation of cycle consistency, identity, adversarial, and novel recurrence losses with configurable weighting schemes.

**Robust Training Pipeline**: Comprehensive training loop with learning rate scheduling, gradient clipping, and automatic mixed precision support.

**Extensive Evaluation**: Real-time monitoring of training metrics, sample generation tracking, and automated checkpoint management.

**Flexible Configuration**: Modular configuration system enabling rapid experimentation with different architectures and hyperparameters.

## Results and Applications

The implementation successfully demonstrates high-quality image translation across multiple domains, with particular strength in maintaining structural consistency while achieving convincing style transfer. The RecycleGAN extension shows promising results for video-based applications where temporal consistency is crucial.

## Technical Skills Demonstrated

- **Deep Learning Frameworks**: PyTorch implementation with advanced features including custom loss functions and complex training loops
- **Research Implementation**: Translation of academic papers into production-quality code with proper software engineering practices
- **High-Performance Computing**: Efficient utilization of GPU clusters with proper job scheduling and resource management
- **Experiment Management**: Professional-grade experiment tracking and reproducible research practices

---

**Contributors**: Zach Charlick, Michael Scutari
**Course**: ECE661 - Computer Vision  
**Institution**: Duke University
