---
# Basic info
title: "PROJECT: Galaxy Zooing my first CNN"
date: 2025-05-28T08:15:04-03:00
draft: false
description: "VGG architectures, galaxy morphology classification, cross-validation, and way too much time spent on data augmentation"
tags: [ "deep-learning", "computer-vision", "pytorch", "cnn", "astronomy", "galaxy-zoo", "project", "vgg", "kaggle"]
author: "Me" # For multiple authors, use: ["Me", "You"]

# Metadata & SEO
canonicalURL: "https://canonical.url/to/page"
hidemeta: false
searchHidden: true

# Table of contents
showToc: true
TocOpen: false
UseHugoToc: true

# Post features
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
comments: false

# Syntax highlighting
disableHLJS: false # set true to disable Highlight.js
disableShare: false

# Edit link
editPost:
  URL: "https://github.com/CrustularumAmator/blog/tree/main/content"
  Text: "Suggest Changes"   # edit link text
  appendFilePath: true      # append file path to the edit link
--------------------

{{< figure src="/images/glaxy_zoo_clasifier.jpg" attr="Project poster for Applied Astrophysics Symposium at University of Washington" target="_blank" >}}

I learned that building your first serious CNN is less about the architecture and more about understanding why galaxies don't have a canonical "up" direction, and why that makes data augmentation both critical and surprisingly fun.

## First Time Building Something That Actually Works

So this was my first real dive into CNNs beyond MNIST and cat/dog classification, and honestly? I picked probably the most complicated dataset I could find. Galaxy Zoo isn't your typical ImageNet problem - instead of "this is a cat," you're trying to predict 37 different probability values that represent how humans answered morphological questions about each galaxy. Because apparently nothing in astronomy can ever be simple.

The whole thing started when I realized that most computer vision projects I'd done were basically following tutorials step-by-step. I wanted to build something from scratch that actually required thinking about the problem domain, not just plugging in a pre-trained ResNet and calling it a day.

## The Dataset: 61,578 Ways to Overthink Image Classification

The Galaxy Zoo dataset gives you 61,578 galaxy images (424x424 RGB) for training, each labeled with a 37-dimensional probability vector. These aren't one-hot encoded classes - they're soft targets derived from aggregating multiple human classifications. So instead of "this galaxy is spiral," you get something like "73% of people said it has spiral arms, 12% said it has a prominent bulge, 45% said it has a bar..." and so on for 37 different morphological features.

The test set has 79,975 unlabeled images where you need to predict these probability distributions. The evaluation metric is RMSE across all 37 dimensions, which immediately made me realize this was going to be more like a regression problem than standard classification.

They also provided some baselines: naive all-ones/all-zeros predictions and a "central pixel benchmark" that uses k-means clustering on the center pixel to assign labels (RMSE: 0.16194). The central pixel thing actually kind of works because galaxy color correlates with morphology, which is both clever and slightly depressing.

## Stealing Ideas from the Winners (Academic Integrity Approved)

The winning solution by Sander Dieleman achieved 0.07941 RMSE, and his approach became my roadmap. He used several techniques that I shamelessly borrowed:

**Rotational Symmetry Augmentation**: Galaxies have no canonical "up" direction, so you can rotate them by any angle. I implemented full 360-degree random rotations during training, plus horizontal and vertical flips. This effectively gave me infinite training data variations, which felt like cheating but was totally legal.

**Smart Cropping with Buffer**: Instead of just random cropping, I resized images to 160x160 then randomly cropped to 128x128. This buffer zone meant I could crop without losing important galaxy features at the edges. Morphological details matter in astronomy, so you can't just YOLO crop like you might with natural images.

**Controlled Color Jittering**: This was tricky because galaxy colors have physical meaning - you can't just randomly shift hues like you would for cats and dogs. I kept brightness, contrast, and saturation changes minimal (0.1 max) and hue shifts tiny (0.05) to preserve the astrophysical information while still adding some robustness.

## The Loss Function Nightmare (And Why I Gave Up on Multi-Head)

Here's where things got academically interesting and practically frustrating. The Galaxy Zoo decision tree structure suggests you should use a multi-head architecture - different output heads for different question branches, with custom loss functions that respect the logical dependencies between questions.

But after spending way too many hours trying to implement this properly, I realized the computational complexity was getting ridiculous, and honestly, my linear algebra wasn't quite up to designing the constraint matrices. So I took the pragmatic route: treat it as a simple 37-dimensional regression problem with MSE loss and sigmoid outputs.

This definitely wasn't optimal - you end up with predictions that might not respect the logical structure (like predicting high probability for both "smooth galaxy" and "has spiral arms"). But it was tractable, and sometimes tractable beats theoretically perfect when you're learning.

## Platform Hopping: A Comedy of GPU Allocation

Training was conducted across multiple platforms because, as a student, you take whatever compute you can get. Started development on my M1 MacBook (surprisingly decent for prototyping), moved to Google Colab's T4 GPU for initial training runs, then got access to the HYAK cluster for hyperparameter sweeps, and finally used Colab Pro's L4 GPU for the final training runs.

The modeling pipeline followed an iterative refinement process that probably looked more systematic than it actually was: VGG11 (too shallow), VGG16 (better but still underfitting), ResNet50 (overly complex for this dataset size), and ultimately a custom 7-block VGG-inspired CNN that I convinced myself was "tailored for the task" but was probably just the result of too much hyperparameter tuning.

Each platform switch required refactoring the data loading code, which taught me way more about file I/O and path management than I expected. Nothing like debugging CUDA out-of-memory errors at 2 AM to really understand batch size tradeoffs.

## Implementation Details (The Stuff That Actually Mattered)

My final architecture was a VGG-style CNN with 5 convolutional blocks (3→64→128→256→512→512 channels) followed by a custom fully connected tail with dropout regularization. The key insight was using adaptive average pooling before the classifier, which made the network more robust to input size variations and reduced the parameter count significantly.

I implemented mixed precision training with PyTorch's autocast, which gave me about 30% speedup and let me use larger batch sizes. Cross-validation with 5 folds helped validate that the model was actually learning generalizable features rather than just memorizing training quirks.

The data loading pipeline included per-batch memory cleanup (probably overkill but CUDA OOM errors are traumatic), and I spent an embarrassing amount of time optimizing the transforms pipeline to avoid CPU bottlenecks.

## Results: Better Than Baselines, Learning More Important Than Leaderboards

Final validation RMSE: 0.1861 ± 0.0056 across cross-validation folds. This beat the central pixel benchmark (0.16194) but was still far from the winning solution (0.07941). Still, for a first serious CNN project, I was pretty happy with the engineering aspects even if the performance wasn't record-breaking.

The model showed reasonable behavior - conservative confidence levels, sensible class distributions, and stable training curves. The predictions looked astronomically plausible, which felt like a small victory given how easy it would be to produce complete nonsense.

An iterative deep learning approach using progressively more complex architectures allowed for improved galaxy morphology predictions. Leveraging varied compute resources and continuous model tuning was critical in building a robust classification pipeline, even if "surpassing the record RMSE" remains an aspirational goal rather than an achieved one.

## What I Actually Learned

Beyond the technical skills (PyTorch proficiency, CNN architectures, hyperparameter tuning), this project taught me that domain knowledge matters way more than I expected. Understanding why galaxies look the way they do informed every design decision, from augmentation strategies to loss function choices.

I also learned that sometimes the theoretically optimal approach isn't practical for a student project timeline. The multi-head architecture would have been more elegant, but the simple regression approach actually worked and let me focus on other important aspects like proper evaluation and reproducible training.

Most importantly: when your validation loss curves look good and your model predicts reasonable probability distributions, that's already a success for a learning project, even if you're not beating Kaggle leaderboards.

---

*Full code available on [GitHub Gist](https://gist.github.com/CrustularumAmator/900505afe938f2af9ae31af57ee20d52), and way too many comments explaining design decisions. Because if you're going to overthink a project, you might as well document the overthinking process.*

*But also a preview is visible here*
{{< gist CrustularumAmator 900505afe938f2af9ae31af57ee20d52 >}}