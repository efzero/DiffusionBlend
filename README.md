# DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction

This repository contains the official implementation of **"DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction"**, published at **NeurIPS 2024**.

---

## Overview

DiffusionBlend introduces a novel method for 3D computed tomography (CT) reconstruction using position-aware diffusion score blending. By leveraging position-specific priors, the framework achieves enhanced reconstruction accuracy while maintaining computational efficiency.

---

## Features

- **Position-aware Diffusion Blending:** Incorporates spatial information to refine 3D reconstruction quality.
- **Triplane-based 3D Representation:** Utilizes a position encoding to model 3D patch priors efficiently.
- **Scalable and Generalizable:** Designed for both synthetic and real-world CT reconstruction tasks.

---

## Requirements

The code is implemented in Python and requires the following dependencies:

- `torch` (>=1.9.0)
- `torchvision`
- `numpy`

You can install the dependencies via:

```bash
pip install torch torchvision numpy
```

## Training
To train the model on synthetic volume CT data, use the following script:

```bash
bash train_SVCT_3D_triplane.sh
```

## Inference
To perform inference and evaluate 3D reconstruction using diffusion score blending, use:

```bash
bash eval_3D_blend_cond.sh
```


## Citation
If you find this work useful in your research, please cite:


@inproceedings{diffusionblend2024,
  title={DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction},
  author={Song, Bowen and Hu, Jason and Luo, Zhaoxu and Fessler, Jeffrey A and Shen, Liyue},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

## Acknowledgements
We thank the contributors and the NeurIPS community for their valuable feedback and discussions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


