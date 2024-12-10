# DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction

This repository contains the official implementation of **"DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction"**, published at **NeurIPS 2024**.

Paper link: https://openreview.net/forum?id=h3Kv6sdTWO&referrer=%5Bthe%20profile%20of%20Bowen%20Song%5D(%2Fprofile%3Fid%3D~Bowen_Song3)


---

## Overview

DiffusionBlend introduces a novel method for 3D computed tomography (CT) reconstruction using position-aware diffusion score blending. By leveraging position-specific priors, the framework achieves enhanced reconstruction accuracy while maintaining computational efficiency.
<img width="919" alt="image" src="https://github.com/user-attachments/assets/3dab4e88-6676-4673-b27a-da6d2e4b8518">



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

## Results
<img width="890" alt="image" src="https://github.com/user-attachments/assets/c9abd6c3-4723-4245-81d1-5d512bfb0c06">

<img width="893" alt="image" src="https://github.com/user-attachments/assets/9abd3ad7-d8cb-4e47-95ce-d955b47405df">

<img width="929" alt="image" src="https://github.com/user-attachments/assets/56110385-4540-49da-88b4-d6145f15bf2c">







## Citation
If you find this work useful in your research, please cite:

```
@inproceedings{diffusionblend2024,
  title={DiffusionBlend: Learning 3D Image Prior through Position-aware Diffusion Score Blending for 3D Computed Tomography Reconstruction},
  author={Song, Bowen and Hu, Jason and Luo, Zhaoxu and Fessler, Jeffrey A and Shen, Liyue},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## Acknowledgements
We thank the contributors and the NeurIPS community for their valuable feedback and discussions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


