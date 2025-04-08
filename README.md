# Diffusion Inversion Methods
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repository is a collection of diffusion inversion methods.

**NOTE** that this repository is only build for inversion, applications to downstream tasks (e.g. image editing, rare concept generation, etc) are not included.


## 1. Installation

Create and activate a `conda` environment:

```bash
conda create -n diffusion-inversion python=3.10

conda activate diffusion-inversion
```

Install PyTorch:

```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other packages:

```bash
pip install -r requirements.txt
```


## 2. Getting Started

Please refer to [examples/](./examples) for a quick start.


## 3. Supported Methods

### 3.1 UNet-based Diffusion Models

| Method | Publication | Paper | Official repo. | Ours imp. |
| ------ | ----------- | ----- | -------------- | --------- |
| DDIM Inversion | ICLR 2021 | [paper](https://arxiv.org/abs/2010.02502) |  | [ours](./inversions/unet_based/ddim) |
| Negative Prompt Inversion (NPI) | ArXiv 2023 | [paper](http://arxiv.org/abs/2305.16807) |  | [ours](./inversions/unet_based/npi) |
| Null-Text Inversion (NTI) | CVPR 2023 | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.html) | [official](https://github.com/google/prompt-to-prompt) | [ours](./inversions/unet_based/nti) |
| Exact Diffusion Inversion via Coupled Transformations (EDICT) | CVPR 2023 | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wallace_EDICT_Exact_Diffusion_Inversion_via_Coupled_Transformations_CVPR_2023_paper.html) | [official](https://github.com/salesforce/EDICT) | [ours](./inversions/unet_based/edict) |
| Accelerated Iterative Diffusion Inversion (AIDI) | ICCV 2023 Oral | [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.html) |  | [ours](./inversions/unet_based/aidi) |
| Prompt Tuning Inversion (PTI) | ICCV 2023 | [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Prompt_Tuning_Inversion_for_Text-driven_Image_Editing_Using_Diffusion_Models_ICCV_2023_paper.html) |  | [ours](./inversions/unet_based/pti) |
| Real-world Image Variation by ALignment (RIVAL) | NeurIPS 2023 Spotlight | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/61960fdfda4d4e95fa1c1f6e64bfe8bc-Abstract-Conference.html) | [official](https://github.com/dvlab-research/RIVAL) | [ours](./inversions/unet_based/rival) |
| Fixed-Point Inversion (FPI) | ArXiv 2023 | [paper](https://arxiv.org/abs/2312.12540v1) | [official](https://github.com/dvirsamuel/FPI) | [ours](./inversions/unet_based/fpi) |
| On Exact Inversion of DPM-solvers | CVPR 2024 | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.html) | [official](https://github.com/smhongok/inv-dpm) | [ours](./inversions/unet_based/inv_dpm) |
| Tuning-free Inversion-enhanced Control (TIC) | AAAI 2024 | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/27931) |  | [ours](./inversions/unet_based/tic) |
| Bi-Directional Integration Approximation (BDIA) | ECCV 2024 Oral | [paper](https://arxiv.org/abs/2307.10829) | [official](https://github.com/guoqiang-zhang-x/BDIA) | [ours](./inversions/unet_based/bdia) |
| ReNoise | ECCV 2024 | [paper](https://arxiv.org/abs/2403.14602) | [official](https://github.com/garibida/ReNoise-Inversion) | [ours](./inversions/unet_based/renoise) |
| Bidirectional Explicit Linear Multi-step (BELM) | NeurlPS 2024 | [paper](https://arxiv.org/abs/2410.07273) | [official](https://github.com/zituitui/BELM) | [ours](./inversions/unet_based/belm) |

All methods above support Stable Diffusion v1, v2 and SDXL in our implementations. The following versions of diffusion model are fully tested:
- [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [stable-diffusion-v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### 3.2 DiT-based Diffusion Models

| Method | Publication | Paper | Official repo. | Ours imp. |
| ------ | ----------- | ----- | -------------- | --------- |
| RF-Inversion | ICLR 2025 | [paper](https://arxiv.org/abs/2410.10792) | [official](https://github.com/LituRout/RF-Inversion) | [link](./inversions/dit_based/rf_inversion) |
| RF-Solver | ArXiv 2024 | [paper](https://arxiv.org/abs/2411.04746) | [official](https://github.com/wangjiangshan0725/RF-Solver-Edit) | [ours](./inversions/dit_based/rf_solver) |
| FireFlow | ArXiv 2024 | [paper](https://arxiv.org/abs/2412.07517) | [official](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing?tab=readme-ov-file) | TODO |

CLAIM: The implementation of RF-Inversion is copy from [diffusers](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_flux_rf_inversion.py).

## 4. Citation

If you use this codebase in your research, please cite our repository and the corresponding papers.

```bibtex
@misc{chen2024inversion,
  title = {Diffusion Inversion Methods},
  author = {Chen, Weiming},
  howpublished = {https://github.com/wmchen/diffusion-inversion},
  year = {2024},
}
```

<!-- If you find LBI useful for your research and applications, please cite our paper: -->


## 5. Contributors

We appreciate all the contributors who add new features or fix bugs, as well as the users who offer valuable feedback. We welcome all contributors, feel free to create an issue or file a pull request and join us! ❤️

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://weimingchen.net/"><img src="https://avatars.githubusercontent.com/u/33000375?v=4?s=100" width="100px;" alt="Weiming Chen"/><br /><sub><b>Weiming Chen</b></sub></a><br /><a href="#ideas-wmchen" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Weiming Chen/diffusion-inversion/commits?author=wmchen" title="Code">💻</a> <a href="#projectManagement-wmchen" title="Project Management">📆</a> <a href="#research-wmchen" title="Research">🔬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/liuqifan67"><img src="https://avatars.githubusercontent.com/u/54019906?v=4?s=100" width="100px;" alt="liuqifan67"/><br /><sub><b>liuqifan67</b></sub></a><br /><a href="#ideas-liuqifan67" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-liuqifan67" title="Research">🔬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/llsysysy"><img src="https://avatars.githubusercontent.com/u/100456149?v=4?s=100" width="100px;" alt="llsysysy"/><br /><sub><b>llsysysy</b></sub></a><br /><a href="#ideas-llsysysy" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-llsysysy" title="Research">🔬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://yushuntang.github.io"><img src="https://avatars.githubusercontent.com/u/75136524?v=4?s=100" width="100px;" alt="TANG"/><br /><sub><b>TANG</b></sub></a><br /><a href="#ideas-yushuntang" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-yushuntang" title="Research">🔬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://nkdailab.github.io/"><img src="https://avatars.githubusercontent.com/u/152594959?v=4?s=100" width="100px;" alt="nkdailab"/><br /><sub><b>nkdailab</b></sub></a><br /><a href="#financial-NKDAILab" title="Financial">💵</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="5">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->


## 6. License

This project is intended for research use only, licensed under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).