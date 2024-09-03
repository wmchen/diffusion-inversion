# Diffusion Inversion Methods
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repository is a collection of diffusion inversion methods, and also the official implementation of our inversion method (AAAI'25 under review).

**NOTE** that this repository is only build for inversion, applications to downstream tasks (e.g. image editing, rare concept generation, etc) are not included.


## Installation

We provide a conda environment that is fully tested. Run the following command to create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate diffusion-inversion
```

You can also install the dependencies manually using pip:

```bash
cd requirements
pip install -r torch.txt
pip install -r base.txt
```

Other versions of Python modules may work, but they are untested. We recommend using our conda environment to avoid any compatibility issues.


## Getting Started

Please refer to [examples.ipynb](./examples.ipynb) for a quick start.


## Supported Methods

- [x] (ICLR'21) DDIM Inversion: [paper](https://arxiv.org/abs/2010.02502)
- [x] Negative Prompt Inversion (NPI): [paper](http://arxiv.org/abs/2305.16807)
- [x] (CVPR'23) Null-Text Inversion (NTI): [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mokady_NULL-Text_Inversion_for_Editing_Real_Images_Using_Guided_Diffusion_Models_CVPR_2023_paper.html), [official implementation](https://github.com/google/prompt-to-prompt)
- [x] (CVPR'23) Exact Diffusion Inversion via Coupled Transformations (EDICT): [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wallace_EDICT_Exact_Diffusion_Inversion_via_Coupled_Transformations_CVPR_2023_paper.html), [official implementation](https://github.com/salesforce/EDICT)
- [x] Fixed-Point Inversion (FPI): [paper](https://arxiv.org/abs/2312.12540v1), [official implementation](https://github.com/dvirsamuel/FPI)
- [x] (ICCV'23) Accelerated Iterative Diffusion Inversion (AIDI): [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Effective_Real_Image_Editing_with_Accelerated_Iterative_Diffusion_Inversion_ICCV_2023_paper.html)
- [x] (ICCV'23) Prompt Tuning Inversion (PTI): [paper](https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Prompt_Tuning_Inversion_for_Text-driven_Image_Editing_Using_Diffusion_Models_ICCV_2023_paper.html)
- [x] (AAAI'24) Tuning-free Inversion-enhanced Control (TIC): [paper](https://ojs.aaai.org/index.php/AAAI/article/view/27931)
- [ ] (ECCV'24) ReNoise: [paper](https://arxiv.org/abs/2403.14602), [official implementation](https://github.com/garibida/ReNoise-Inversion) 
- [ ] (AAAI'25 under review) our method


## Citation

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


## Contributors

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


## License

This project is intended for research use only, licensed under the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).