<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL-3.0 License][license-shield]][license-url]


<br />
<div align="center">
  <a href="https://github.com/project-kxkg/ViDove">
    <img src="images/logo_draft.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">🐦ViDove: End-to-end Video Translation Toolkit</h3>

  <p align="center">
    Transcribe and Translate Your Video with a Single Click
    <br />
    <a href="https://pigeonai.club/"><strong>Offical Website »</strong></a>
    <br />
    <br />
    <a href="https://huggingface.co/spaces/StarPigeon/ViDove">Try Demo</a>
    ·
    <a href="https://github.com/project-kxkg/ViDove/issues">Report Bug</a>
    ·
    <a href="https://github.com/project-kxkg/ViDove/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#release">Release</a></li>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Release
- [12/20]🔥**ViDove V0.1 Released**: We are happy to release our initial version of ViDove: End-to-end Video Translation Toolkit. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

Introducing **ViDove**, a pioneering video automated machine translation toolkit, meticulously crafted for professional domains. Developed by Pigeon.AI, ViDove promises rapid, precise, and relatable translations, revolutionizing the workflow of subtitle groups and translation professionals. It's an open-source tool, offering unparalleled flexibility, transparency, and security, alongside scalable architecture for customization. Featuring domain adaptation, ViDove effortlessly adjusts to various professional fields, and its end-to-end pipeline turns video links into captioned content with a single click. ViDove is not just a translation tool; it's a bridge connecting content across languages, making video translation more accessible, efficient, and accurate than ever.


Here's why:
* **End-to-End Pipeline** (from video link to captioned video): 
  - **One-Click Deployment:** Users can deploy the tool with just one click.
  - **Video Link to Translated Video:** Simply input a video link to generate a translated video with ease.
* **Domain Adaptation:**
  - Our pipeline is adaptable to various professional fields (e.g., StarCraft II). Users can easily upload customized dictionaries and fine-tune models based on specific data corpora.
* **Open Source:**
  - Our toolkit is entirely open source, and we warmly welcome and look forward to the participation of the broader developer community in the ongoing development of the toolkit.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Main Contributors
[![](https://github.com/yichen14.png?size=50)](https://github.com/yichen14)
[![](https://github.com/TheAnsIs42.png?size=50)](https://github.com/TheAnsIs42)
[![](https://github.com/JiaenLiu.png?size=50)](https://github.com/JiaenLiu)
[![](https://github.com/worldqwq.png?size=50)](https://github.com/worldqwq)
[![](https://github.com/Yuhan-Lu.png?size=50)](https://github.com/Yuhan-Lu)
[![](https://github.com/CanYing0913.png?size=50)](https://github.com/CanYing0913)
[![](https://github.com/willbe03.png?size=50)](https://github.com/willbe03)
[![](https://github.com/pinqian77.png?size=50)](https://github.com/pinqian77)

[**Web Dev: Tingyu Su**](https://www.sutingyu.com/)



<!-- GETTING STARTED -->
## Getting Started

**We recommend you use UNIX like operating systems(MacOS/Linux Family) for local installation.**

### Installation

1. Get a OpenAI API Key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Clone the repo
   ```sh
   git clone https://github.com/project-kxkg/ViDove.git
   cd ViDove
   ```
3. Install Requirments
   ```sh
   conda create -n ViDove python=3.10 -y
   conda activate ViDove
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Enter your API in bash
   
   UNIX Like:

   ```sh
   export OPENAI_API_KEY="your_api_key" 
   ```

   Windows:
   ```sh
   set OPENAI_API_KEY="your_api_key" 
   ```
5. Install FFmpeg:

   Download FFmpeg [here](https://ffmpeg.org/)

   For more specfic guide on FFmpeg installation on different platforms: [Click Here](doc/ffmpeg_guide_en.md) | [点击此处](doc/ffmpeg_guide_zh.md)

   ~~We recommand you use [Chocolatey Package Manager](https://chocolatey.org/) to install ffmpeg~~


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Quick Start with Gradio User Interface
   ```sh
   python3 entries/app.py
   ```

   #### Updated 2024/3/3: we are happy to tell you that you could try to use RAG boosted translation by selecting is_assistant in the UI under the translation section.

### Launch with configs
  - Start with Youtube Link input:
    ```sh
    python3 entries/run.py --link "your_youtube_link"
    ```
  - Start with Video input:
    ```sh
    python3 entries/run.py --video_file path/to/video_file
    ```
  - Start with Audio input:
    ```sh
    python3 entries/run.py --audio_file path/to/audio_file
    ```
  - Terminal Usage:
    ```sh
    usage: run.py [-h] [--link LINK] [--video_file VIDEO_FILE] [--audio_file AUDIO_FILE] [--launch_cfg LAUNCH_CFG] [--task_cfg TASK_CFG]

    options:
      -h, --help            show this help message and exit
      --link LINK           youtube video link here
      --video_file VIDEO_FILE
                            local video path here
      --audio_file AUDIO_FILE
                            local audio path here
      --launch_cfg LAUNCH_CFG
                            launch config path
      --task_cfg TASK_CFG   task config path
    ```

### Configs
  Use "--launch_cfg" and "--task_cfg" in run.py to change launch or task configuration
  - configs/local_launch.yaml 
    ```yaml
    # launch config for local environment
    local_dump: ./local_dump # change local dump dir here
    environ: local
    ```
  - configs/task_config.yaml
    
    copy and change this config for different configuration
    ```yaml
    # configuration for each task
    source_lang: EN
    target_lang: ZH
    field: General

    # ASR config
    ASR:
      ASR_model: whisper
      whisper_config:
        whisper_model: tiny
        method: stable
      
    # pre-process module config
    pre_process: 
      sentence_form: True
      spell_check: False
      term_correct: True

    # Translation module config
    translation:
      model: gpt-4
      chunk_size: 1000

    # post-process module config
    post_process: 
      check_len_and_split: True
      remove_trans_punctuation: True

    # output type that user receive
    output_type: 
      subtitle: srt
      video: True
      bilingual: True
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
<!-- ## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/project-kxkg/ViDove/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the  GPL-3.0 license. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Developed by **Pigeon.AI**🐦 from Star Pigeon Fan-sub Group.

See Our [Bilibili Account](https://space.bilibili.com/195670539)

Official Email: gggzmz@163.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)
* 

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/project-kxkg/ViDove.svg?style=for-the-badge
[contributors-url]: https://github.com/project-kxkg/ViDove/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/project-kxkg/ViDove.svg?style=for-the-badge
[forks-url]: https://github.com/project-kxkg/ViDove/forks
[stars-shield]: https://img.shields.io/github/stars/project-kxkg/ViDove.svg?style=for-the-badge
[stars-url]: https://github.com/project-kxkg/ViDove/stargazers
[issues-shield]: https://img.shields.io/github/issues/project-kxkg/ViDove.svg?style=for-the-badge
[issues-url]: https://github.com/project-kxkg/ViDove/issues
[license-shield]: https://img.shields.io/github/license/project-kxkg/ViDove.svg?style=for-the-badge
[license-url]: https://github.com/project-kxkg/ViDove/blob/main/LICENSE
