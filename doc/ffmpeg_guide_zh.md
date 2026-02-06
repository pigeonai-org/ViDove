# FFMPEG安装指南

本过程包括下载FFmpeg二进制文件并将它们添加到系统的PATH中，以便您可以从命令提示符或PowerShell运行FFmpeg。

<br>

## WINDOWS用户： 🪟

### 第一步: 下载FFMPEG
注意：我们建议仅通过 **官方源** 下载FFMPEG： [**FFMPEG官网**](https://ffmpeg.org/download.html#build-windows), <br>

或访问 [**BtbN GitHub页面**](https://github.com/BtbN/FFmpeg-Builds/releases) 下载最新的稳定版本

### 第二步: 解压FFMPEG文件
在您偏好的目录解压.7z包。 <br>

注意：如果您下载了.tar.xz文件，您可以通过**Windows Powershell**运行以下命令来解压tar文件：<br>

```sh
tar -xf yourfile.tar.xz
```

如果上面的行运行失败，可能是因为**tar**版本过旧，无法识别XZ压缩。您可以添加"**-J**"标志来解决问题。 <br>

```sh
tar -xJf yourfile.tar.xz
```

### 第三步: 设置环境变量
在Windows搜索栏搜索 "**环境变量**"，并选择"**编辑系统环境变量**". <br>

在系统属性窗口，点击“**环境变量...**”按钮。<br>

在“**系统变量**”部分，滚动并找到“**Path**”变量，然后选择它并点击“**编辑...**” <br>

在编辑环境变量窗口，点击“**新建**”并输入FFmpeg bin目录的路径。如果您遵循了常见的提取路径，这个路径可能会是**C:\FFmpeg\bin**。<br>

点击“**确定**”关闭所有打开的对话框并保存您的更改。

### 第四步: 验证您的安装
打开命令提示符或Powershell窗口 <br>

输入 `ffmpeg -version`。如果FFMPEG正确安装并添加到PATH，您应该会看到返回的FFMPEG版本和配置信息。 <br>

<br>

## MacOS 用户 🍎

我们推荐使用Homebrew来在MacOS上安装包。<br>

### 第一步: 安装Homebrew（如果Homebrew**未安装**）
将以下命令粘贴到您的终端并按回车。此命令也可以在[Homebrew网站] (https://brew.sh/)上找到：<br>

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

按照屏幕上的指示完成安装。<br>

### 第二部: 安装FFMPEG
安装Homebrew后，您现在可以通过在终端运行以下命令来安装FFmpeg：<br>

```sh
brew install ffmpeg
```

若需个性化安装，您可以通过运行以下命令查看可用选项

```sh
brew options ffmpeg
```

示例：带有选项的安装（libvpx编解码器和libvorbis音频编码器）:
```sh
brew install ffmpeg --with-libvpx --with-libvorbis
```

### 第三步: 验证您的安装
安装过程完成后，您可以通过运行以下命令来验证FFmpeg是否成功安装：<br>

```sh
ffmpeg -version
```

<br>

## Linux/Ubuntu用户 🐧

我们推荐在Linux/Ubuntu上使用apt进行包安装。 <br>

### 第一步: 更新包列表
在您的终端运行以下命令以确保您有更新的包列表<br>

```sh
sudo apt update
```

### 第二步：安装FFMPEG
包列表更新后可以通过运行以下命令安装FFmpeg:
```sh
sudo apt install ffmpeg
```

### 第三步: 验证您的安装
安装过程完成后，您可以通过运行以下命令来验证FFmpeg是否成功安装：<br>

```sh
ffmpeg -version
```

<br>

## 其他故障排除 🛟
如果您收到“ffmpeg不被识别为内部或外部命令”的报错，请确保您已将正确的路径添加到系统的PATH变量，并且没有错别字。 <br>

如果您最近将FFmpeg添加到了您的PATH，您可能需要重启您的命令提示符、PowerShell或您的计算机以确保PATH被更新。 <br>