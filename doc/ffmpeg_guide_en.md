# FFMPEG Installation Guide

This process involves downloading the FFmpeg binaries and adding them to your system's PATH so you can run FFmpeg from the Command Prompt or PowerShell.

<br>

## For WINDOWS Users 🪟

### Step 1: Download FFMPEG
NOTE: We recommend downloading FFMPEG from **official sources ONLY** by visiting [**here**](https://ffmpeg.org/download.html#build-windows), <br>

or by visiting [**here**](https://github.com/BtbN/FFmpeg-Builds/releases) for the latest stable releases. 

### Step 2: Extract FFMPEG file
Unzip the .7z package at the directory you prefer. <br>

NOTE: If you downloaded a .tar.xz file, you may run the following commends through **Windows Powershell** to unzip a tar file: <br>

```sh
tar -xf yourfile.tar.xz
```

If the above line fails to run, it may be due to the **tar** version being outdated and is not able to recognize a XZ compression. You can add a **-J** flag to resolve the issue. <br>

```sh
tar -xJf yourfile.tar.xz
```

### Step 3: Setting Up Environment Variables:
Search for "**Environment Variables**" in the Windows search bar and select "**Edit the system environment variables**". <br>

In the System Properties window, click on the "**Environment Variables...**" button. <br>

Under the "**System variables**" section, scroll and find the "**Path**" variable, then select it and click "**Edit...**"

In the Edit Environment Variable window, click "**New**" and enter the path to the FFmpeg bin directory. This path will likely be something like __C:\FFmpeg\bin__ if you followed the common extraction path.

Click "**OK**" to close each of the open dialogs and save your changes.


### Step 4: Verifying your Installation
Open a Command Prompt or Powershell Window <br>

Enter `ffmpeg -version`. If FFMPEG is correctly installed and added to PATH, you should see the FFMPEG version and configuration information returned. <br>

<br>

## For MacOS Users 🍎

We recommend using Homebrew for package installation on MacOS. <br>

### Step 1: Install Homebrew (If Homebrew **NOT** Installed)
Paste the following command into your Terminal and press Enter. This command is also available on [the Homebrew website] (https://brew.sh/): <br>

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions to complete the installation.<br>

### Step 2: Installing FFMPEG
With Homebrew installed, you can now install FFmpeg by running the following command in Terminal: <br>

```sh
brew install ffmpeg
```

For customized installation, you can view available options by running

```sh
brew options ffmpeg
```

An example for installing with options (libvpx codec and libvorbis audio encoder):
```sh
brew install ffmpeg --with-libvpx --with-libvorbis
```

### Step 3: Verifying your Installation
After the installation process is complete, you can verify that FFmpeg has been successfully installed by running: <br>

```sh
ffmpeg -version
```

<br>

## For Linux/Ubuntu Users 🐧

We recommend using apt for package installation on Linux/Ubuntu. <br>

### Step 1: Update Package Lists
Run the following command in your terminal to make sure you have the updated package lists<br>

```sh
sudo apt update
```

Follow the on-screen instructions to complete the installation.<br>

### Step 2: Installing FFMPEG
Once the package list is updated, install FFmpeg by running:
```sh
sudo apt install ffmpeg
```

### Step 3: Verifying your Installation
After the installation process is complete, you can verify that FFmpeg has been successfully installed by running: <br>

```sh
ffmpeg -version
```

<br>

## Other Troubleshoots 🛟
If you receive an error that "**ffmpeg is not recognized as an internal or external command**", ensure you've added the correct path to the system's PATH variable and that there are no typos. <br>

If you've recently added FFmpeg to your PATH, you might need to restart your Command Prompt, PowerShell, or your computer to ensure the PATH is updated. <br>