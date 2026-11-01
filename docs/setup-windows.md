# Windows setup through WSL2

Use an x86-64 Windows computer with WSL2. The phone is not connected or used during conversion. These recipes target the Snapdragon S24; they do not target Exynos variants.

1. In an elevated PowerShell terminal, install Ubuntu 24.04:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   Restart if Windows requests it, open Ubuntu and finish its initial account setup. Check the distribution version with `wsl --list --verbose`; use WSL2. See Microsoft's [WSL installation guide](https://learn.microsoft.com/windows/wsl/install).

2. Ensure WSL exposes at least 24 GiB of RAM for the model export check. Its default memory limit may be lower. For a 32 GiB Windows host, a possible configuration is `memory=26GB` and `swap=16GB` under `[wsl2]` in `.wslconfig` in the Windows user profile. Choose limits that leave sufficient memory for Windows. Save running work before applying changes with `wsl --shutdown`, then reopen Ubuntu. See Microsoft's [WSL memory configuration](https://learn.microsoft.com/windows/wsl/wsl-config).

3. Install Git inside Ubuntu if needed, as described in the Linux setup guide, then clone this repository using its GitHub **Code** menu URL. Work in the Linux home filesystem for faster access to the many calibration files. Keep the project, SDK and virtual environments accessible to the same Ubuntu distribution.

4. Obtain the QAIRT SDK separately from Qualcomm. If its archive was downloaded in Windows, copy it into Ubuntu or extract it into the Linux filesystem. Select the extracted directory containing `sdk.yaml`, `bin` and `lib` as `QAIRT_SDK_ROOT`. The archive must contain the Linux x86-64 host tools.

5. Continue with [Linux setup](setup-linux.md) **inside Ubuntu**. Run all subsequent shell commands there, including `fetch`, `prepare`, `build` and `package`. Use Linux Python executables and Linux paths throughout that workflow.

There is one shared conversion implementation. WSL uses `x86_64-linux-clang` SDK tools and `.so` libraries. The host abstraction also supports native Windows SDK tools; the documented Windows setup uses WSL to share the Linux installation process.

For an existing Ubuntu 22.04 distribution, the export stage needs Python 3.12 while the SDK dependency installer expects Python 3.10. The Linux guide describes that split. Ubuntu 24.04 keeps both environments on Python 3.12.
