# CP2K 2026.2 · MI50 Stage 6

本仓库记录 AMD MI50（gfx906）上的 CP2K 2026.2 Stage 6 GPU 流水线改动。**仓库根目录中的 `src/`、`tests/` 是历史增量文件；GitHub 自动生成的 “Source code” 下载不是完整可构建源码。** 完整交付请下载下面的专用 Release 资产。

[完整源码与构建包：Stage 6 r3](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-mi50-stage6-r3)

专用资产 `cp2k-2026.2-mi50-stage6-source-20261106-r3.tar.gz` 包含 CP2K 根 CMake、完整 `src/data/tests/tools`、上游公开算例、许可证、构建/运行入口和逐文件校验清单。该包源自目标机实际使用的完整源码，物理代码保持原始字节与权限；仅修正一处 ELPA mock 的过期预期，并附原始/替换摘要和补丁。

这是面向既有 Ubuntu / ROCm 环境的源码交付。Docker 镜像、ROCm、Spack 和已安装依赖不在归档内；需要保留包内 README 列出的精确依赖。它不是包含所有依赖的离线安装器。

```bash
archive=cp2k-2026.2-mi50-stage6-source-20261106-r3.tar.gz
sha256sum -c "$archive.sha256"
tar -xzf "$archive"
package="$PWD/${archive%.tar.gz}"
bash "$package/scripts/verify.sh" --archive "$PWD/$archive" --launcher-tests
bash "$package/scripts/build.sh" --prefix "$HOME/cp2k-mi50-stage6-release-r3" --jobs 16
bash "$package/scripts/run.sh" --prefix "$HOME/cp2k-mi50-stage6-release-r3" --check
```

使用新的空安装目录。编译完成后，在具体算例目录运行：

```bash
"$HOME/cp2k-mi50-stage6-release-r3/bin/cp2k.psmp" -i input.inp -o run.out
```

运行入口固定绑定 MI50 UUID，采用单 MPI rank；默认 OpenMP 16 线程、OpenBLAS 1 线程，缓存跨进程保留。MI50 不设置默认 8 GiB 限制。此入口不跟随 `use_9070xt` 切卡，也不代表多卡已验收；源码的其它运行能力和旧安装仍保留。

具体构建、算例和测试结果按归档摘要记录在同一 Release 的 `VALIDATION.md`。PW residency 与 ELPA mock、真实 MI50 运行、独立打包审查分别报告。独立的旧 `pw_gpu_integrated` 测试入口仍缺两个自然序内核的 mock，链接失败；不能宣称全部 CP2K 测试通过。真实 ELPA 对角化、多 rank、GEO_OPT/MD、UKS 与非正交输入尚未全面验收，累积 Stage 6 物理源码也尚无完整独立审查。

旧 `deliver_stage6_full_20261106.tar.gz` 是增量覆盖包，保留用于追溯，已被上述完整源码包替代。原版 CP2K 及各组件版权和许可证继续适用，详见包内 `LICENSE`、`NOTICE` 与源码文件头。

## ELPA 调用封装优化

Stage 6 r3 已包含 ELPA 对象/setup 复用、GPU/CPU kernel 选择及 QR 状态处理；本次将根目录 mock 测试与 r3 已接受版本对齐，生产代码未变。参见[优化与启用说明](docs/ELPA-OPTIMIZATION.md)和[测试说明](tests/elpa_gpu/README.md)。现有套件通过 32 个 mock 配置及无 ELPA 编译检查；真实 GPU 对角化、多 MPI rank、ELPA 数值准确性与性能仍待验收。
