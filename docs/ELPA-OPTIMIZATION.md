# CP2K ELPA 调用封装优化

优化位于 [`src/fm/cp_fm_elpa.F`](../src/fm/cp_fm_elpa.F)，针对 CP2K 调用 ELPA 时的对象生命周期、配置与重复 setup 开销，不涉及新 ELPA GPU 内核或物理算法改动。生产代码已包含在 CP2K 2026.2 MI50 **Stage 6 r3** 中；本次仅将根目录 mock 测试同步到 r3 已接受版本并补充文档，未修改生产代码。

## 已实现的行为

- 缓存多态 `CLASS(elpa_t)` 对象并保留其父并行环境，相同配置复用 allocation/setup。缓存键包括矩阵阶数 `n`、块大小 `nblk`、通信器、`neig`、本地行列数、进程网格坐标、kernel、solver 和 OpenMP 线程数。
- 每次通过 `group%max(rebuild)` 汇总失效决定，任一 rank 的变化要求集体重建。替换时先释放 ELPA 对象再释放父环境；最终清理也先释放缓存，再调用 `elpa_uninit`。
- 仅解析为 `AMD_GPU` 时探测 `can_set("gpu", 1)` 和 `can_set("amd-gpu", 1)`，在设置 kernel 前启用受支持的选项。显式 CPU kernel 在 HIP 构建中仍保持 CPU 路径，从 GPU 切回 CPU 会重建对象，清除旧标志。NVIDIA/Intel 的既有启用分支保留。
- 仅在 ELPA 支持时设置 `omp_threads`。两阶段 solver 在每次求解前重设 QR，包括诊断求解临时关闭 QR 后的下一次调用。

复用范围是 wrapper 持有的 ELPA 对象及其 setup；没有验证或承诺全部 ELPA 内部工作缓冲都被缓存。

## 启用与选择

完整构建需启用 `CP2K_USE_ELPA=ON` 并链接兼容 ELPA。AMD GPU 路径需要具备相应 ROCm 支持的 ELPA；HIP 构建使用 `CP2K_USE_ACCEL=HIP`、`CP2K_ENABLE_ELPA_GPU=ON`，OpenMP 支持使用 `CP2K_ENABLE_ELPA_OPENMP_SUPPORT=ON`。具体依赖与构建步骤沿用完整 Release 的配方。

在实际采用对角化的输入中，将以下选项合入已有 `&GLOBAL`：

```text
&GLOBAL
  PREFERRED_DIAG_LIBRARY ELPA
  ELPA_KERNEL AUTO
&END GLOBAL
```

缓存随 ELPA 调用自动生效，无额外缓存开关。仅设置首选库不会将 OT 计算变成 ELPA 对角化；小问题还可能因 `ELPA_NEIGVEC_MIN` 回退到 ScaLAPACK。

| 选择 | wrapper 行为 |
|---|---|
| `AUTO` | 未定义 `__NO_OFFLOAD_ELPA` 时，HIP/CUDA/OpenCL 构建分别选择 AMD/NVIDIA/Intel GPU kernel；否则按 CPU 能力选择或使用 ELPA 默认值。 |
| `AMD_GPU` | 显式请求 AMD 路径，不依赖 HIP 宏；能否实际运行取决于所链接 ELPA 和设备支持。 |
| `GENERIC` 等 CPU kernel | 保持 CPU 路径，包括 HIP 构建以及从 GPU 切回 CPU 的情况。 |
| `NVIDIA_GPU` / `INTEL_GPU` | 保留既有后端选择，仍需相应 ELPA 构建与设备。 |

`real_kernel` 和 QR 设置适用于两阶段 solver；选择 `ELPA_ONE_STAGE` 时不设置这两项。

## 测试与适用边界

在仓库根目录执行 `python3 tests/elpa_gpu/run_tests.py`，需要 Python 3.9+ 和支持 OpenMP 的 GNU Fortran。本次在 WSL Ubuntu 24.04 通过现有套件：完整生产 Fortran wrapper 配合 CP2K/ELPA mocks，CPU/HIP/CUDA/OpenCL 四种编译变体 × 八种 GPU/OpenMP 能力组合，共 **32 个配置**，另通过无 ELPA 编译检查。覆盖缓存复用与失效、模拟远端 rank 失效、通信器生命周期、CPU/AUTO/GPU 切换及 QR 恢复；详情见 [测试说明](../tests/elpa_gpu/README.md)。

这些测试没有运行真实 GPU 或 MPI。此项工作尚未测量 MI50 或 RX 上真实 ELPA 路径的数值准确性和性能，也没有专门的 ELPA 对角化前后对比；既有 84 原子 OT DIIS 数据不能作为 ELPA 加速结果。真实 DIAG 与参考求解器对照、多 MPI rank 运行仍待验收。

既有 RX 构建记录显示 ELPA、ELPA GPU 和 OpenMP 选项已开启，动态依赖解析无缺失；已安装 ELPA 2026.02.002 链接 rocSOLVER/rocBLAS 并含 `gfx1201` 目标字符串。这些只证明配置、链接和目标信息，不能证明运行正确，也不代表所有可选库均启用或完成物理验证。RDNA4/RX 支持仍须通过面向 `gfx1201` 的新构建和真实 DIAG 验证；候选源码一致不等于已完成新 RX 构建。

## 源码来源与完整交付

来源为既有 MI50 Stage 6 完整源码快照；本次沿用公开 wrapper 和 r3 测试，按 Release 归档摘要与 manifest 追溯。下载 [Stage 6 r3 完整源码 Release](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-mi50-stage6-r3) 中的专用资产 `cp2k-2026.2-mi50-stage6-source-20261106-r3.tar.gz`，并查阅其 `VALIDATION.md`。仓库根目录是历史增量文件，GitHub 自动生成的 “Source code” 下载不是完整可构建源码；依赖与历史构建入口见[仓库首页](../README.md)。
