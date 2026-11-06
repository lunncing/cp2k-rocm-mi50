# CP2K 2026.2 · Stage 7 Final (MI50 & RX 9070 XT Dual-GPU)

本仓库记录 AMD Instinct MI50（gfx906）与 Radeon RX 9070 XT（gfx1201）上的 CP2K 2026.2 Stage 7 最终优化版。

**Stage 7 已经定版并停止进一步优化，支持每张卡唯一一套正式构建。**
仓库根目录中的 、 为核心改动源码与运行适配层。完整可直接构建的源码包请下载下方的 Release 专用资产。

## 最新正式发布

👉 **[CP2K 2026.2 Stage 7 Final 完整发布资产与操作手册](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-stage7-final)**

Release 中包含以下专用资产：
- （9053 项完整源码，包含根 CMakeLists.txt、src、data、tests、tools、许可证及构建脚本）
- （相对 CP2K 2026.2 原版的完整累积补丁）
- （完整 Git 历史归档）
-  /  / （完整操作说明书，涵盖单卡运行、独立双卡 A/B 并发、全部 20 个诊断变量、显存预算与故障排查）
- （校验清单）

---

## 核心特性与 Stage 7 收益

1. **流水线全显存常驻（VRAM Residency）**：
   - 彻底消除 SCF 迭代中实空间网格与 G 空间密度的主机 PCIe 往返；
   - 延迟回传窗口（Lazy Retention）将中间势（{	ext{tot}}$, {	ext{xc}}$, {	ext{H}}$）永久留在显存；
   - 自然序 FFT 盒子流式处理，完全旁路 scatter/gather。
2. **算子融合与实数配对 FFT**：
   - 密度的逆向 FFT 顺带计算 $ 方向梯度（$\partialho/\partial z$），XC 每步少做一次三维 FFT；
   - 自然序 XC 实数配对 FFT（7 次 3D FFT 缩减为 5 次）；
   - 原生 PBE 泛函导数融合核。
3. **Stage 7 网格建立并行化**：
   - 启动时的网格建立（赋值、计数、映射、gidx、壳层与 gather）实现完整多线程并行，大幅缩短几何优化/多次单点计算的初始化冷启动时间。
4. **两张卡各一套构建并存**：
   - **MI50**（PCI , gfx906）：位于 ，运行于固定容器环境；
   - **RX 9070 XT**（PCI , gfx1201）：位于 ，主机原生运行；
   - 既有主要科学计算库（ELPA、DFT-D4、SIRIUS、COSMA、DLA-Future、DBCSR）功能完整保留。

---

## 基准实测数据（84 原子电解液团簇，16 线程）

| GPU / Cutoff | SCF 内循环步数 | 能量绝对偏差 / Ha | 最大力偏差 / Ha·bohr⁻¹ | CP2K 最终总耗时 | 平均单步 SCF 耗时 |
|---|:---:|:---:|:---:|:---:|:---:|
| **MI50** / 500 Ry | 9 步 | 1.70e-13 | 2.39e-12 | **12.58 s** | **0.48 s / 步** |
| **MI50** / 320 Ry | 12 步 | 2.27e-13 | 2.60e-13 | **8.87 s** | **0.30 s / 步** |
| **RX 9070 XT** / 500 Ry | 9 步 | 1.14e-13 | 7.18e-13 | **16.30 s** | **0.83 s / 步** |
| **RX 9070 XT** / 320 Ry | 12 步 | 2.85e-13 | 2.76e-12 | **13.14 s** | **0.60 s / 步** |

> 注：四组回归全部退出码为 0，数值严格通过 $|\Delta E| \le 10^{-10}$ Ha、全部 84 原子 252 个力分量最大差 $\le 10^{-9}$ Ha/bohr 门槛，底层 [PW_GPU] 路径 fallback 均为 0。

---

## 快速使用

加载环境并选择目标 GPU：



运行计算：



支持两张卡同时计算互不干扰的两个独立体系（例如 MI50 算体系 A，RX 算体系 B），具体命令与排错说明详见 [操作说明](docs/STAGE7-MANUAL.md)。

---

## 历史发布

- [v2026.2-elpa-wrapper-r1](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-elpa-wrapper-r1)（ELPA 调用封装源码补充版）
- [v2026.2-mi50-stage6-r3](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-mi50-stage6-r3)（Stage 6 完整发布版）
- [v2026.2-mi50-stage6](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-mi50-stage6)（Stage 6 初版补丁）
