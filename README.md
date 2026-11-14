# CP2K 2026.2 · Stage 8（MI50 / RX 9070 XT）

Stage 8 已在 MI50（gfx906）与 RX 9070 XT（gfx1201）各自的正式路径完成构建及六案数值验收。所有 CP2K 构建集中在 `/home/nowr/cp2k-hip`，每卡保留一套正式安装和原 Stage 7 备份。

**[Stage 8 正式发布与下载](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-stage8-final)** · **[当前使用说明](docs/STAGE8-MANUAL.md)** · [实现与开发测试](docs/STAGE8.md) · [受控重建与回滚](docs/STAGE8-DEPLOYMENT.md) · [验收摘要](docs/STAGE8-VALIDATION.md)

## 本版变化

- 纯 PBE 开壳层 UKS 的双自旋密度、梯度、两个 Vxc 通道及驻留生命周期；保留不支持方法的原路径回退。
- 单 MPI rank 的 Cholesky/三角矩阵快路径，以及失败时的 CPU 回退。
- 保留 ELPA handle 缓存与 AMD GPU 能力检查，AUTO 使用保守的矩阵大小门槛。
- 正式运行器绑定源码、架构、依赖、ELF、libcp2k、配置和实际动态库映射；RX 必须先验证候选，再在最终路径重新构建。

跨卡显存目前仅有独立原型，未接入 CP2K，没有合并显存。两卡分别计算独立作业时，共享主机资源仍可能影响吞吐。

## 正式入口

| GPU | 日常命令行入口 | 原 Stage 7 备份 |
|---|---|---|
| MI50 / gfx906 | `/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp` | `/home/nowr/cp2k-hip/build-mi50-stage7backup` |
| RX 9070 XT / gfx1201 | `/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp` | `/home/nowr/cp2k-hip/build-9070xt-stage7backup` |

先检查安装身份，以下不运行 CP2K：

```bash
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp --check
```

已部署用户无需重建。MI50 运行前加载既有环境并保留 high 准备：

```bash
source /home/nowr/cp2k_env.sh
use_mi50
cat /sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level
cat /sys/bus/pci/devices/0000:03:00.0/pp_dpm_mclk
```

随后进入已准备好的独立算例目录，核对输入和同卡负载，再运行：

```bash
OMP_NUM_THREADS=16 /home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
  -i input.inp -o output.out > output.stdout 2> output.err
```

RX 改用其绝对包装入口。完整退出码记录、tmux、双作业组织、全部运行开关、显存预算、回滚及 GUI 兼容入口见 [Stage 8 使用说明](docs/STAGE8-MANUAL.md)。

## 实测结果与边界

各卡均通过真正开壳层 PBE UKS、O2 三重态、PBEsol/revPBE 回退、84 原子 RKS 和 512 Si 真实 DIAGONALIZATION。固定数值门槛为能量绝对差 ≤ 1e-10 Ha、每个力分量绝对差 ≤ 1e-9 Ha/bohr，并检查 SCF、正常终止和适用的自旋/电子数证据。

| 28 原子 UKS，同卡相同输入的单次 CP2K 总计时 | 参考 / s | 正式 Stage 8 / s |
|---|---:|---:|
| MI50 | 441.538 | 92.816 |
| RX 9070 XT | 397.516 | 86.402 |

这些是各卡自身的软件版本比较，不是跨卡排名或所有体系的速度承诺。512 Si 的收益归于 Cholesky/矩阵路径，不能宣称 ELPA 求解器本体已加速。完整 SCF 与 ELPA 子阶段、参考来源和计时限制见 [验收摘要](docs/STAGE8-VALIDATION.md)。

## 下载资产与源码关系

本仓库是定制源码、运行器和测试的覆盖层，根目录没有完整 CP2K 构建系统。可构建源码请下载 Release 专用资产：

| 资产 | 用途 |
|---|---|
| `stage8-full-source-final.tar.gz` | 产生正式科学库的完整 CP2K 源码快照，含根 CMakeLists.txt、data、许可证及构建工具 |
| `cp2k-stage8-runtime-and-tests.tar.gz` | 正式运行器、原依赖配方、冻结 RX cache、当前回归测试与说明；在解包的完整源树内解开以补充 runtime/tests/docs，科学源码不变 |
| `cp2k-stage7-to-stage8.patch` | 本仓库覆盖层从 Stage 7 发布提交到本版的变化；只适用于本仓库，不是上游 CP2K 通用补丁 |
| `CP2K-Stage8-Manual.md` / `.docx` / `.pdf` | 当前完整操作说明 |
| `Stage8-Validation.md` | 公开的数值、性能和适用范围摘要 |
| `SHA256SUMS.txt` | 本次专用资产的 SHA256 校验清单 |

完整源包保留原编译快照；其中开发测试并未全部同步到最新。更新的 mock/回归测试以本仓库及配套运行器测试包为准。依赖环境和绝对目录仍需按维护文档准备，源包不是通用离线二进制安装器。GitHub 更新不会自动改变目标机安装。

## 历史版本

- [Stage 7 Final](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-stage7-final)，[历史使用说明](docs/STAGE7-MANUAL.md)。
- [ELPA wrapper r1](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-elpa-wrapper-r1)。
- [Stage 6 r3](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-mi50-stage6-r3)。

历史 tag、资产与手册原件保留。当前日常操作和 Stage 8 验证边界以上述新手册为准。
