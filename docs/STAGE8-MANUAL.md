# CP2K 2026.2 Stage 8 双卡使用说明

版本：2026-11-14。适用于 `/home/nowr/cp2k-hip` 下已部署的 MI50 与 RX 9070 XT Stage 8。本文是当前操作入口；Stage 7 手册保留为历史版本。完整源码、运行适配层、说明书及校验清单见 [Stage 8 Release](https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-stage8-final)。

## 1. 版本、目录与适用范围

| 项目 | MI50 | RX 9070 XT |
|---|---|---|
| 架构 | gfx906 | gfx1201 |
| 固定设备 UUID | GPU-37ac3981732c7340 | GPU-6284fb6bbe682160 |
| 正式根目录 | `/home/nowr/cp2k-hip/build-mi50` | `/home/nowr/cp2k-hip/build-9070xt` |
| 日常入口 | `build-mi50/bin/cp2k.psmp` | `build-9070xt/bin/cp2k.psmp` |
| 执行环境 | 固定 digest 的 gfx906 Docker 镜像，单 MPI rank | 主机原生 gfx1201，单进程 |
| 选卡函数 | `use_mi50` | `use_9070xt` |
| Stage 7 备份 | `build-mi50-stage7backup` | `build-9070xt-stage7backup` |

表中相对路径均以 `/home/nowr/cp2k-hip` 为根。每卡的 `source/`、`build/cp2k/`、`install/cp2k/` 分别是源码、编译目录和安装目录。不要在其他位置编译 CP2K。Stage 8 正式运行器为 `runtime/stage8.py`；原 `runtime/stage7.py`、两份依赖 JSON 和 `/home/nowr/cp2k_env.sh` 保持原内容，供兼容和回滚使用。

Stage 8 增加纯 PBE 开壳层 UKS 的双自旋密度、梯度和两个 Vxc 通道驻留链路；不满足条件的方法保留原路径。revPBE、PBEsol 的原路径回退已有 O2 三重态测试。Cholesky 单 rank 路径与 ELPA 的 AUTO 选择策略同时更新。源码资格条件见 `docs/STAGE8.md`。

当前依然是每次计算使用一张卡。两卡可以分别处理独立作业；跨卡原型未接入 CP2K，没有合并显存。两卡独立并发的联合数值和吞吐验收不属于本次六案单卡验收。

## 2. 计算前检查

已部署用户无需重新编译、移动目录或执行第 10 节的维护步骤。先使用不启动 CP2K 的安装检查：

```bash
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp --check
```

两项均应打印 `CHECK_OK`。它检查安装文件身份及实际动态库解析，不运行 SCF、不证明新体系正确，也不代表 GPU 当前空闲。

检查已有覆盖变量、内存和作业：

```bash
printenv | grep -E '^(OMP_|OPENBLAS_|CP2K_GPU_|CP2K_PW_GPU_|CP2K_ELPA_|CP2K_CHOLESKY_|ROCR_|HIP_VISIBLE_)' || true
free -h
docker ps
cat /sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used
cat /sys/bus/pci/devices/0000:07:00.0/mem_info_vram_used
```

sysfs 以字节报告整卡占用。设备号是本部署的历史映射，换机或硬件调整后需重新确认。启动前还需确认同卡 CP2K、MLIP 或大模型任务已经结束；单次低占用读数不能替代任务归属检查。

每个计算使用独立目录，保持输入、坐标、基组/赝势、自定义文件和重启波函数的来源一致。不要在 home、项目、源码、安装、备份、依赖或磁盘根目录启动。目录及祖先不能是符号链接。MI50 只将当前算例目录作为可写工作区挂载，目录外的任意输入路径不会自动进入容器。

## 3. MI50 环境与 high 准备

在将要使用 MI50 的终端或 tmux 会话执行：

```bash
source /home/nowr/cp2k_env.sh
use_mi50
command -v cp2k.psmp
cat /sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level
cat /sys/bus/pci/devices/0000:03:00.0/pp_dpm_mclk
```

入口应为 `/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp`。现用环境脚本在 source 时调用 `lock_mi50_high`，保留用户要求的计算前高性能准备。检查 DPM 为 `high`，并查看显存频率带 `*` 的当前档位；既往回读为 `1000Mhz *`，不能仅凭 source 退出成功就认定调频成功。

`use_mi50` 只切换 PATH 与 UUID，不重复请求 high。已加载脚本后可在任务开始前调用 `lock_mi50_high` 并再次回读。直接调用绝对入口也不会替你执行 high 准备。该状态属于整张卡，不是 shell 私有设置；关闭终端不会保证恢复 auto。

source 默认选回 MI50，并可能保留原有诊断变量和预算。切换 RX 时随后明确执行 `use_9070xt`。每个终端的选卡环境独立，入口本身固定绑定对应 UUID，不要额外叠加全局 HIP 设备编号。

## 4. 单卡计算与退出码

以下命令中的算例目录和输入文件需要替换为实际已准备的路径。示例放入子 shell，前置检查失败即停止；计算的真实退出码会写入文件。输出名需未被已有计算使用。

MI50，先完成第 3 节准备：

```bash
(
  set -eu
  cd /media/nowr/Data/cp2k-work/system-A
  test -f input.inp
  /home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
  rc=0
  OMP_NUM_THREADS=16 /home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
    -i input.inp -o A.out > A.stdout 2> A.err || rc=$?
  printf '%s\n' "$rc" > A.exit
  exit "$rc"
)
```

RX，无需执行 MI50 的 high 准备：

```bash
(
  set -eu
  cd /media/nowr/Data/cp2k-work/system-B
  test -f input.inp
  /home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp --check
  rc=0
  OMP_NUM_THREADS=16 /home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp \
    -i input.inp -o B.out > B.stdout 2> B.err || rc=$?
  printf '%s\n' "$rc" > B.exit
  exit "$rc"
)
```

主输出为 `.out`，包装器/标准输出为 `.stdout`，GPU 统计及错误可能在 `.err`。exit 文件应为 0，还要核对 SCF 收敛、正常结束、最终能量与完整力。启动检查和容器初始化可能先于主输出出现，不要仅因短时间无 SCF 输出而再启动一个进程。

默认 16 个 OpenMP 线程；允许显式设置整数 1–32。OpenBLAS 常规默认 1；Cholesky 快路径会在受控局部调用中临时调整线程数并恢复。启动器固定 `OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=FALSE`、`OMP_STACKSIZE=256M`。外层设置 `OMP_PLACES` 或 taskset 包住 Docker 客户端，不能证明容器 CP2K 已绑定到指定 CCD。

## 5. 长任务与两卡独立作业

用命名 tmux 保持会话：

```bash
tmux new -s cp2k-mi50-A
```

在该会话内完成第 3 节，然后执行第 4 节 MI50 命令。按 Ctrl+B、D 脱离。另开 `tmux new -s cp2k-rx-B`，在 B 的独立目录执行 RX 命令。回看用 `tmux attach -t cp2k-mi50-A`；`tmux ls` 查看已有会话。不要重复创建同名会话来假定新任务已经启动。

双作业可以先从每卡 `OMP_NUM_THREADS=8` 起步，再实际测吞吐；8 不是已验证最优值。两个作业各用自己的输入、输出与重启文件，不在正式包装器外再套 `mpirun -np 2`。它们是独立进程，不共享一个计算的显存。

判断进度要看当前算例输出字节增长、求解器子进程 CPU 时间变化和当前计算阶段。容器 Up、PID 存在、累计完成计数或几次 GPU busy=0 都不能单独说明正常或卡死。MI50 每卡只运行一个受控 CP2K 容器；为观察当前任务而再调用一次包装入口会启动另一个计算。

## 6. 显存与完整运行开关

PW 预算只约束 PW 显式分配账本，不能覆盖 FFT、Grid、DBCSR、ELPA 及其他进程的全部显存。采样最大值是离散观测下界；`RES_RESERVE_MB` 是分配时余量检查，不是永久划出的空间。主机回退可能消耗大量 RAM，不是显存容量扩展方案；有其他任务共享主机时不要启用它来规避 OOM。

预算优先级为 `CP2K_GPU_MEM_LIMIT_MB_RANK0` → `CP2K_GPU_MEM_LIMIT_MB` → 输入 `MAX_GPU_MEMORY_MB`。变量名写 MB，换算实际按 MiB（1024 × 1024 字节）。默认不注入 PW 预算，0 表示该账本不限额；MI50 不默认限为 8 GB。`CP2K_GPU_HOST_FALLBACK` 非空时覆盖输入 `HOST_FALLBACK`；未设置时以输入为准。

以下默认值指正式入口且没有遗留覆盖。“1”只允许尝试，实际还受方法、布局和容量条件限制。变量必须在新进程启动前设置。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| CP2K_PW_GPU_RES | 1 | 驻留总开关；不等于所有 GPU 功能总开关 |
| CP2K_PW_GPU_RES_DENSITY | 1 | RHO_G/RHO_R 密度驻留 |
| CP2K_PW_GPU_RES_HARTREE | 1 | Hartree/DDAPC 驻留路径 |
| CP2K_PW_GPU_RES_XCIN | 1 | XC 使用驻留密度 |
| CP2K_PW_GPU_RES_POT | 1 | 势组装及势变换驻留 |
| CP2K_PW_GPU_RES_LAZY_DENSITY | 1 | 合法窗口内延迟密度下载 |
| CP2K_PW_GPU_RES_LAZY_POT | 1 | 合法窗口内延迟势下载 |
| CP2K_PW_GPU_RES_NATURAL_G | 1 | 自然 FFT 顺序 |
| CP2K_PW_GPU_RES_NATURAL_XC | 1 | 自然顺序 XC |
| CP2K_PW_GPU_RES_XC_PAIRS | 1 | 配对实数 FFT |
| CP2K_PW_GPU_RES_XC_DZ | 1 | 复用密度 FFT 的 z 梯度 |
| CP2K_PW_GPU_MAP_CACHE | 1 | 映射及 g 数据缓存 |
| CP2K_PW_GPU_FUSED_XC | 1 | 融合 XC 资格开关 |
| CP2K_PW_GPU_RES_VERIFY | 0 | 验证模式；关闭 lazy，改变执行路径与耗时 |
| CP2K_PW_GPU_RES_STATS | 1 | 驻留统计，受 RES 约束 |
| CP2K_PW_GPU_MEM_SAMPLE | 1 | 整卡离散显存采样；关闭后 PW 账本峰值仍可打印 |
| CP2K_PW_GPU_RES_RESERVE_MB | 1024 | 正式启动器注入的余量；裸源码默认另有区别 |
| CP2K_GPU_MEM_LIMIT_MB | 未设 | PW 预算覆盖 |
| CP2K_GPU_MEM_LIMIT_MB_RANK0 | 未设 | 优先级更高的当前单 rank 预算 |
| CP2K_GPU_HOST_FALLBACK | 未设 | 覆盖输入的主机回退选择 |
| CP2K_PW_GPU_UKS | 1 | Stage 8 双自旋 PW/UKS 路径 |
| CP2K_PW_GPU_XC_FULL_PIPELINE | 1 | UKS 融合 XC 流程 |
| CP2K_ELPA_GPU_N_MIN | 12288 | AUTO 选 AMD GPU 的矩阵阶数下限 |
| CP2K_CHOLESKY_FAST_SINGLE_RANK | 1 | 单 rank/单进程网格的 Cholesky 快路径 |

另转发 `CP2K_PW_GPU_PBE`、`CP2K_PW_GPU_PBE_LSD` 和 `CP2K_ELPA_GPU` 供诊断；它们不等于 UKS 或 ELPA 的总开关，具体读取位置见源码。布尔覆盖使用 0/1，数值用明确的非负整数。依赖链包括 RES → NATURAL_G → NATURAL_XC → XC_PAIRS → XC_DZ；关闭上游可能连带关闭下游。

单次排查示例，保持输入和起始波函数相同，并使用新输出名：

```bash
CP2K_PW_GPU_UKS=0 /home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
  -i input.inp -o uks-off.out > uks-off.stdout 2> uks-off.err
```

同样可单独设置 `CP2K_PW_GPU_XC_FULL_PIPELINE=0` 或 `CP2K_CHOLESKY_FAST_SINGLE_RANK=0`。关闭一项优化不是恢复未经修改的原版。清除遗留环境预算可用 `unset CP2K_GPU_MEM_LIMIT_MB CP2K_GPU_MEM_LIMIT_MB_RANK0`，但输入里的预算仍有效。现用环境脚本没有 `cp2k_defaults` 命令。

## 7. UKS、ELPA 与输入边界

UKS 新融合路径限定单个活跃的原生 PBE、原始 PBE 参数化、PW 导数、无密度平滑及不支持的额外请求。PBEsol、revPBE、混合泛函与其他不满足资格的情形保留原实现。`UKS .TRUE.`、电荷、多重度及泛函属于科学输入，不为节省显存或提速任意更改。

`CP2K_ELPA_GPU_N_MIN=12288` 是超出已测区间的保守 AUTO 策略，不是测得的精确 CPU/GPU 交叉点。设 0 可移除 AUTO 大小门槛；输入中显式 `ELPA_KERNEL AMD_GPU` 绕过大小门槛，但仍可能因能力、块大小或初始化失败回退。只有输入真正执行 DIAGONALIZATION 且计时显示对应调用时，才能评价对角化。OT-DIIS 耗时不能作为 ELPA 加速证据。

Cholesky 快路径要求一个 MPI rank、1 × 1 进程网格、当前没有活跃 OpenMP team，并满足本地存储条件。必要时尝试 HIP 三角操作；缺库、容量不足或执行失败回到 CPU。它没有重写 ELPA 特征值求解器。完整 SCF、ELPA 子阶段和 Cholesky 子阶段应分别报告，父子 timer 不相加。

## 8. 已验证结果与性能读法

两张卡各有六个正式路径实算：28 原子 PBE UKS 双重态、O2 三重态 PBE、O2 PBEsol 回退、O2 revPBE 回退、84 原子 RKS、512 Si 真实 DIAGONALIZATION（完整 512 原子力）。没有用两卡互相复算替代同卡可信参考。

固定数值门槛为能量绝对差 ≤ 1e-10 Ha、每个力分量绝对差 ≤ 1e-9 Ha/bohr；SCF 收敛和正常结束必须通过。UKS 同时核对积分自旋、绝对自旋与 alpha/beta 电子数；RKS 显式核对真实 RKS banner、单自旋态和总电子数，不伪造零自旋。

RX 六案均对照同卡切换前 Stage 7 输出。MI50 主 UKS 与三个 O2 案对照同卡 Stage 7；84 原子 RKS 对照同卡已验收 Stage 8 RKS；512 Si 对照同卡原 Cholesky 参考路径。后两项不是本次新取得的 Stage 7 独立参考。

| 同卡 28 原子 UKS，单次 CP2K 总计时 | 参考 / s | 正式 Stage 8 / s | 参考/新版 |
|---|---:|---:|---:|
| MI50 | 441.538 | 92.816 | 4.76 |
| RX 9070 XT | 397.516 | 86.402 | 4.60 |

表中是各自同卡、同输入比较，不是跨卡速度排名；CP2K 内部总计时不含外层身份检查与容器启动。MI50 完整 SCF 为 425.002→69.656 s，XC 为 297.460→26.924 s。此前开发候选的 71.626 s 不是本次正式入口成绩。

| 512 Si 对角化原始计时 | CP2K 总计时 / s | 完整 SCF / s | ELPA 子阶段 / s | Cholesky restore / s |
|---|---|---|---|---|
| MI50 | 106.483→80.938 | 98.775→72.423 | 31.489→32.028 | 32.291→6.745 |
| RX | 129.687→111.541 | 117.984→102.922 | 31.604→32.317 | 32.450→17.240 |

RX 对角化参考与早期主机编译重叠，仅保留原始计时，不用该行宣称严格加速。MI50 的收益来自 Cholesky/矩阵路径；两卡 ELPA 子阶段都没有因此证明求解器本体加速。不同输入、线程、缓存和负载下需重新测量。

真实验收不覆盖所有 GEO_OPT/MD 轨迹、多 rank、全部泛函、应力或所有预算组合。新体系先以相同输入、相同设备的可靠参考核对能量、完整力、收敛和实际路径，再推进长计算。`--timing-only` 与日志自比较不能授予分子数值验收。

## 9. 跨卡原型、容量与 GUI 入口

有界独立原型完成了 16/64/256 MiB 的双向 peer copy、远端显存 kernel 读写、回迁及全元素校验；它没有接入正式 CP2K。256 MiB 双向 copy 约 14.17/14.11 GB/s，这是 copy 吞吐。

包含校验和 host waits 的完整缓存流程，远端 RX cache 对 pinned-host 的五轮总耗时分别为 0.059508147/0.045220651 s、0.405763593/0.318688298 s、1.65114225/1.291301021 s，远端方案约慢 27%–32%，因此未默认启用。没有超单卡容量的 CP2K 实算、跨设备 VMM 或长期压力验证；不能把两卡显存相加用于估算可运行问题规模。

Ester 既有 RX 原生入口 `/home/nowr/cp2k-hip/build-9070xt/build/bin/cp2k.psmp` 已以正式 install ELF 的同 inode 硬链接保留，版本和动态库解析检查通过。它是特定 GUI 的兼容入口，命令行日常运行仍走 `bin/cp2k.psmp`。本次未做 Ester GUI 验收，未修改其配置。MI50 不能直接把容器 ELF 当作主机程序使用。

## 10. 源码下载、重建与升级

Release 的 `stage8-full-source-final.tar.gz` 是产生正式科学库的完整源码快照，SHA256 为 `a190656cb2dc727ee8a53d203629a2bf85fc8231dfe8703f82c3e02d01ae34de`。它与本仓库的定制覆盖层用途不同：本仓库没有根 CMakeLists.txt，不能直接当完整 CP2K 树构建。

`cp2k-stage8-runtime-and-tests.tar.gz` 提供正式运行器、原依赖配方、冻结 RX cache、更新的 mock/回归测试及说明。完整编译源包中的科学源码与覆盖层一致；其中开发测试并未全部同步到最新，更新测试以该配套包和本仓库为准。测试差异不被描述为正式计算源码差异。

下载后先在下载目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

校验清单覆盖本次各专用资产，不包含清单自身。不要把运行器包直接覆盖到正在使用的安装；它绑定绝对路径、固定镜像、依赖配方与源码/安装清单。新机器需要先准备匹配的依赖，完整源码包本身不是通用离线二进制安装器。

已有 Stage 8 用户无需重新做以下维护流程。受控重建时，先保留当前计算和版本证据，在 `/home/nowr/cp2k-hip` 内准备完整 source、所需 deps/recipe 及空输出目录；具体字段、安装和 resume 规则见 `docs/STAGE8-DEPLOYMENT.md`。MI50 使用 `stage8.py build --gpu mi50`；RX 必须先建立 `build-9070xt-stage8-validation`，运行 `build --gpu rx-validation`，完成同卡参考与候选的科学比较并写入绑定哈希的验收记录后，才允许 `build --gpu rx`。

本次临时候选编译产物已删除；`stage8-deployment-evidence/rx-candidate-retired` 只保留历史证据，不能作为新的可运行候选。正式 RX 必须在最终绝对路径 fresh configure/build/install，不能直接把候选目录改名转正。`certify-mi50` 仅用于本次受信任 legacy 收据的特殊身份导入，不是日常重建或跳过科学验收的方法。

GitHub 更新仅发布源码与说明，不会自动升级目标机安装。再次维护已部署目录时应另行安排停止相关作业、备份、构建、验收与切换。

## 11. 故障与回滚

| 现象 | 检查与处理 |
|---|---|
| CHECK_OK 不通过 | 保留原错误，核对 wrapper、ELF、libcp2k、源码与 ldd；不要手改 manifest 绕过身份门禁 |
| invalid device / invalid image | 核对 gfx906/gfx1201 与固定入口，排除混用库或额外 HIP 编号 |
| owned directory / symlink / protected root | 使用合法、归本人所有且无链接的独立计算目录；不要扩大容器挂载到整个 home |
| 主机有文件、容器找不到 | 核对算例挂载和输入引用，把算例专用文件放入算例目录 |
| OOM / hipMalloc | 检查整卡及主机占用、网格和所有库分配；不擅自改单个体系科学参数 |
| 无统计或 fallback=0 | 同时检查正向调用、RES_STATS 和正常终止；零 fallback 不证明路径激活 |
| tmux 启动成功但没有结果 | 检查实际子进程、输入准备、完整序列退出码及首个错误，不能只看启动命令 exit 0 |
| 时间异常 | 核对负载、DPM、输入/WFN、线程、缓存与计时口径 |

两份 Stage 7 备份保留原绝对路径，不能直接在 `*-stage7backup` 内运行。回滚前先停止使用目标目录的计算，保留 Stage 8；以下是维护步骤，不应在作业进行中执行：

```bash
(
  set -eu
  cd /home/nowr/cp2k-hip
  test -d build-mi50-stage7backup
  test -d build-9070xt-stage7backup
  test ! -e build-mi50-stage8failed
  test ! -e build-9070xt-stage8failed
  mv -T -- build-mi50 build-mi50-stage8failed
  mv -T -- build-mi50-stage7backup build-mi50
  mv -T -- build-9070xt build-9070xt-stage8failed
  mv -T -- build-9070xt-stage7backup build-9070xt
  build-mi50/bin/cp2k.psmp --check
  build-9070xt/bin/cp2k.psmp --check
)
```

原 Stage 7 运行器与配方保留，恢复 canonical 目录名后恢复其引用。这里的两卡移动不是原子事务，任一步失败应保留当前目录状态逐项核对，不盲目重跑。本次只核对了备份字节和恢复条件，没有实际执行回滚实算。

清理只限已核实不再引用的旧 build/install/cache/原生探针，保留源码、原始输出、检查点和备份。共享 `/opt/dbcsr-rocm101`、Spack、data 和现用 deps 仍是依赖，不能按目录名当旧产物删除。

## 12. 测试、身份与资料

本轮正式科学验收已完成两卡各六案；发布前再次在 Linux 执行运行器 40/40、比较工具 17/17 测试，两正式入口再次 CHECK_OK。科学源码的已有独立审查、真实编译和数值验收，与本轮仅发布文档/资产的核验分别记录；没有因更新手册重新运行 GPU 基准。

| 正式共享库 | SHA256 |
|---|---|
| MI50 libcp2k.so.2026.2 | 6e81915c8331d74cf93d4fb8c58b486c1444aebbf51658c3210adc12f2f8306c |
| RX libcp2k.so.2026.2 | c787e3d5acbbd4f1b951376958491a810cbc1c677225126a4e5b0d00b51b9e8f |

主 ELF 可能不变而科学库已更新，不能仅凭可执行文件或版本串识别 Stage 8。源包 SHA 与源码逐文件清单摘要是不同对象；后者为 `81833f1cd07dd475e37f2bf1a8b8ca5aec750bf7f7877406e47aecb8fd8b8ad9`。

远端原始部署证据位于 `/home/nowr/cp2k-hip/stage8-deployment-evidence/`；算例位于两正式目录的 `validation/`。内部审查、失败现场和复盘继续保留于本地项目档案。GitHub 公开使用说明、源码/工具及必要验收摘要，不上传内部审查会话、原始代理事件流或本机档案路径。

配套文件：`docs/STAGE8.md` 为实现与开发测试说明，`docs/STAGE8-DEPLOYMENT.md` 为维护运行器说明，`docs/STAGE8-VALIDATION.md` 为公开验收摘要。`tools/stage8-validation/parse_compare.py` 可离线比较日志；同目录的历史 `stage8_runner.py` 仅随既有比较工具测试保留，含已退役开发路径，不用于正式构建或运行。
