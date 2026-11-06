# CP2K 2026.2 Stage7 双卡操作说明

按实际部署核对的修订版 R2 · 2026-11-07 · 使用者：柳毅

R2 修正：保留原环境脚本的 MI50 计算前 high 准备。R1 将这一操作删除，误解了用户避免低频升频等待的实际用途；本版撤回该建议，随包附目标机现用 cp2k_env.sh 的原样副本。

本文替代 Claude 原稿中已经过时的安装、入口、回滚与使用边界说明。Stage7 已正式部署，两张卡各保留一套构建。可以让 MI50 计算体系 A，同时让 RX 9070 XT 计算独立的体系 B。两个作业各有自己的进程、GPU 和工作目录；这与一个体系同时使用两张卡是不同的运行方式。

本次以目标机只读回查、冻结源码、原始验收记录为依据。现场确认了两个正式入口的 CHECK_OK、源码提交、构建目录、环境脚本和硬件。没有启动新的 GPU 计算，没有修改目标机配置、输入或构建，也没有向 GitHub 写入。双卡独立并发的运行组织可直接配置；并发数值、资源争用和吞吐尚未做实机联合验收。

## 1. 当前版本与唯一入口

| 项目 | MI50 | RX 9070 XT |
|---|---|---|
| 架构与设备 | gfx906；GPU-37ac3981732c7340 | gfx1201；GPU-6284fb6bbe682160 |
| PCI 地址 | 0000:03:00.0 | 0000:07:00.0 |
| 唯一构建目录 | /home/nowr/cp2k-hip/build-mi50 | /home/nowr/cp2k-hip/build-9070xt |
| 日常入口，相对上述目录 | bin/cp2k.psmp | bin/cp2k.psmp |
| 实际运行环境 | 固定 gfx906 Docker 镜像，单 MPI rank | 主机原生 gfx1201 环境，直接单进程 |
| 环境切换函数 | use_mi50 | use_9070xt |
| 当前 DPM，只读观测 | high；当时核心 1700 MHz、显存 1000 MHz | auto；频率随负载变化 |

共用源码位于 `/home/nowr/cp2k-hip`，本地分支为 `main`，提交为 `5a0097832e52f1e5a8b74042028f6ab46d5b5a70`；本次回查工作树干净。共用启动工具在 `runtime/`。构建目录中的 `source/`、`build/`、`install/` 是该套构建的源码、编译和安装层，不是三套版本。

旧 `build-rocm101` 和 `build-mi50-stage7` 已经删除；旧版回滚材料只保存在 Windows。`/opt/dbcsr-rocm101` 是仍在使用的依赖，必须保留，不能因名称相似而删除。

MI50 和 RX 都使用各自的 `bin/cp2k.psmp` 包装入口。不要绕到 `install/cp2k/bin/cp2k.psmp`：MI50 的 ELF 需要容器库环境，RX 的包装入口也负责依赖与身份检查、设备选择和环境隔离。直接改 ROCR_VISIBLE_DEVICES 不会把 MI50 包装入口变成 RX 入口；正式包装入口固定绑定本卡 UUID。

现场硬件为 Ryzen 9 9950X，16 核、32 逻辑 CPU；Linux 报告物理内存总量 98,762,264,576 字节。Ubuntu 24.04.4 LTS，内核 6.17.0-35-generic；主机 ROCm 路径 `/opt/rocm/core-10.1`，版本文件为 10.1.0。两卡 sysfs 当前链路均为 x16，MI50 为 16.0 GT/s、RX 为 32.0 GT/s。原稿“双卡都是 CPU 直连 x8”与本次链路回查不符；这里不另行推断主板上游拓扑。

## 2. 计算前的环境配置

### 2.1 已部署脚本的实际行为

当前文件为 `/home/nowr/cp2k_env.sh`。用户提交的 `cp2k_env-2.sh` 与它逐字节相同，SHA256 为 `2b409a491e36f2eb6df77ab2467a9bf37cde93491a1cfc471c63b6bb45f72b88`。

加载后默认选择 MI50；`use_mi50` / `use_9070xt` 同时切换 PATH 和 UUID。每个终端有自己的环境，因此终端 B 切卡不会改变终端 A 已启动的进程。重新 source 又会默认选回 MI50；随后按作业需要明确调用选卡函数。

现有脚本在 source 时调用 `lock_mi50_high`：若 MI50 当时不是 high，就通过既有 Docker 方式请求 high。这是用户有意保留的计算前准备，用于提前将 MI50 从低频待机状态切到高性能状态，避免等计算开始后才升频。应保留该行为；R1 将它删除的建议已撤回。本次未测量升频等待具体多长，不新增性能收益数值。

原脚本不等待显存频率稳定，也会吞掉 Docker 写入失败的退出码。因此不能只看 source 的返回值；按 §2.5 回读 DPM 和实际显存频率。重复 source 仍会前置 PATH / LD_LIBRARY_PATH，且不会清掉旧诊断开关或 PW 预算；这两点与保留计算前 high 是不同事项。

R2 随包附 `cp2k_env.sh`，与用户附件及目标机现用脚本逐字节一致，保留自动 high 函数。R1 的无调频 `cp2k_env-reviewed.sh` 不再推荐使用，R2 不包含它；其 cp2k_status / cp2k_defaults 也不属于现用脚本。目标机脚本从未被 R1 替换，无需远程回滚。

### 2.2 已固定的值与用户可调的值

| 配置 | 正式启动器的实际规则 | 日常建议 |
|---|---|---|
| OMP_NUM_THREADS | 默认 16；接受整数 1–32；保留调用者明确设置 | 单卡先用 16；双作业先用各 8，随后再按吞吐测量调整 |
| OPENBLAS_NUM_THREADS | 固定 1 | 不另行提高，避免嵌套线程争用 |
| OMP_DYNAMIC | 固定 FALSE | 外层同名设置会被启动器覆盖 |
| OMP_PROC_BIND | 固定 FALSE | 外层设 TRUE 不等于已绑 CCD |
| OMP_STACKSIZE | 固定 256M | 不把外层改值当作已生效 |
| GPU 可见性 | 包装入口重建环境，仅暴露对应 UUID | 不额外叠加全局编号的 HIP_VISIBLE_DEVICES |
| 库路径 / CP2K_DATA_DIR | 启动器按固定配方重建 | 无需自行 spack load 或拼接库路径 |
| 驻留预留量 | 启动器未收到设置时传 1024 MiB | 保持验收值；源码裸默认 1536 MiB 是另一个层次 |
| 内部显存采样 | 默认 1；显式 0 可关闭 | 换体系先保持开启 |
| PW 显存预算 | 启动器不默认注入预算；输入默认 0 | MI50 不固定限 8 GB；RX 按实际需求单独设置 |

启动器使用环境白名单。除 OMP_NUM_THREADS 与第 6 节列出的 CP2K 变量外，其他外层变量不能默认视为会传入实际 CP2K。尤其 `OMP_PLACES`、`GOMP_CPU_AFFINITY`、外层 MPI 参数和自定义 LD_LIBRARY_PATH 不能用来证明当前正式进程已经采用相应设置。另需区分：环境脚本设置 HWLOC_COMPONENTS=-gl，RX 配方也固定此值，但当前 MI50 的环境白名单没有该项，不能声称它已传入 MI50 容器。显存开关读入后改变终端变量不会影响已启动的计算，必须重新启动该算例。

### 2.3 开始前做什么检查

在现有计算目录中保留原输入、坐标、基组/赝势引用和必要的重启波函数。每个体系使用独立目录，避免多个进程写同一个 PROJECT 输出、重启文件或 trajectory.xyz。MI50 容器只读挂载固定源码、依赖和 data，另外只把当前算例目录作为可写工作区；算例引用的自定义文件尽量放在该目录内。指向目录外的任意绝对路径不会自动出现在容器里。

包装入口要求当前目录属于 nowr、是独立计算目录、路径及祖先不含符号链接。不要在 `/home/nowr`、CP2K 项目根、依赖根或磁盘根运行。`/media/nowr/Game/cp2k-work/system-A` 这类目录层级可作为例子；必须换成真实、已准备好的目录。路径不要含逗号、冒号、分号、反斜杠等包装器拒绝的字符。

先检查安装身份，不启动 GPU：

```bash
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp --check
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp --check
```

两个入口均应出现 `CHECK_OK`。该检查验证已安装文件与实际动态库解析；它不等于 SCF、GPU 或新体系正确性验收。`--version` 会进入实际运行流程，不能当作无 GPU 的替代检查。

检查已有覆盖变量与负载：

```bash
printenv | grep -E '^(OMP_|OPENBLAS_|CP2K_GPU_|CP2K_PW_GPU_|ROCR_|HIP_VISIBLE_)'
free -h
cat /sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used
cat /sys/bus/pci/devices/0000:07:00.0/mem_info_vram_used
```

sysfs 两个数以字节计，是各整卡当时的占用；瞬时空闲不能代替确认长测/模型作业已经结束。若 RX 正运行大模型或 MLIP，不直接照双卡计算命令启动第二个重负载。当前 PATH 下没有 `rocm-smi`；已发现 `/opt/rocm/core-10.1/bin/amd-smi`，具体子命令以该版本 `--help` 为准。

### 2.4 保持科研输入与软件开关分离

原 84 原子回归采用 RKS PBE、D3(BJ)、GPW、OT DIIS、MULTIPOLE / PERIODIC NONE、30 Å 立方盒、5 层网格、REL_CUTOFF 60 Ry、EPS_SCF 1e-7，并分别验证 500 / 320 Ry。它是软件回归基准，不是所有体系的通用输入模板。新体系的边界、真空层、基组、截断能、自旋与溶剂模型必须按科研问题决定。

原输入虽然含 `&MOTION / &GEO_OPT` 段，实际 `RUN_TYPE ENERGY_FORCE`，所以本轮仍是单点能量和力计算，没有执行几何优化。不能因输入里出现 GEO_OPT 参数就把几何轨迹写为已验证。500 Ry 与 320 Ry 之间的差异属于数值设置变化，不能混入“同输入的软件版本误差”。

### 2.5 计算前将 MI50 准备到 high

在实际准备运行 MI50 时加载环境，随后回读：

```bash
source /home/nowr/cp2k_env.sh
use_mi50
cat /sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level
cat /sys/bus/pci/devices/0000:03:00.0/pp_dpm_mclk
```

第一项应显示 high；第二项查看带 `*` 的当前显存频率。此次只读回查为 `1000Mhz *`。这是该次实测，不是仅凭写入 high 就推断成功。若仍是低频或没有升到目标状态，先确认读数与调频执行情况，再开始计时或计算；不凭空指定固定等待秒数。

已加载脚本的同一终端，可在任务开始前再次调用 `lock_mi50_high` 并回读；`use_mi50` 本身只选程序与卡，不调用调频函数。仅用绝对包装入口也不会执行 high 准备，所以 MI50 的 tmux、批次和采样示例均需先完成本步骤。

DPM 是整张 MI50 的设备状态，可能持续到显式恢复或系统重新设置，并非某个 shell 私有变量。现脚本没有任务结束自动恢复 auto 的逻辑；关闭终端不会保证恢复。使用期间保持高性能策略符合本计算流程，但不能由一个作业结束就擅自恢复 auto，影响同卡其他任务。RX 的功耗策略不由此函数改变。

R2 只恢复原样脚本与操作说明，没有新增实现或改写目标配置。R1 新脚本的 13 项测试与独立审查证明的是当时的实现与旧规格一致，不能证明删除 high 符合用户需求；其结果不冒充本版脚本新验收。

## 3. 单张卡运行一个体系

以下在 Bash 中执行；目录和文件名替换为实际值。输出名选未使用的名字。`source` 必须在当前终端执行，`bash cp2k_env.sh` 不能把环境带回父终端。

MI50 示例：

```bash
source /home/nowr/cp2k_env.sh
use_mi50
cd /media/nowr/Game/cp2k-work/system-A
command -v cp2k.psmp
cp2k.psmp --check
OMP_NUM_THREADS=16 cp2k.psmp -i input.inp -o A.out \
  > A.stdout 2> A.err
printf '%s\n' "$?" > A.exit
```

只有 cd 与 --check 成功、显示正确入口后，才继续运行后两行。预期入口为 `build-mi50/bin/cp2k.psmp`。程序启动时先检查文件身份，再启动固定容器，所以屏幕或日志可能短时间没有 SCF 输出；不能只据此判断卡住。结束后 A.exit 应为 0，再检查 A.out 的正常结束、SCF 收敛、能量与力，以及 A.err 的诊断信息。

RX 示例：

```bash
source /home/nowr/cp2k_env.sh
use_9070xt
cd /media/nowr/Game/cp2k-work/system-B
command -v cp2k.psmp
cp2k.psmp --check
OMP_NUM_THREADS=16 cp2k.psmp -i input.inp -o B.out \
  > B.stdout 2> B.err
printf '%s\n' "$?" > B.exit
```

预期入口为 `build-9070xt/bin/cp2k.psmp`。A.stdout / B.stdout 保留包装器或标准输出的补充信息；CP2K 主输出由 `-o` 写入 .out；stderr 单独写 .err。不要只保存 .out 后寻找其中不存在的 `[PW_GPU]` 统计。

在已经完成 §2.5 的 MI50 high 准备后，也可以用绝对入口明确指定程序：

```bash
OMP_NUM_THREADS=16 \
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
  -i input.inp -o output.out > output.stdout 2> output.err
```

这种写法仍继承允许的 CP2K 诊断变量；它并不会自动清除旧预算或排错开关。启动器会提供默认线程、预留量和采样设置，并配置实际库环境，但不会自行请求 MI50 high；不要漏掉前面的准备步骤。

## 4. 两张卡同时计算两个独立体系

### 4.1 正确的任务组织

| 场景 | 当前使用方法 | 证据边界 |
|---|---|---|
| MI50 算 A，RX 算 B | 两个独立进程、各用正式入口、各用独立目录 | 入口与设备隔离已核查；双作业联合实机验收待补 |
| A、B 两条彼此独立扫描链 | 每卡负责一条链；各自按依赖顺序推进 | 不共写重启文件，不并行打乱链内前后依赖 |
| 同一体系由两卡协作 | 当前固定单卡正式包装入口没有该运行模式 | 底层多 MPI/GPU 能力未被删除，但需独立配置和验收 |
| RX 同时跑 CP2K 与 MLIP | 同卡资源共用，必须另测预算与吞吐 | 不等同于 MI50 + RX 各跑一个 CP2K |

两个独立进程都可以叫 cp2k.psmp，两个作业的 MPI rank 也都可以是 0。它们不属于同一个 MPI 通信组，无需 `mpirun -np 2`。不要在正式包装入口外再包 mpirun：MI50 内部已经固定单 rank，RX 为直接单进程；外层多 rank 会改变当前已验收运行契约。

推荐起点是每个作业 8 个 OpenMP 线程、OpenBLAS 固定 1。8 是便于初次并发验证的起点，不是硬限制，也不是已经测得的最优值。单卡 16 线程的耗时不能直接拿来预测双卡并发耗时；主机内存带宽、缓存、I/O 与功耗仍会共享。

### 4.2 使用两个命名 tmux 会话

会话 A：

```bash
tmux new -s cp2k-mi50-A
```

在新会话内先完成 MI50 high 准备，再进入体系 A 的目录并使用绝对入口：

```bash
source /home/nowr/cp2k_env.sh
use_mi50
cat /sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level
cat /sys/bus/pci/devices/0000:03:00.0/pp_dpm_mclk
cd /media/nowr/Game/cp2k-work/system-A
OMP_NUM_THREADS=8 \
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
  -i input.inp -o A.out > A.stdout 2> A.err
printf '%s\n' "$?" > A.exit
```

确认 MI50 的 high 与显存频率读数符合 §2.5、cd 成功后再执行计算。按 Ctrl+B，然后 D 脱离会话，计算继续。回到外层终端后创建会话 B：

```bash
tmux new -s cp2k-rx-B
```

在 B 会话内：

```bash
cd /media/nowr/Game/cp2k-work/system-B
OMP_NUM_THREADS=8 \
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp \
  -i input.inp -o B.out > B.stdout 2> B.err
printf '%s\n' "$?" > B.exit
```

查看或重新进入：

```bash
tmux ls
tmux attach -t cp2k-mi50-A
```

进入 B 时把名字换成 `cp2k-rx-B`。tmux 新窗口可能继承 tmux 服务较早的环境，因此每次命令明确写 OMP_NUM_THREADS 和绝对入口；若需要预算或关闭某项优化，也在该次命令前明确设置，或在对应会话里加载现用环境脚本并检查 printenv；source 会执行 MI50 high 准备。

本节命令是运行方案，没有在本次文档核对中实际启动 A / B。首次并发先用已知短算例，各自与同卡、同线程、相同输入的串行结果比较，确认退出码、收敛、能量/全力及设备归属，再用于长队列。

### 4.3 CPU 绑定的真实限制

当前 lscpu 回查显示，L3 组 0 为 CPU 0–7、16–23，L3 组 1 为 8–15、24–31；它们包含每组 8 个核心及 SMT 兄弟。仅设置 8 线程不意味着自动绑定其中一组。

正式启动器固定 OMP_PROC_BIND=FALSE，并过滤 OMP_PLACES / GOMP_CPU_AFFINITY。MI50 的 Docker 容器由 daemon 创建，外层 `taskset` 绑定 Docker 客户端不能据此证明容器里的 CP2K 也已绑定。当前操作说明不提供“已生效的 CCD 绑定”承诺。若确实需要硬绑定，应给启动器增加明确的容器 cpuset 支持并实际回读进程亲和性；这属于后续独立运行配置变更，本次没有修改启动器。

## 5. 显存预算、预留量与采样

### 5.1 三个不同概念

| 名称 | 实际约束 / 统计范围 | 不能据此保证什么 |
|---|---|---|
| PW 显存预算 | PW 模块显式分配的账本；0 为不设置该预算 | 不是整进程或整卡显存上限 |
| RES_RESERVE_MB | 部分可选驻留分配时对当前可用显存的余量检查 | 不是永久划出的空间，不保证 MLIP 后续得到固定余量 |
| device in use at most / sysfs 最大观测 | 在离散采样时刻看到的整卡占用，包含其他库与其他进程 | 是真实峰值下界，不是安全容量上界 |

预算不会覆盖 Grid、rocFFT 内部工作区、DBCSR、ELPA 等全部分配。可选驻留可能放弃，必需 HIP 分配仍可能失败并中止。不能写“超预算结果必然正确，只是更慢”；某些路径有回退设计，实际输入和预算组合仍须验证。

### 5.2 输入与环境变量的优先级

在已有 `&GLOBAL / &GRID` 内合并设置，不另建重复的 GLOBAL 段：

```text
&GLOBAL
  &GRID
    BACKEND GPU
    MAX_GPU_MEMORY_MB 0
    HOST_FALLBACK .TRUE.
  &END GRID
&END GLOBAL
```

变量名虽然写 MB，源码换算为 1024 × 1024 字节，本文统一称 MiB。预算优先级为：有效的 `CP2K_GPU_MEM_LIMIT_MB_RANK0` → `CP2K_GPU_MEM_LIMIT_MB` → 输入 MAX_GPU_MEMORY_MB。当前正式入口只运行单 rank，只转发 RANK0；源码认识其他 RANK<n> 不代表包装入口允许多 rank。每个独立作业的 rank 都是 0，设置互不相连。

CP2K_GPU_HOST_FALLBACK 的非空值优先于输入 HOST_FALLBACK。预算、预留量应写明确的非负十进制整数；不要依赖非法字符串被忽略的行为。开关推荐只写 0 / 1，不使用空串表达“恢复正常”。空 reserve 会绕过启动器的未设置默认，最终可能回到源码 1536 MiB。

只给 RX 指定一个示例 PW 预算：

```bash
OMP_NUM_THREADS=8 CP2K_GPU_MEM_LIMIT_MB=6144 \
CP2K_GPU_HOST_FALLBACK=1 \
/home/nowr/cp2k-hip/build-9070xt/bin/cp2k.psmp \
  -i input.inp -o B.out > B.stdout 2> B.err
```

6144 MiB 仅演示设置方式，不是本体系已验证预算。若当前 shell 遗留 RANK0 预算，它优先级更高，先清理或显式写同值。MI50 不共享这个限制。单次命令前的赋值只作用于该进程，避免把 RX 预算 export 到公共环境后又误用于 MI50。

取消环境预算：

```bash
unset CP2K_GPU_MEM_LIMIT_MB CP2K_GPU_MEM_LIMIT_MB_RANK0
```

此时输入里的预算仍有效。如需明确覆盖输入为不设 PW 预算，可在单次命令前同时写 `CP2K_GPU_MEM_LIMIT_MB=0 CP2K_GPU_MEM_LIMIT_MB_RANK0=0`。0 不是“预留零显存给系统”，也不关闭所有 GPU。

### 5.3 内部与外部观测

内部采样默认开启，正常结束后在 .err 中输出 device memory 行。设置 `CP2K_PW_GPU_MEM_SAMPLE=0` 关闭整卡采样，但不关闭 GPU，也不关闭 PW 自身账本峰值打印。设置 RES_STATS=0 关闭驻留统计，不等于关闭独立显存打印。

当前目标源码树未发现 `vram_peak.sh`。本次交付包的 `tools/vram_peak.sh` 是从已冻结 r3fix 套件提取的原文件，未改动。把它放在可访问位置，用 Bash 显式执行；不能假定 `./stage7m/vram_peak.sh` 已在机器上。

避免 `pgrep -n -f input.inp`：并发时两个输入可能同名，甚至匹配到自身命令。短任务容易在另开终端之前已经退出。需要外部采样时，先完成 §2.5 的 MI50 high 准备，然后在同一终端启动包装进程并立即用 `$!` 捕获其 PID：

```bash
OMP_NUM_THREADS=16 \
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
  -i input.inp -o sample.out > sample.stdout 2> sample.err &
cp2k_pid=$!
bash /实际路径/vram_peak.sh "$cp2k_pid" 0.05 > sample.vram.txt
wait "$cp2k_pid"
printf '%s\n' "$?" > sample.exit
```

这是需要采样时替代第 3 节前台启动的写法，不要在原任务运行时又启动一次。PID 用于限定采样时段，采样仍统计所有 GPU 的整卡占用；脚本输出文字“峰值”应读作“最大观测值”。开始时占用是采样开始时的值，可能已经包含 CP2K 启动分配，不能自动称作计算前基线。若必须记录真正启动前基线，先单独读取 sysfs，再启动进程。

两张卡并发时，两卡占用都升高是预期行为。单次整卡采样不能将某张卡上的多个进程用量拆开，也不能把两个不同接口的最大值相减推导“其他库峰值”。PID 提前结束、启动失败或脚本未覆盖全部运行时段时，应保留该局限。

## 6. 功能开关完整表

以下为本次核对的定制 PW 开关，不声称穷举 CP2K 全部原生输入。所有变量都必须在启动新进程前设置。默认值指走当前正式包装入口、没有遗留覆盖的情况；“默认开”仅表示允许尝试，仍受方法、布局与显存条件限制。

### 6.1 日常优化与逐项排查

| 环境变量 | 默认 | 作用与关闭后的实际边界 |
|---|---|---|
| CP2K_PW_GPU_RES | 1 | 驻留功能总开关；关后关闭下属驻留功能及其统计，融合 XC 可独立存在 |
| CP2K_PW_GPU_RES_DENSITY | 1 | 密度 RHO_G / RHO_R 驻留；关后不使用该驻留入口 |
| CP2K_PW_GPU_RES_HARTREE | 1 | 驻留 Hartree / DDAPC 设备链；关后对应调用走原有路径 |
| CP2K_PW_GPU_RES_XCIN | 1 | XC 使用驻留密度输入；关后需要常规输入传递 |
| CP2K_PW_GPU_RES_POT | 1 | 势组装及势变换驻留；关后走对应常规路径 |
| CP2K_PW_GPU_RES_LAZY_DENSITY | 1 | 获授权窗口内延迟密度下载；关后不保留该延迟状态 |
| CP2K_PW_GPU_RES_LAZY_POT | 1 | 获授权窗口内延迟 Vxc / Hartree 势下载 |
| CP2K_PW_GPU_RES_NATURAL_G | 1 | 密度 / Hartree 自然 FFT 顺序；关闭会连带禁用下面的自然 XC、配对和 dz |
| CP2K_PW_GPU_RES_NATURAL_XC | 1 | 自然顺序 XC / 势变换；关闭连带禁用配对和 dz |
| CP2K_PW_GPU_RES_XC_PAIRS | 1 | 配对实数 FFT；关闭连带禁用 dz，不承诺恢复全计算逐位一致 |
| CP2K_PW_GPU_RES_XC_DZ | 1 | 密度 FFT 顺带形成 z 梯度，供后续 XC 复用 |
| CP2K_PW_GPU_MAP_CACHE | 1 | 映射与 g 数据缓存；密度/Hartree/势驻留和自然 XC 均另有此开关门槛 |
| CP2K_PW_GPU_FUSED_XC | 1 | 原生 PBE 融合 XC 流程；关闭时回到常规 XC，不等于完全 CPU 计算 |

明确依赖链：RES → NATURAL_G → NATURAL_XC → XC_PAIRS → XC_DZ。其余分支还有布局/缓存/调用条件，不能只根据这个链推断所有功能已激活。

### 6.2 检查与统计

| 环境变量 | 默认 | 作用 |
|---|---|---|
| CP2K_PW_GPU_RES_VERIFY | 0 | 增加驻留数据核对与 dz 重算检查；开启时关闭 lazy 授权，显著改变执行路径与耗时 |
| CP2K_PW_GPU_RES_STATS | 1 | 打印驻留、lazy、natural、槽位统计；受 RES 总开关约束 |
| CP2K_PW_GPU_MEM_SAMPLE | 1 | 整卡离散显存采样；独立于 RES_STATS，关闭后 PW peak 仍可能打印 |
| CP2K_PW_GPU_RES_RESERVE_MB | 1024 | 部分可选分配的显存余量阈值；裸源码默认 1536；不是总预算 |
| CP2K_GPU_MEM_LIMIT_MB | 未设 | 全局 PW 预算覆盖；0 表示不设该预算 |
| CP2K_GPU_MEM_LIMIT_MB_RANK0 | 未设 | 当前单 rank 的 PW 预算覆盖，优先于上一项 |
| CP2K_GPU_HOST_FALLBACK | 未设 | 覆盖输入 HOST_FALLBACK；开启并不覆盖其他库的 OOM |

布尔解析器把未设置或空值当作相应源码默认；以 0、n/N、f/F 开头，以及以 of/OF 等大小写组合开头的值判为关闭，其他非空值判为开启。为避免误输入，实际操作只使用 0 / 1。数值变量不采用这一布尔解析规则。

单次排错示例：

```bash
OMP_NUM_THREADS=16 CP2K_PW_GPU_RES_XC_DZ=0 \
/home/nowr/cp2k-hip/build-mi50/bin/cp2k.psmp \
  -i input.inp -o dz-off.out > dz-off.stdout 2> dz-off.err
```

这一行不改变父终端后续默认值。若用 export，必须主动 unset 对应变量；现用脚本没有 cp2k_defaults。逐次试验使用独立输出名及同一输入/WFN，不让前一次写出的重启文件改变后一次比较条件。

### 6.3 原生输入与编译功能

| 项目 | 控制位置 / 当前状态 | 使用说明 |
|---|---|---|
| Grid 后端 | GLOBAL / GRID / BACKEND；原基准 GPU | 可选 AUTO、REFERENCE、CPU、DGEMM、GPU；改为 CPU 只改变 Grid 后端，不是整个 CP2K 关闭 GPU |
| Grid 对照检查 | GLOBAL / GRID / VALIDATE，默认 FALSE | 对所选 Grid 后端与参考后端做比较；属于额外验证路径，不是整体系全面正确性保证 |
| PW 预算/回退 | GLOBAL / GRID / MAX_GPU_MEMORY_MB、HOST_FALLBACK | 与第 5 节一致，环境覆盖优先 |
| 稠密对角化库 | GLOBAL / PREFERRED_DIAG_LIBRARY | 源码支持 ELPA、ScaLAPACK/SL、CUSOLVER、DLAF 枚举；实际可用性取决于编译，不因枚举出现就支持 NVIDIA 库 |
| ELPA 内核 | GLOBAL / ELPA_KERNEL | wrapper 仅在解析为 AMD_GPU 时启用 AMD GPU；显式 CPU 选择不能被 HIP 构建强行覆盖 |
| ELPA 单阶段 | GLOBAL / ELPA_ONE_STAGE | 原生求解器选择项，可能忽略 ELPA_KERNEL 等选项；不是 Stage7 总开关 |
| 本地 DGEMM | GLOBAL / PREFERRED_DGEMM_LIBRARY | SPLA / BLAS，影响部分本地 DGEMM，不等于替换所有分布式矩阵乘 |
| Cholesky 库 | GLOBAL / PREFERRED_CHOLESKY_LIBRARY | ScaLAPACK/SL 或 DLAF；具体算法应独立验收 |
| 已编译库功能 | ELPA、DFT-D4、SIRIUS、COSMA、DLA-Future、SPLA、Libxc、Libint2、FFTW、MPI 为 ON | 库可链接不等于当前算例调用该库，也不等于所有操作都在 GPU |
| 当前未编译集成 | DEEPMD、LIBTORCH、PLUMED、HDF5 为 OFF | 不能靠环境开关打开；RX 另跑外部 MLIP 与 CP2K 内置 MLIP 集成是两件事 |

DBCSR 是已编译并链接的稀疏矩阵 GPU 后端，原基准保留 GLOBAL / DBCSR / ACC / MIN_FLOP_PROCESS 0；不要为了“全开 GPU”盲目改动科学输入或库选择。ELPA 缓存修复没有新增独立环境总开关。DDAPC 多线程及 Stage7 网格并行随 OMP_NUM_THREADS 工作，设 1 会改变线程使用，但不能把“单线程”称作“恢复原版”。

## 7. 哪些输入能使用哪些优化

| 路径 | 源码条件摘要 | 新用法需要验证什么 |
|---|---|---|
| 融合 PBE XC | 非自旋极化、无 virial、无额外能量网格/权重；仅一个活跃原生 PBE 段；ORIG/REVPBE/PBESOL；PW 导数、无 rho smoothing | 原基准只实测 PBE；其他允许的参数化仍需代表输入验收 |
| Hartree / DDAPC 设备链 | MULTIPOLE 去耦、RKS、非 k 点、非 GAPW；当次不求力、不算 virial、非 just_energy；并排除溶剂、部分约束、SCCS、PCC、PAEP、GCE、QM/MM 等 | 换 Poisson、周期边界或溶剂模型时，核对实际正向调用计数 |
| lazy 延迟回传 | OT、非 g 空间混合、单自旋、无 k 点，并排除 ADMM、meta-GGA 的动能密度、SCCS、外场/外势、嵌入、部分约束和其他主机读者 | 源码具体条件较长，以 qs_scf.F 的 lazy_window 为准，不凭“仍然 OT”就断言全开 |
| 自然顺序 / Grid 直通 | 单 rank 本地全盒网格、布局一致、对应 GPU buffer 可用；具体分支还受缓存开关约束 | 多 rank、球形/分布式网格及不同布局要重新核对 |
| dz 复用 | 自然 XC 与配对 FFT 开启，密度驻留和 g 数据/代次符合当前调用 | formed / used 与当前输入的调用序列一致，不套用固定公式 |
| Stage7 网格建立 | 对相应网格准备工作并行；受线程和数据规模分支影响 | 与计算方法的其他 GPU 分支分开判断，方法变更不等于此优化消失 |

求力或 virial 构建会先使驻留数据失效，且求力时不会进入上述 Hartree 设备链；这不表示该次 KS 的所有优化全部关闭。原 500 Ry 日志中 natural XC=10，而驻留 Hartree=9，已经说明“最后求力一次一律完全原版”的说法过宽。

UKS、其他泛函、杂化、ADMM、slab、应力、晶胞优化、几何优化和 MD 都不应概括为“一定全回退、一定不提速且结果保证正确”。应分别判断每条优化的资格、实际激活以及数值验收。设计上有回退分支不等于已测试所有组合。

## 8. 如何看输出与参考数据

### 8.1 正式化后四组真实回归

以下来自已结束的正式入口回归，每卡独立运行、16 线程、无 profiler、每组一次。对照是同卡此前已经验收的 Stage7，目的是验证最终路径和部署一致；不是新增优化速度比较。

| 卡 / cutoff | CP2K 总计时 / s | SCF 总计时 / s | SCF 步数 | pw_grid_setup / s |
|---|---:|---:|---:|---:|
| MI50 / 500 Ry | 12.582 | 4.317 | 9 | 1.784 |
| MI50 / 320 Ry | 8.870 | 3.565 | 12 | 0.911 |
| RX / 500 Ry | 16.299 | 7.487 | 9 | 1.798 |
| RX / 320 Ry | 13.136 | 7.184 | 12 | 0.909 |

四组实际退出码均为 0；同卡能量、全部 84 原子共 252 个力分量及 SCF 步数通过预先设定的门槛：能量绝对差 ≤1e-10 Ha，最大力分量差 ≤1e-9 Ha/bohr，步数相同。

| 卡 / cutoff | 对此前同卡 Stage7 的能量绝对差 / Ha | 最大力分量差 / Ha/bohr |
|---|---:|---:|
| MI50 / 500 Ry | 1.70e-13 | 2.3911e-12 |
| MI50 / 320 Ry | 2.27e-13 | 2.599e-13 |
| RX / 500 Ry | 1.14e-13 | 7.178e-13 |
| RX / 320 Ry | 2.85e-13 | 2.760e-12 |

| 卡 / cutoff | PW 账本峰值 / MiB | HIP 整卡最大观测 / MiB | 设备报告总量 / MiB |
|---|---:|---:|---:|
| MI50 / 500 Ry | 9003 | 12010 | 16368 |
| MI50 / 320 Ry | 4611 | 7010 | 16368 |
| RX / 500 Ry | 9003 | 13306 | 16304 |
| RX / 320 Ry | 4611 | 8716 | 16304 |

原稿 12.546 / 8.779 s 与 16.142 / 13.025 s 属于迁移前 Stage7 测量，不能冒充本次正式入口回归。原稿历史折线图已随审查原件保存，本操作手册用上表表达当前部署数据；没有把不同阶段、频率或缓存条件拼成新的加速结论。

RX 320 Ry 正式回归 stderr 有 CPU time base 探测告警并使用默认值；数值门槛通过，该告警仍是计时证据的限制。已有 DBCSR untuned kernel 告警也保留。计时表取 CP2K 内部 TOTAL TIME / MAXIMUM，不等于含容器启动、哈希和链接检查在内的终端总耗时。父子 timer 不得相加；短例程因打印阈值未出现不等于耗时为零或失败。

### 8.2 500 Ry 的实际诊断参考

以下仅对应原 84 原子、500 Ry、9 次 SCF 内迭代的完整运行：

| 字段 | 真实参考值 | 判读 |
|---|---|---|
| residency | density 10；hartree 9；xc-input 9；vtot 10；potential 10；fallback 均 0 | 正向调用与 fallback 一起看，0 fallback 本身不证明激活 |
| lazy | density 8；vxc 9；v_h 9；discarded 10；host syncs 1 | 新方法会改变主机读取和调用序列，不照搬固定次数 |
| natural order | xc 10；axis tables；paired FFTs 10；potential 10；rs levels 50 | 当前 5 层网格下成立；不是所有输入必须 50 |
| dz | formed 9；used 8 | formed 与 used 不同可由调用生命周期导致，不能单凭差值定为缺陷 |
| slots hits/misses | RHO_G 18/1；RHO_R 29/1；VXC 20/0；V_H_R 9/1；RHO_CORE 8/1；INFL 8/1 | 18/1 是 18 次命中与 1 次未命中，不是 100% 命中 |
| device memory | MI50：PW peak 9003；device 12010 / 16368；PW at sample 9003 | 括号 PW 是整卡最大观测那个时刻的账本值 |
| device memory | RX：PW peak 9003；device 13306 / 16304；PW at sample 7483 | 账本峰值与整卡最大观测不一定同一时刻 |

320 Ry 的 density/Hartree/natural XC 为 13/12/13，dz formed/used 为 12/11。上述计数是基准实例，不是换体系后的固定验收阈值。没有可靠原版双卡对照时，跨卡差值只报告，不套用同卡门槛判定通过。

## 9. 验证边界与新体系验收

已经完成的证据包括：MI50 最终路径完整构建、两卡架构与依赖检查、正式入口四组原输入数值回归、环境/包装工具测试，以及既有独立源码审查。当前四组是 ENERGY_FORCE 单点，不能推广为多构型、所有泛函和任意显存预算已经验证。

| 使用场景 | 使用前最小验证 | 当前状态 |
|---|---|---|
| 相同方法的新分子/离子簇 | 先完成一次收敛单点；核对能量、全力、诊断与显存；必要时用可信基线比较 | 依新体系补做 |
| GEO_OPT / MD / REFTRAJ | 短轨迹逐步比较能量、力和结构，检查多构型缓存失效 | 未执行真实轨迹验收 |
| UKS / 其他泛函 / slab / 应力 | 核对每条路径资格，并以该方法的可信结果比较 | 未覆盖 |
| 双卡 A+B 并发 | 各自与同卡同线程串行参考比较，并记录双卡显存与整体吞吐 | 未做联合实机验收 |
| RX + MLIP | 测两者各自正确性、并发峰值观测与失败处理 | 未覆盖；PW 预算不构成共存保证 |
| 更大体系 / 更大盒子 / 更高 cutoff | 用真实输入测容量，保留科学设置与输入哈希 | 84 原子结果不能外推任意 200 原子体系 |
| ELPA DIAG / DFT-D4 / 多 MPI | 分别运行真实对应路径并比对 | 库已保留，但相应计算验收未完成 |

旧同卡构建不再在目标机，不能要求日常使用者运行不存在的目录。原基准可直接使用已冻结的旧输出比较。新体系需要原版基线时，应安排隔离参考环境或按已归档材料受控恢复；在此前提未具备时，关闭优化开关只能用于诊断，不能冒充独立原版对照。

离线套件包括 pw_gpu_residency、pw_grid_setup、elpa_gpu 和独立 pw_gpu_integrated。最后一个仍有两个已知 mock 缺符号链接问题：`pw_gpu_launch_res_natural_extract_prefix`、`pw_gpu_launch_res_natural_cdot_terms`。这不是已观察到的生产数值失败，但确实留下测试覆盖缺口，不能写“全部测试通过”或“绝不影响计算”。工具测试与源码套件也不能相互替代。

自检模式可用 CP2K_PW_GPU_RES_VERIFY=1，但会关闭 lazy，改变被执行路径；它通过不能代替默认生产路径验收。检查 dz verify mismatches 为 0，同时保留退出码与完整错误信息，不以这一计数代替全部能量/力比较。

## 10. 常见故障、排查与回滚

| 现象 | 优先检查 | 处理边界 |
|---|---|---|
| CHECK_OK 不通过 | 当前包装入口、已安装 ELF+libcp2k、依赖路径、源码身份是否变化 | 保留错误原文；不改清单强行绕过检查 |
| 架构错误 / invalid device / no binary for GPU | 是否用错 install ELF、混设 HIP_VISIBLE_DEVICES、绕过固定入口 | 回到正式包装入口；不是只改一个 UUID |
| owned directory / symlink / protected root 错误 | 当前目录、祖先链接、文件所有者 | 移到已有合法专用目录，不扩大容器挂载到整个 home |
| 文件在主机存在，MI50 容器找不到 | 输入引用是否超出当前算例目录及固定 data/依赖挂载 | 将算例专用文件准备在该目录内，保持输入来源记录 |
| hipMalloc / out of memory | 整卡占用、其他进程、盒子/网格与全部库分配 | PW 预算不能防止全部 OOM；不为迁就显存擅自改科学设置 |
| Memory limit reached | PW 预算与 RANK0 覆盖、HOST_FALLBACK | 确认目标卡余量与已验证配置；回退后仍需看数值 |
| 无优化统计 / 正向计数为零 | RES_STATS、RES、方法门槛和程序是否正常结束 | 不只凭 fallback=0 判断成功 |
| 能量或 SCF 步数变化 | 输入/WFN、同卡版本、线程、参数、收敛状态及优化路径 | 先确认比较对象一致；不是看到差异就确定某个 kernel 有 bug |
| 运行变慢 | 并发负载、DPM、温度、线程、缓存与初始化时间 | 单次时间不用于强归因；不能凭 timer 缺失推断为零 |

逐项排错可按 dz → pairs → natural XC → natural G → RES → fused XC / map cache 的顺序累积关闭，每次保持相同输入与独立输出。某一步恢复只能缩小到该路径及其依赖/交互范围，不能据此断言“错误就在刚关闭的唯一开关”。关闭这些开关仍保留 Stage7 的网格变更、DDAPC 并行及其他修改，因此不是逐版本回滚。

回滚材料目录：`D:/hermes/workspace/projects/cp2k-gpu-fft-mi50/releases/stage7-final-rollback/`。历史二进制有绝对路径依赖，需要按归档原位置恢复一整套匹配的程序、库和环境。不可把旧文件随意改名丢进现有安装，或从已删除的候选路径直接运行。本次没有恢复旧构建，也不增加第三套常驻构建。

## 11. GitHub 发布状态与冻结材料

本次通过 GitHub API 只读回查了分支、tag、所有 release、latest 和最终提交。结论：**并非所有版本都已上传；当前实际使用的 Stage7 两卡最终版尚未发布到 GitHub。**

| 内容 | GitHub 状态 | 本地状态 |
|---|---|---|
| Stage6 初版累计补丁/套件 | 已公开，v2026.2-mi50-stage6 | 历史版本 |
| Stage6 r3 完整源码 | 已公开，v2026.2-mi50-stage6-r3，仍为 latest [1] | 有冻结包与历史验收 |
| ELPA wrapper r1 | 已公开，v2026.2-elpa-wrapper-r1，标记 prerelease [2] | 独立源码组件，不是完整 Stage7 |
| GitHub 默认分支 | master，5a4200781da3f8c3b05a4f6b28f3ed9ca2659d98 [3] | 与目标机本地 main 不同 |
| Stage7 + 显存观测 + 最终两卡入口 | 无对应公开 tag/release；最终 main 提交查询返回未找到 [3][4] | 已部署并冻结于 Windows |
| 本次 R2 说明与现用环境脚本原样副本 | 未上传 | 恢复计算前 high；目标正式配置未被修改 |

历史 r3 与 ELPA 发布授权不扩展为本次 GitHub 写入。本次问题是核实状态，因此只读检查，不自动 push、开 PR、改 latest 或上传文件。

Stage7 冻结目录为 `D:/hermes/workspace/projects/cp2k-gpu-fft-mi50/releases/stage7-final/`：

| 文件 | 用途 |
|---|---|
| cp2k-2026.2-stage7-main-full-source.tar.gz | 最终 main 的完整源码，9053 个条目 |
| cp2k-2026.2-to-stage7-final.patch | 相对 CP2K 2026.2 基线的完整补丁 |
| cp2k-stage7-main.bundle | 本地 Git 历史与最终提交材料 |
| final-binaries.tar.gz | 两卡最终 ELF/lib 身份与归档；不是可迁移通用离线安装器 |
| deliver_stage7m_20261106-r3fix.tar.gz | 历史准备套件与采样脚本来源，不单独代表最终运行适配层 |
| FREEZE_MANIFEST.json / FINAL_ACCEPTANCE.json | 完整哈希、构建身份与正式回归结果 |

重建以最终完整源码、依赖配方与实际路径为依据。原稿从已删除的 `build-mi50-stage7/source` / `build-rocm101` 读取配置的步骤已经不可执行；也不能把 r3fix 的 70 文件清单当作最终整个项目清单。正式 MI50 build helper 要求空构建目录并拒绝覆盖已有完成状态，不应在现有正式构建上直接重复调用。依赖升级或迁移另开维护任务，先归档再 fresh configure，并核对架构、功能、链接和真实数值。

## 12. 证据定位与身份核对

### 12.1 正式版本的关键哈希

| 对象 | SHA256 |
|---|---|
| MI50 实际 ELF | f11dbeb544b419d1a0f2e46171b5a32b30c13850f87699d75c8bd071312e93bf |
| MI50 libcp2k.so.2026.2 | fb02ef9bba7a9fc2a8c091d1fc5add00d68bad78dfacdc3392292e7b049a710c |
| RX 实际 ELF | 103f158e072921b259091b11e973f60e18d2403546d22c205f048dca845f7a97 |
| RX libcp2k.so.2026.2 | 82978ba0b5d9bfe8cce994d5c8ad13947008a06b3961a04e79b6a298b3e2570d |
| Stage7 完整源码包 | 1f33bc3b92b4129654c019f64d163ff6f0105ef3262d3ddd429047d41beadc5a |
| Stage7 完整补丁 | 1ef8fa093c5409e420e02444b01a2cd55ad8ba248ec8c569467b8623bc8004ad |

CP2K 是动态库布局；主 ELF 与 libcp2k 必须一起识别。MI50 同路径重建后主 ELF 可以与旧版相同，但计算库已变化。原稿 `3377b006...` 是已退役候选 ELF，不能继续标为正式版本身份。程序内部版本串也不能单独辨别定制阶段。

### 12.2 本次资料的证据等级

知识库证据：RAGFlow 主入口实际检索连接超时；备用入口数据集枚举和 retrieval 均返回 HTTP 502。本次没有取得可引用的知识库内容，“知识库中未找到您要的答案！”在这里仅说明未获得答案，不表示库内资料被删除。

本地及目标机直接证据：正式化 FINAL_ACCEPTANCE.json、原始 .out/.err、当前 target-live.json、target-details.json、feature-cache.json、逐字节脚本比对和 frozen source。来源是代码与执行记录，不是文献推断。

外部补充：GitHub API 分支、tags、releases、latest 与提交查询，原始响应存于 evidence/github-*.json；只用于发布状态。

运行建议与未验证部分：双卡各 8 线程、首次并发检查和新体系验收流程属于操作建议；没有伪称本轮已经跑过并发或新科学路径。

源码定位以最终 main 为锚：runtime/stage7.py 的 environment、docker、execute；src/pw/gpu/pw_gpu_internal.c 的 env_flag、read_memory_environment、pw_gpu_init、mem_print；src/input_cp2k_global.F 的 GRID / ELPA / library 输入定义；src/qs_scf.F 的 lazy_window；src/qs_ks_methods.F 的失效与 Hartree 门槛；src/xc/xc.F 的 xc_vram_pbe_parameters。详细行号与纠错对应关系见随附纠错清单。

## Sources

[1] https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-mi50-stage6-r3
[2] https://github.com/lunncing/cp2k-rocm-mi50/releases/tag/v2026.2-elpa-wrapper-r1
[3] https://api.github.com/repos/lunncing/cp2k-rocm-mi50/branches
[4] https://api.github.com/repos/lunncing/cp2k-rocm-mi50/releases
