# Stage 6 r3 验收记录

归档：`cp2k-2026.2-mi50-stage6-source-20261106-r3.tar.gz`  
大小：96,016,581 bytes  
SHA-256：`b15932fbb996d9f9f2cb3893a315432cb72ba2a0f7480464eab5da7d1a898c9a`  
验收日期：2026-11-06；目标 Ubuntu x86-64、Ryzen 9 9950X、MI50/gfx906、既有 ROCm 容器环境。

完整包含 9,041 个 CP2K 普通源文件；其中 1,555 个 `src/` 文件全部与冻结目标源码逐字节一致，源文件权限保留。唯一源码树内变化为 `tests/elpa_gpu/test_elpa.F` 的测试预期修正，原始/替换 SHA-256 与补丁随包记录。完整包不含私有研究输入、波函数、内部审查原文或凭据。外部 Docker 镜像和依赖库不在包内。

## 完整性与构建

| 项目 | 结果 |
|---|---|
| 两次生成归档 | SHA-256 相同，逐字节一致 |
| Ubuntu 全新解包校验 | 目录、成员哈希、权限、shell 语法通过 |
| 包装/启动器测试 | 24 项通过 |
| r3 全新安装前缀构建 | 4,168 个 Ninja action 完成，exit 0 |
| 动态链接 | 无 `not found`；libcp2k 指向 r3 安装 |
| 无 GPU `--check` | `CHECK_OK`，仅校验链接和二进制/库哈希 |
| MI50 正常启动器 `--version` | exit 0，CP2K 2026.2，含 offload_hip、ELPA、DBCSR、COSMA、SIRIUS、DLAF 等构建标志 |
| 独立打包审查 | r2 16 文件完整读取；r3 六文件增量复审 PASS，读取覆盖与哈希已核验。范围仅打包脚本及一处测试修正 |

r3 修正了 r2 的检查模式：该 CP2K 在处理 `--version` 前会初始化 GPU，故无 GPU 检查不再调用 CP2K；查看版本走正常 MI50 入口。源代码内的 `REVISION=git:c92cc08` 只是原快照标签，不作为完整 Git 源码身份。归档摘要与 manifest 才是本次交付标识。

## 测试范围

PW residency（含 C、生产内核 CPU 仿真、Fortran 绑定和 Grid dispatch）及修正后的 ELPA mock 在目标机通过。ELPA 覆盖 CPU/HIP/CUDA/OpenCL 四种配置、各八种能力组合及无 ELPA 编译；这是 mock 证据，不是真实 GPU 对角化。

另行执行的旧 `tests/pw_gpu_integrated/run_tests.py --sanitize` **在链接阶段失败**：mock 缺少 `pw_gpu_launch_res_natural_extract_prefix` 和 `pw_gpu_launch_res_natural_cdot_terms`。此入口原样保留，未删除断言、跳过配置或修改生产代码来掩盖失败。包内 `--mock all` 明确只运行 PW residency 和 ELPA 两套测试，不能据此声称全部 CP2K 测试通过。

## MI50 实际运行

采用原样复制的 84 原子、闭壳层 PBE、正交晶胞、MULTIPOLE/DDAPC、OT DIIS 输入，两种截断能均收敛。每个算例独立启动一个 CP2K 进程，单 MPI rank、OpenMP 16 线程。首轮与暖缓存复跑均完成，所有已报告 fallback 计数为 0。

| 截断能 | SCF 内循环次数 | 暖缓存内循环总时 / 每次 | 暖缓存总墙钟 | 首轮总墙钟 |
|---|---:|---:|---:|---:|
| 500 Ry | 9 | 4.323 s / 0.480 s | 14.200 s | 23.049 s |
| 320 Ry | 12 | 3.543 s / 0.295 s | 9.636 s | 12.505 s |

每次耗时由 `scf_env_do_scf_inner_loop` 总计时除以调用数计算，不是每个完整 SCF 步的所有开销。初次进程的内核/库初始化会增加墙钟，持久缓存已跨进程保留；以上为本次单次记录，不是统计分布或所有输入的性能承诺。

相对此前保留的 Stage 6 输出，暖缓存复跑的全原子比较如下：

| 截断能 | 总能量绝对差 / Ha | 最大力分量绝对差 / Ha·bohr⁻¹ |
|---|---:|---:|
| 500 Ry | 1.14×10⁻¹³ | 5.9052×10⁻¹² |
| 320 Ry | 3.98×10⁻¹³ | 8.444×10⁻¹³ |

密度 FFT 形成/被 XC 使用的 z 导数计数分别为 9/8 和 12/11，与保留记录一致。上述数值不作逐位一致声明。

## 使用边界

专用启动器固定绑定 MI50，单 MPI rank，不跟随旧 `use_9070xt` 函数切卡。MI50 没有默认 8 GiB 上限；当前 PW 显存预算不代表所有库及外部 MLIP 的全进程预算。多卡源码能力仍在，未作为本入口的已验证能力。

真实 ELPA 对角化与 ScaLAPACK 对照、多 rank、同进程多构型 GEO_OPT/MD、UKS、非正交晶胞、其它泛函尚未全面验收；完整累积 Stage 6 物理源码独立审查亦未闭合。独立打包审查与数值运行的范围分别记录，不把其中任何一项扩展成全模式保证。
