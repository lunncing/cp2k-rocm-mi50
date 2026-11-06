# CP2K 2026.2 MI50 Stage 6 源码包

这是 2026-11-06 从目标机安装环境取得的 Stage 6 完整源码实验快照，附带独立构建、运行及校验脚本。`source/` 含 CMake、src、data、tests、tools 和上游公开 benchmarks，共 9,041 个普通文件。全部物理源码字节与全部文件权限保留；唯一例外是下述明确记录的 ELPA 测试预期修复。不附带个人计算输入、波函数、轨迹或旧构建目录。

`source/REVISION` 的 `git:c92cc08` 只是快照自带标签，不能证明完整源码对应某个干净上游提交。本包不依赖旧的部分覆盖仓库，也不附带累积补丁。来源与许可证见 [NOTICE](NOTICE)。

## 适用环境与保留项

本配方用于既有 Ubuntu x86-64 / AMD MI50（gfx906）环境。它是完整 CP2K 源码包，依赖现有 Docker 镜像及已安装库，不能作为包含全部依赖的离线系统安装器。主机需要 Bash、Python 3.11+、Docker 及当前用户的 Docker 权限；容器中的编译器、CMake、Ninja、MPI、ROCm 等由既有镜像提供。

以下项目必须保留，不能在“清理旧 CP2K”时一起删除：

- 本地镜像 `sha256:eb883e666c1d9787ce3696c50c2256c45f6c4a5107dc6af66c2a781cda7f4fd4`。
- `/home/nowr/cp2k-mi50-20261103`：DBCSR、ELPA、SIRIUS 等依赖安装与 pkgconfig 配置。
- `/home/nowr/spack/opt/spack`：OpenBLAS、FFTW、LibXC、ScaLAPACK 等精确依赖版本。
- `/home/nowr/MI50_CP2K_full_build_20261103`：原构建配方，容器只读挂载为 `/recipe`。
- `/home/nowr/cp2k-hip/data`、`/opt/rocm/core-10.1`、现用 `/home/nowr/cp2k-mi50-production` 和现有 `cp2k_env.sh`。

运行默认使用包内完整 `source/data`，同时只读挂载现有 `/home/nowr/cp2k-hip/data`，兼容输入中的绝对数据文件名。ROCm 路径指容器内既有安装；不下载 CUDA、不拉取镜像或源码。具体依赖路径与 63 个原构建参数见 [recipe/dependencies.json](recipe/dependencies.json)。移植到不同依赖布局需要重新适配和验收。

## 解包、校验、重建

在包所在目录执行；将归档名保持为下述名称：

```bash
archive=cp2k-2026.2-mi50-stage6-source-20261106-r3.tar.gz
sha256sum -c "$archive.sha256"
tar -xzf "$archive"
package="$PWD/${archive%.tar.gz}"
bash "$package/scripts/verify.sh" --archive "$PWD/$archive" --launcher-tests
bash "$package/scripts/build.sh" --prefix "$HOME/cp2k-mi50-stage6-release-r3" --jobs 16
```

先校验可信渠道给出的归档摘要，再解包。内置 `manifest.json` 记录路径、类型、权限、长度与 SHA-256；`SHA256SUMS` 覆盖全部普通载荷文件及 manifest，自身由归档摘要覆盖。`verify.sh` 对照归档和解包文件、检查必需路径及辅助脚本 `bash -n`。Linux 解包权限也必须一致。

`--prefix` 默认 `$HOME/cp2k-mi50-stage6-release`；目标机该默认目录已用于 r2，因此本文 r3 示例明确使用新的 `/home/nowr/cp2k-mi50-stage6-release-r3`。prefix 必须是绝对、独立、新建或空目录，父目录须已存在。空目录必须属于当前 UID。路径可以含空格，但不支持符号链接、`..`、逗号、冒号、分号、反斜杠或控制字符。旧生产版、依赖目录、源码包目录以及它们的父子目录均不能用作 prefix。

每次重建使用新的 prefix，例如 `--prefix "$HOME/cp2k-mi50-stage6-rebuild-02"`。失败构建也不会被自动清空或复用，不复制旧 CMakeCache。日志位于 `<prefix>/logs`，安装位于 `<prefix>/install/cp2k`。只有编译、安装和完整依赖解析通过后才生成 `.cp2k-stage6-build-ok.json`；出现任何 `not found` 或加载到旧 `libcp2k` 都会失败。构建不挂载 GPU，不运行 SCF，不修改全局配置。

成功构建还会生成 `<prefix>/bin/cp2k.psmp` 启动器，它固定绑定对应的包路径与 prefix，并将任意引号参数和当前目录交给 `scripts/run.sh`。这是 MI50 单卡、单 MPI rank 入口，不响应 `use_9070xt`，不能充当双卡通用启动器。源码的多 GPU 能力仍保留，原其它 GPU 启动路径也应保留。需要接入现有环境配置时，由操作者先备份、完成目标验收，再把其 PATH 中的程序目录指向此 `bin`；打包和构建脚本不编辑现有环境文件。

## 运行与 A/B 检查

先进入属于当前 UID 的独立计算目录，使用自己的输入及其相对引用文件。仅当前工作目录以原路径可写挂载，包、安装和依赖只读挂载。允许用户在 `/media/<用户>/<卷>/<工作区>/<具体算例>` 或更深的算例目录主动运行；拒绝挂载 `/media`、卷根目录、工作区根目录、整个 home、包、prefix 或依赖目录。既有输入绝对引用 `/home/nowr/cp2k-hip/data/BASIS_MOLOPT`、`GTH_POTENTIALS`、`dftd3.dat` 时，专用只读兼容挂载保持其有效，不需要修改研究输入字节。其它宿主绝对路径不会自动暴露。

`HOME=<prefix>/container-home` 和 `XDG_CACHE_HOME=<prefix>/cache` 在不同计算进程间保持，保存内核及 DBCSR 的热缓存；这两个专用目录单独可写挂载，拒绝符号链接。

```bash
mkdir -p "$HOME/cp2k-stage6-jobs/example"
cd "$HOME/cp2k-stage6-jobs/example"
# package 为上一步的绝对包目录；先放入自己的 input.inp 等文件。
bash "$package/scripts/run.sh" --prefix "$HOME/cp2k-mi50-stage6-release-r3" --check
bash "$package/scripts/run.sh" --prefix "$HOME/cp2k-mi50-stage6-release-r3" -- -i "input.inp" -o "run.out"
# 等价的已构建入口：
"$HOME/cp2k-mi50-stage6-release-r3/bin/cp2k.psmp" -i "input.inp" -o "another.out"
CP2K_PW_GPU_RES_XC_DZ=0 bash "$package/scripts/run.sh" --prefix "$HOME/cp2k-mi50-stage6-release-r3" -- -i "input.inp" -o "ab.out"
```

`scripts/run.sh --check` 只检查 ldd 及已安装可执行文件/libcp2k 的构建时摘要，成功打印 `CHECK_OK`；不调用 CP2K 或 mpirun，不挂载 GPU。它可从包目录或 home 执行，工作目录设为容器 `/tmp`，不会挂载宿主当前目录。此快照的 `--version` 也会初始化 GPU，查看版本须在专用 cwd 使用正常的 MI50 启动器，例如 `<prefix>/bin/cp2k.psmp --version`。正常运行使用调用用户 UID:GID、`mpirun -np 1 --nooversubscribe --bind-to none`，保留参数及退出码。默认 OpenMP 16 线程、OpenBLAS 1 线程、OMP 栈 256M。

启动器读取 sysfs，必须同时确认 UUID `GPU-37ac3981732c7340`、AMD `1002:66a1` 和 render 节点的设备号；当前已知 PCI 地址为 `0000:03:00.0`，不硬编码 `renderD128`。身份无法验证时拒绝运行。仅挂载该 render 节点与 `/dev/kfd`，以 `ROCR_VISIBLE_DEVICES` 绑定 MI50。传给 CP2K 的环境清除 `HIP_VISIBLE_DEVICES`、GPU 序号覆盖和 `HSA_OVERRIDE_GFX_VERSION`；不访问 RX9070XT、不设置时钟、不调用原环境脚本。

只转发明列的线程变量和当前源码支持的诊断变量，完整列表见 `scripts/launcher.py` 中的 `THREAD_DEFAULTS`、`DIAGNOSTICS`。支持 residency、natural FFT、paired XC、XC_DZ、verify/stats、host fallback 及 `CP2K_GPU_MEM_LIMIT_MB` / `CP2K_GPU_MEM_LIMIT_MB_RANK0`；不会按变量名前缀透传宿主环境或密钥。

MI50 没有默认 8 GiB 人工上限；仅沿用 `CP2K_PW_GPU_RES_RESERVE_MB=1024` 的 PW 预留设置。当前 `CP2K_GPU_MEM_LIMIT_MB` 和 `_RANKn` 预算只约束 PW 分配，不约束 DBCSR、ELPA、其它库或外部 MLIP。本包不是按 GPU UUID 实施的全进程显存预算器。RX9070XT 的可调预留/按 rank 策略仍未实现，需要另外开发和验收。

## 验证范围

协调方已从冻结的 r2 完整归档在目标机全新构建成功（4,168 个 Ninja action），并确认新安装的 libcp2k 解析正确。r3 保持相同源码及测试覆盖，只修正无需 GPU 的包装检查。本文记录打包时的作者本地检查和已收到的 r2 证据；发布时的目标重建、运行和审查结果由协调方以与最终归档 SHA-256 对应的独立 `VALIDATION` 记录提供，应按摘要查阅。旧目录内复制过的 benchmark 输出不作为新归档的运行证据。打包校验和启动器夹具测试不等于真实 GPU 验收；打包时累积独立物理代码审查尚未完成，后续状态以该验证记录为准。

本地 PW residency 完整 mock 已通过。冻结快照中的 ELPA 测试曾在 HIP / support mask=1 因过时的“CPU kernel 也启用 GPU”预期失败。针对该失败的独立有限审查确认应修正测试预期；本包只覆盖 `tests/elpa_gpu/test_elpa.F`，删除过时预期并增加显式 CPU 和 AUTO 解析断言。修正后的 CPU/HIP/CUDA/OpenCL 四配置 × 八种能力组合以及无 ELPA 编译均已在 WSL 通过。没有修改生产 ELPA wrapper、物理源码、测试容差或跳过配置，也没有新增实测 ELPA 加速结论。

唯一测试覆盖的原始/替换 SHA-256 和原因在 manifest 的 `source_overrides` 中，公开差异见 [patches/elpa-test-expectation.patch](patches/elpa-test-expectation.patch)。`overrides/` 保存该修正的完整测试文件，以便重现打包。

可选 `--mock all` 只选择 PW residency 和 ELPA 两套测试，不表示全部 CP2K 测试。主机须预先安装 Python 的 fypp 模块、gcc、g++、gfortran；缺少工具即失败，不自动下载：

```bash
bash "$package/scripts/verify.sh" --mock all  # 或 --mock pw / --mock elpa
```

另一套独立的旧 `tests/pw_gpu_integrated/run_tests.py --sanitize` 已知在链接阶段失败：其 mock 缺少 `pw_gpu_launch_res_natural_extract_prefix` 和 `pw_gpu_launch_res_natural_cdot_terms`。本包原样保留该套件，不修补、不跳过或把其失败算作通过，因此不能声称“全部测试通过”。

测试在临时源码副本中运行；任一套失败即返回非零，不能将 mock 通过宣称为真实 ELPA 数值验证。ELPA 包含 handle cache 和 respect-kernel 修正，但尚无真实 diagonalization 对照 ScaLAPACK 的验收。实数场成对打包使用 C2C FFT，并非原生 R2C/C2R。

已有性能记录仅适用于单构象的 84 原子 PBE / RKS / MULTIPOLE：`scf` 内层计时按调用数平均为 `4.282/9 ≈ 0.476 s` 与 `3.522/12 ≈ 0.294 s`。它们不是首次完整迭代的耗时，也不能推导全部 GEO_OPT、MD、UKS 或非正交输入都已验证。

## 回退与清理

构建和运行都不接管旧生产目录、全局 PATH 或 `cp2k_env.sh`。保留旧版及其原启动方式；新版本验收失败时停止该计算，使用旧版经过验证的启动命令和单独输出目录。不要把新输出覆盖到旧计算结果上。包与 prefix 使用绝对路径，安装后不宜移动；保留每个 prefix 对应的源码包。

默认不做自动清理。若需要释放空间，先确认没有任务使用该失败 prefix，核对其中 `.cp2k-stage6-package.json` 的绝对路径和所属 UID，并逐项检查没有个人文件，再由操作者只清理该明确目录。不要递归清理 `/home/nowr`、通配符匹配的旧 CP2K 目录或上述依赖保留项；磁盘不足时先停止重建。旧生产版应在新包重建和实际验收完成、回退需求结束后另行处理。
