# 构建配方

`dependencies.json` 从目标机成功的 `build_inside_container.sh` 提取，保留原有 63 个 CMake 参数名与功能开关。仅把旧安装前缀替换为 `{prefix}/install/cp2k`，并用库路径列表生成 CMake prefix/RPATH。新安装的 lib/lib64 排在动态库搜索路径首位；启动前必须核对 `libcp2k` 的完整解析路径和构建时摘要。

`prefix_paths`、`pkg_config_paths`、`library_paths` 保存目标依赖的精确版本路径，没有从宿主继承 `LD_LIBRARY_PATH`。`rocm` 指容器内 `/opt/rocm/core-10.1`。`mounts` 和运行专用 `runtime_mounts` 是明确列出的外部只读依赖。默认 data 使用包内 `source/data`；同时保留 `/home/nowr/cp2k-hip/data` 的只读挂载及存在性检查，兼容原输入的绝对数据路径，不挂载整个 cp2k-hip 目录。原脚本中固定容器名、固定 render 节点、硬编码研究目录、旧构建和 benchmark 复制操作均不沿用。正常运行仅挂载用户主动选定的算例 cwd，以及 prefix 内独立的持久 cache/container-home；检查模式不挂载宿主 cwd。

入口 `build_inside_container.sh` 调用包内 Python 启动器，显式启用 FetchContent disconnected，并在 `--network=none --pull=never` 的固定镜像内编译。图形设备仅用于正常运行；编译和 `--check` 都不挂载设备。不使用 `--privileged`，不修改时钟或机器配置。安装目录固定为 `<prefix>/install/cp2k`，日志位于 `<prefix>/logs`。

发布维护者可在包装仓库运行：

```bash
python3 -B scripts/package.py --source-tar /absolute/path/source-stage6-target.tar.gz
```

这会重建 `dist/` 中固定名称的一个 tar.gz 及其 `.sha256` 摘要。输入必须匹配已锁定的原始摘要；使用明确的辅助文件白名单，源码从原 tar 读取。条目排序、uid/gid、时间戳及 gzip 头固定，全部 mode 保留。仅 `SOURCE_OVERRIDES` 中列出的一个 ELPA 测试文件可替换；原文件、替换文件和公开 diff 均须匹配锁定值，其余源码字节完全保留。manifest 不对自身递归哈希；SHA256SUMS 包含 manifest，归档摘要再覆盖 SHA256SUMS。

需要独立核验原始来源时，将原 tar 留在包目录外运行：

```bash
bash /absolute/package/scripts/verify.sh \
  --archive /absolute/release.tar.gz \
  --source-tar /absolute/source-stage6-target.tar.gz
```

不要用包仓库提交号或 `source/REVISION` 替代原 tar、manifest 和最终归档的摘要。
