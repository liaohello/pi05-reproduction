# 迁移记录

2026-10-24从旧内部仓库按审阅白名单复制，重建干净Git历史；不合并、不cherry-pick、不复制 `.git`。源修订记录为 `5d85d13`（迁移前本地记录，不能据此声称第三方镜像代码有该上游commit）。

部署和追踪脚本迁入 `scripts/`，私有值改为环境变量/本地忽略配置；台账移入 `experiments/`，移除远端根目录等字段并替换路径；LIBERO门禁使用P编号，M0保持 in progress。SO101接入和后续证据按当前仓库记录。

第三方源码不提交。来源文件数、字节数、聚合哈希、镜像digest及获取方式见[PROVENANCE](../third_party/PROVENANCE.md)；许可证见[清单](../THIRD_PARTY_LICENSES.md)。不分发权重或数据集。

迁移后执行过安全扫描、preflight、部署核验和一次LIBERO smoke；旧结果不代表当前状态，当前实验以ledger和报告为准。安全规则见[AGENTS](../AGENTS.md)。
