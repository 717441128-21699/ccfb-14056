# MoCap Clean — 光学动捕标记点清洗台

纯 TypeScript 实现的网页端光学动捕数据清洗与骨骼解算工具。标记点数据导入后，自动定位丢点空档、
按刚体关系与前后轨迹补点、识别并还原标签互换，再解算成骨骼动画，锁定着地脚消除打滑、标出肢体穿模，
并可把同一套动作重定向到不同身材的角色。全部计算在浏览器本地完成，三维预览基于原生 WebGL2。

## 能力与验收指标

| 需求 | 实现 | 验收测试 |
| --- | --- | --- |
| 丢点检测 + 刚体/轨迹补点 | Kabsch 刚体预测为主，三次/线性轨迹兜底，时间传播处理欠定帧 | `tests/gap-fill.test.ts` |
| 补点误差 ≤ 2 mm | 零测量噪声下补点 RMS 为 0；0.5 mm 噪声下 RMS < 2 mm | 同上 |
| 标签互换识别还原 | 簇内转置枚举 + 全局匈牙利指派，双峰 RMS 门控 | `tests/swap-detect.test.ts` |
| 1 万帧清洗 ≤ 30 s | 实测约 2.8 s（10k 帧 × 51 标记），吞吐 > 3000 帧/秒 | `tests/pipeline.bench.test.ts` |
| 标记点 → 骨骼 | 簇锚关节还原 + 解析二骨 IK + 骨长投影约束 | `tests/solver.test.ts` |
| 脚着地锁定防滑 | 自适应速度/高度门限检测支撑相，整腿链世界坐标锁定（漂移 < 2 mm） | 同上 |
| 肢体穿模标出 | 胶囊-胶囊最近距离，拓扑近邻过滤 | 同上 |
| 不同身材重定向 | 4 套身材预设，姿态保持 + 支撑脚世界锁定 | 同上 |
| 走/跑/跳测试数据 | 程序化骨骼动作反推标记点，再加噪声/丢点/互换 | 全部测试夹具 |

## 流水线

```
程序化动作 (walk/run/jump) 或 CSV/JSON 导入
        │  骨架 FK → 标记点反算 → 加噪声/丢点/互换
        ▼
原始 MarkerClip
        │  Pass A 簇内刚体拟合 + 转置枚举修簇内互换
        │  Pass B 全局匈牙利指派修跨簇互换
        │  Pass C 重拟合刷新预测 → 观测帧去噪 → 间隙填补（刚体优先/轨迹兜底）
        ▼
清洗后 MarkerClip（无丢点、标签归位）
        │  簇→锚关节还原、二骨 IK、骨长约束、脚锁、穿模检测
        ▼
SolvedSkeleton（世界/局部关节位姿 + 接触区间 + 穿模事件）
        │  骨长缩放重定向到目标身材
        ▼
RetargetedSkeleton → WebGL2 预览 / BVH·CSV·JSON 导出
```

## 目录结构

```
src/
  math/        向量、四元数、矩阵、Kabsch（Jacobi 特征分解）、随机数、样条
  core/
    skeleton.ts    24 关节人体 + 51 标记的 13 个刚体簇定义
    motion.ts      走/跑/跳程序化动作
    synthetic.ts   骨架反推标记点 + 噪声/丢点/互换注入
    hungarian.ts   匈牙利指派
    cleaner.ts     清洗主流水线（互换、补点、去噪）
    ik.ts          解析二骨逆运动学
    solver.ts      骨骼解算、脚锁、穿模
    retarget.ts    身材重定向
    fk.ts          正向运动学
  io/formats.ts   CSV / JSON 导入导出、BVH 写出
  render/         WebGL2 渲染器、轨道控制、后台 Worker
tests/           39 个 Vitest 用例
```

## 常用命令

```bash
npm install
npm run dev        # 启动开发服务器
npm test           # 运行全部单测与性能基准
npm run bench      # 仅跑 1 万帧性能基准
npm run build      # 类型检查 + 生产构建到 dist/
npm run preview    # 预览生产构建
```

## 数据格式

导入 CSV 表头为 `frame,time,<marker>.x,<marker>.y,<marker>.z, ...`，缺失采样留空；
JSON 采用 `mocap-clean-markers` 结构，每帧给出标记名到三维坐标或 `null` 的映射。
标记名匹配内置人体布局的 51 个标记时，自动获得刚体簇与锚关节信息；自定义标记集则按
可识别的簇子集工作。导出支持清洗后标记的 CSV/JSON，以及解算骨骼的 BVH。

## 精度说明

补点误差的物理下限由输入测量噪声决定：刚体拟合会把标记点噪声传播到被补点的坐标上。
测试因此区分两层指标——零噪声下的纯算法精度（要求严格 2 mm 内，实测 0 mm）与含噪端到端
精度（随 σ 线性变化，0.5 mm σ 时 < 2 mm，1.5 mm σ 时约 4 mm）。当一个簇在某帧可见点
不足三个时，系统沿用上一可信簇姿态做时间传播，避免欠定拟合产生厘米级尖峰。
