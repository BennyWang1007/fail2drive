# Steering policy：最快啟用方式

以下指令假設目前位於 `fail2drive/` repo root。
三個 vector 的順序固定是 `brake,left,right`。請依 planner 與實際資料夾名稱替換
`{planner_name}`、`{brake_vector_name}`、`{left_vector_name}`、`{right_vector_name}`。
例如 TransFuser 是 `transfuser`、`brake`、`left`、`right`；不同 HiPAD vector
可能使用不同的 action 資料夾。切換 policy 時保留下面的 `=0`，可避免舊 shell
中殘留的開關蓋過目前選擇。

## fixed frame（debug）

在指定 frame 範圍固定注入一個 steering vector，適合快速確認 hook、vector 與 alpha
是否正常。這個模式不設定 named policy；只要 `STEERING_ALPHA > 0`，程式就會選用
`FixedAfterFramePolicy`。

```bash
export VLM_STEERING=0
export DEPTH_TTC_STEERING=0
unset ACTIVATION_POLICY STEERING_POLICY
unset PDM_ORACLE_STEERING PDM_ORACLE_POLICY ORACLE_STEERING ORACLE_POLICY

export STEERING_ALPHA=1.0
export START_STEERING_FRAME=100
export END_STEERING_FRAME=150

unset ACTIVATION_VECTOR_PATHS
export ACTIVATION_VECTOR_PATH="$PWD/steering/{planner_name}/post_process/{vector_name}/steering_vector.pt"
```

- `START_STEERING_FRAME` 與 `END_STEERING_FRAME` 都包含在啟用範圍內。
- 若要從 start frame 一直開到 route 結束，執行 `unset END_STEERING_FRAME`。
- `{vector_name}` 可替換成實際的 `brake`、`left`、`right` 或該 planner 的 action 資料夾。
- Debug 指定單一 action 時建議使用上面的 `ACTIVATION_VECTOR_PATH`。如果改用三向
  `ACTIVATION_VECTOR_PATHS`，fixed policy 的 scalar alpha 會被解讀成
  `[alpha, 0, 0]`，因此只會注入 brake vector。

## privilege

程式中的正式名稱是 `pdm_oracle`。這條 policy 使用 CARLA / scenario-runner 的 privileged state，不需要額外 inference server。

```bash
export VLM_STEERING=0
export DEPTH_TTC_STEERING=0
export ACTIVATION_POLICY=pdm_oracle
export ACTIVATION_VECTOR_PATHS="$PWD/steering/{planner_name}/post_process/{brake_vector_name}/steering_vector.pt,$PWD/steering/{planner_name}/post_process/{left_vector_name}/steering_vector.pt,$PWD/steering/{planner_name}/post_process/{right_vector_name}/steering_vector.pt"
```

會直接使用目前 code default：

```bash
export PDM_ORACLE_ALPHA=1.0
export PDM_ORACLE_TRIGGER_DISTANCE=50.0
export PDM_ORACLE_HOLD_FRAMES=8
export PDM_ORACLE_COOLDOWN_FRAMES=20
export PDM_ORACLE_GENERAL_BRAKE=0
```

最少只需要前一段；第二段只有要固定、調整或記錄 default 時才需要設。

## occupancy

程式中的正式名稱是 `depth_ttc`。Planner 端最快可用設定：

```bash
export VLM_STEERING=0
export DEPTH_TTC_STEERING=1
export ACTIVATION_POLICY=depth_ttc
export DEPTH_TTC_SERVER_URL=http://127.0.0.1:8766/score
export DEPTH_TTC_EVERY_N=5
export DEPTH_TTC_HOLD_FRAMES=5
export DEPTH_TTC_VERBOSE=0
export ACTIVATION_VECTOR_PATHS="$PWD/steering/{planner_name}/post_process/{brake_vector_name}/steering_vector.pt,$PWD/steering/{planner_name}/post_process/{left_vector_name}/steering_vector.pt,$PWD/steering/{planner_name}/post_process/{right_vector_name}/steering_vector.pt"
```

這條 policy 必須先啟動 Depth Anything / Depth-TTC server；只設 env 不會自動啟動 server：

```bash
export DEPTH_ROOT=/path/to/Depth-Anything-V2
export DEPTH_CHECKPOINT=/path/to/depth_anything_v2_metric_vkitti_vitl.pth

python tools/depth_ttc_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --device cuda:0 \
  --depth-anything-root "$DEPTH_ROOT" \
  --checkpoint "$DEPTH_CHECKPOINT" \
  --encoder vitl \
  --max-depth 80
```

目前 candidate alpha defaults；通常不必另外設：

```bash
export DEPTH_TTC_BRAKE_ALPHA=3.0
export DEPTH_TTC_LEFT_ALPHA=1.0
export DEPTH_TTC_RIGHT_ALPHA=1.0
```

## VLM

以下使用目前 evaluation script 採用的 Alpamayo server backend。Planner 端最快可用設定：

```bash
export DEPTH_TTC_STEERING=0
export VLM_STEERING=1
export ACTIVATION_POLICY=vlm
export VLM_BACKEND=alpamayo_server
export VLM_SERVER_URL=http://127.0.0.1:8765/generate
export VLM_EVERY_N=5
export VLM_DECISION_TTL_FRAMES=5
export VLM_SERVER_FRAME_OFFSETS=-6,-4,-2,0
export VLM_VERBOSE=0
export ACTIVATION_VECTOR_PATHS="$PWD/steering/{planner_name}/post_process/{brake_vector_name}/steering_vector.pt,$PWD/steering/{planner_name}/post_process/{left_vector_name}/steering_vector.pt,$PWD/steering/{planner_name}/post_process/{right_vector_name}/steering_vector.pt"
```

這條 policy 必須先啟動 Alpamayo server；只設 env 不會自動啟動 server：

```bash
export ALPAMAYO_ROOT=/path/to/alpamayo1.5

"$ALPAMAYO_ROOT/a1_5_venv/bin/python" tools/alpamayo_vlm_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --alpamayo-version 1.5 \
  --model nvidia/Alpamayo-1.5-10B \
  --max-new-tokens 256
```

目前 action alpha defaults；通常不必另外設：

```bash
export VLM_BRAKE_ALPHA=2.0
export VLM_WEAK_BRAKE_ALPHA=0.5
export VLM_LATERAL_ALPHA=1.0
```

## 快速確認

Agent 啟動 log 應顯示：

- fixed frame：`Steering Policy: FixedAfterFramePolicy`
- privilege：`Steering Policy: PDMOraclePolicy`
- occupancy：`Depth TTC steering enabled: http://127.0.0.1:8766/score`
- VLM：`Steering Policy: VLMPolicy`，以及 `AsyncVLMGate loaded backend=alpamayo_server`

若 policy 名稱正確但 steering 完全沒作用，先檢查 `ACTIVATION_VECTOR_PATH`，或
`ACTIVATION_VECTOR_PATHS` 的三個檔案是否存在；vector path 沒有設定時 injector
會停用。
