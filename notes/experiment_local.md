# 本機實驗（Linux／無 Slurm）

先啟用模型的 Python 環境（TFv6 使用 fail2drive，HiPAD 使用 hipad），
再執行 `notes/experiment_local.sh`。入口預設使用目前 PATH 的 `python3`；
也可用 `F2D_LOCAL_PYTHON=/path/to/python` 指定，或沿用 `F2D_EXPERIMENTS_PYTHON`。
腳本不會自動啟用 HPC 上的 Conda 環境。

```bash
conda activate fail2drive
bash notes/experiment_local.sh tfv6 oracle 1,2,3
bash notes/experiment_local.sh tfv6_depth_ttc 2
bash notes/experiment_local.sh hipad depth 2 --dry-run
bash notes/experiment_local.sh --list
bash notes/experiment_local.sh all --dry-run
```

與 `experiment.sh` 共用實驗名稱、允許的 seeds 與模型／策略設定。
支援 `MODEL METHOD SEED[,SEED...]`、`TASK_NAME SEED[,SEED...]`、`--list`、`all`；
`depth` 是 `depth_ttc` 的別名。`all` 按清單順序執行，需要目前 Python
環境同時包含各模型的依賴；一般建議分開啟用環境及執行各模型。

預設逐 seed、逐 route 執行，每次 route attempt 都啟動並關閉 CARLA。
已完成且不屬於可重試錯誤的結果會跳過；失敗最多嘗試 5 次，每次 evaluator
最多 4 小時。結果寫入 `results/<experiment>/<seed>/{res,out,err,viz}/`，
沿用實驗原有的 `_fused` 等輸出名稱後綴，與 HPC 的 `results_shome` 分開。
要延續搬過來的結果，可用 `--out-root results_shome/tfv6_oracle_fused`。

```bash
# 先檢查參數；不啟動 CARLA、模型或評估。
bash notes/experiment_local.sh tfv6 vlm 2 --dry-run

# 僅執行指定路線；也可設定 --route-limit 1。
bash notes/experiment_local.sh tfv6 oracle 1 --route-filter 0078

# 使用已啟動的 CARLA，腳本不會關閉這個 CARLA。
bash notes/experiment_local.sh tfv6 oracle 1 --existing-carla --port 2000
```

本機仍需 CARLA、地圖、模型權重及 steering vectors。預設從專案旁邊找
`lead-cvpr2026`、`HiP-AD`、`Depth-Anything-V2`、`alpamayo1.5`；CARLA 預設
在專案內 `f2d_carla`。可透過下列環境變數覆寫，不必保留 HPC 的絕對路徑：

```bash
export CARLA_ROOT=/data/CARLA
export LEAD_ROOT=/data/lead-cvpr2026
export HIP=/data/HiP-AD
export DEPTH_ROOT=/data/Depth-Anything-V2
export DEPTH_CHECKPOINT=/data/models/depth_anything_v2_metric_vkitti_vitl.pth
export ALPAMAYO_ROOT=/data/alpamayo1.5
export ALPAMAYO_PY=/data/alpamayo1.5/a1_5_venv/bin/python
```

Depth／VLM 預設自動啟動本機 server，輪詢 `/health` 就緒後再開始評估。
執行結束或中斷時會關閉自己啟動的 server；server log 在 `logs/local_eval/`。
如已提供 `DEPTH_TTC_SERVER_URL`、`VLM_SERVER_URL` 或 `ALPAMAYO_SERVER_URL`，
則使用該 server，並保留它。外部 server 需提供同一個 host／port 的 `/health`。

評估使用 `CUDA_VISIBLE_DEVICES` 的第一張 GPU，server 使用第二張；只有一張時
共用該卡，需容納 CARLA、agent 與 server 的總顯存。也可明確指定：

```bash
F2D_LOCAL_EVAL_GPU=0 F2D_LOCAL_SERVER_GPU=1 \
  bash notes/experiment_local.sh tfv6 vlm 2 --graphics-adapter 0
```

CARLA 的 Vulkan adapter 與 CUDA 的編號可能不同，請用 `--graphics-adapter`
或 `CARLA_GRAPHICS_ADAPTER` 指定；預設為 0。
完整選項見 `bash notes/experiment_local.sh --help`。
