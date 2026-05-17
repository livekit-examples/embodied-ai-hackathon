uv run python run_policy_operator.py \
      --policy pick_up=$HOME/outputs/act-spc-pick-stuff-final_steps40000_bs64_lr5e-5_chunk50/checkpoints/last/pretrained_model \
      --script put_down=$(realpath ../../utils/data_collection/data/binhpham/spc-mock-put-down) \
      --reward pick_up=$HOME/outputs/reward-binhpham_spc-pick-stuff-20260517-2043/best.pt --reward-threshold 0.4 \
      --default-policy pick_up