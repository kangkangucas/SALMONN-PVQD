# Test 3D model
python eval_multi_task.py \
    --cfg-path configs/config_3d.yaml \
    --ckpt-path output_dir_3d/202510292335/checkpoint_best.pth \
    --device cuda:0

# run batch checkpoints
bash batch_eval.sh

# result
============================================================
EVALUATION SUMMARY
============================================================
breathiness     - RMSE: 12.2201, MAE: 8.6972
roughness       - RMSE: 17.5317, MAE: 11.7542
strain          - RMSE: 13.3515, MAE: 9.2097

Average RMSE across all tasks: 14.3678

Results saved to eval_results_multi_task.json
[1/3] ✅ Completed: checkpoint_best

============================================================
EVALUATION SUMMARY
============================================================
breathiness     - RMSE: 7.5437, MAE: 6.0819
roughness       - RMSE: 10.1371, MAE: 7.5472
strain          - RMSE: 9.5519, MAE: 6.7819

Average RMSE across all tasks: 9.0776

Results saved to eval_results_multi_task.json
[2/3] ✅ Completed: checkpoint_1


============================================================
EVALUATION SUMMARY
============================================================
breathiness     - RMSE: 8.7811, MAE: 6.7586
roughness       - RMSE: 14.1231, MAE: 9.7361
strain          - RMSE: 17.3120, MAE: 10.5944

Average RMSE across all tasks: 13.4054

Results saved to eval_results_multi_task.json
[3/3] ✅ Completed: checkpoint_3


