# Start training job
jid1=$(sbatch --array=2-4 ebds_train.sh | awk '{print $NF}')
echo "Training job: $jid1"

# Start eval job after training jobs completed
jid2=$(sbatch --dependency=afterok:$jid1 ebds_eval.sh | awk '{print $NF}') 
echo "Evaluation job: $jid2"
