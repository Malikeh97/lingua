if [[ $(hostname) == "trig-login01" ]]; then
    module load StdEnv/2023
fi
module load cuda/12.6
module load rust

source .venv/bin/activate

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
if [ -f ~/hf_token.txt ]; then
    python -c "from huggingface_hub import login; login(token='$(cat ~/hf_token.txt)')"
fi

if [ -f ~/wandb_token.txt ]; then
    wandb login $(cat ~/wandb_token.txt)
fi


# if hostname starts with rack (vulcan compute node) or trig0 (trillium compute node)
# if OFFLINE_MODE is set to 1, then also go into offline mode
if [[ $OFFLINE_MODE == 1 ]]; then
    echo "Running in offline mode"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=offline
fi


# Check if job should be skipped (already running/pending or completed in past 2 days)
function should_skip_job() {
    local job_name="$1"

    # Check if job is currently running or pending (same user)
    if squeue --name="$job_name" --user="$USER" --noheader 2>/dev/null | grep -q .; then
        echo "SKIP: Job '$job_name' is already running or pending."
        return 0
    fi

    # Check if job completed successfully in the past 2 days (same user)
    local two_days_ago=$(date -d '2 days ago' +%Y-%m-%d 2>/dev/null || date -v-2d +%Y-%m-%d)
    if sacct --name="$job_name" --user="$USER" --starttime="$two_days_ago" --state=COMPLETED --noheader 2>/dev/null | grep -q .; then
        echo "SKIP: Job '$job_name' completed successfully in the past 2 days."
        return 0
    fi

    return 1
}

if [[ $(hostname) == klogin* ]]; then
    # define job submission function (killarney)
    function submit() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        sbatch --job-name="$job_name" --output="logs/$job_name.out" --error="logs/$job_name.out" setup/submit_killarney.sbatch "$command"
    }
    function submit_h100() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        sbatch --job-name="$job_name" --output="logs/$job_name.out" --error="logs/$job_name.out" setup/submit_h100_killarney.sbatch "$command"
    }
elif [[ $(hostname) == vulcan* ]]; then
    # define job submission function (vulcan)
    function submit() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        sbatch --job-name="$job_name" --output="logs/$job_name.out" --error="logs/$job_name.out" setup/submit_vulcan.sbatch "$command"
    }
elif [[ $(hostname) == trig* ]]; then
    # define job submission function (trillium)
    function submit_h100() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        sbatch --job-name="$job_name" --output="logs/$job_name.out" --error="logs/$job_name.out" setup/submit_h100_trillium.sbatch "$command"
    }
elif [[ $(hostname) == login* ]]; then
    # define job submission function (fir)
    function submit_h100() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        sbatch --job-name="$job_name" --output="logs/$job_name.out" --error="logs/$job_name.out" setup/submit_h100_fir.sbatch "$command"
    }
else
    echo "Unknown hostname: $(hostname) - cannot define submit function"
fi
