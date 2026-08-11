#!/usr/bin/env bash

load_cluster_environment() {
    local env_file
    for env_file in /home/tiger/.rh2/entrypoint_envs/*; do
        [ -f "${env_file}" ] && export "$(basename "${env_file}")=$(<"${env_file}")"
    done
    export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-/opt/tiger/arnold/hdfs_client/conf/celer_china-north5}"
    export PATH="/opt/tiger/arnold/hdfs_client:${PATH}"
    export HTTP_PROXY="${HTTP_PROXY:-http://sys-proxy-rd-relay.byted.org:8118/}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${HTTP_PROXY}}"
    export http_proxy="${http_proxy:-${HTTP_PROXY}}"
    export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
    export no_proxy="${no_proxy:-.byted.org,.bytedance.net,localhost,127.0.0.1,::1,10.0.0.0/8,fd00::/8,100.64.0.0/10,fe80::/10,172.16.0.0/12,169.254.0.0/16,192.168.0.0/16}"
    export NO_PROXY="${NO_PROXY:-${no_proxy}}"
}

load_wandb_environment() {
    local repo_root="$1"
    local xtrace_was_on=0
    [[ "$-" == *x* ]] && xtrace_was_on=1
    export WANDB_ENTITY="${WANDB_ENTITY:-hanlinw}"
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
    if [ -z "${WANDB_API_KEY:-}" ]; then
        set +x
        WANDB_API_KEY="$(sed -n 's/^[[:space:]]*export[[:space:]]*WANDB_API_KEY=//p' \
            "${repo_root}/run_prefix_ripo_experiments.sh" | head -n 1)"
        export WANDB_API_KEY
        ((xtrace_was_on)) && set -x
    fi
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "WANDB_API_KEY is unavailable" >&2
        return 1
    fi
}

hdfs_require_absent() {
    local path="$1"
    if hdfs dfs -test -e "${path}"; then
        echo "Refusing to overwrite existing HDFS path: ${path}" >&2
        return 1
    fi
}


