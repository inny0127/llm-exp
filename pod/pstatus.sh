#!/usr/bin/env bash
# Print the pod experiment status via the RunPod MCP exec-pod tool. Usage: pstatus.sh POD_ID TOKEN_FILE
cd "$(dirname "$0")"
/home/user/venv/bin/python rp_mcp.py "$2" call exec-pod "{\"id\":\"$1\",\"body\":{\"command\":[\"sh\",\"-c\",\"cat /workspace/out/STATUS; tail -2 /workspace/pod_start.log; ls /workspace/llm-exp/exp04_curriculum/runs/sel /workspace/llm-exp/exp05_order_compression/runs/order 2>/dev/null | grep -c json; for f in /workspace/llm-exp/exp0*/runs/log_*; do echo \$(basename \$f): \$(tail -1 \$f | cut -c1-95); done 2>/dev/null | head -14; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader\"]}}" | /home/user/venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d.get('stdout', d)[-3500:]); print(d.get('stderr','')[-400:])"
