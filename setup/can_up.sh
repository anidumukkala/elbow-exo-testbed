#!/usr/bin/env bash
# Make sure the testbed CAN interfaces are up in CAN FD mode: 1 Mbit/s nominal + 5 Mbit/s data
# (bit-rate switch), automatic bus-off recovery, no loopback / listen-only. The motor must be
# configured for CAN FD too (done once per motor; see README). The kernel picks the transceiver delay
# compensation itself (an explicit `tdc-mode auto` is rejected by this driver).
#
# Each interface's live configuration is checked first; sudo (one password prompt) is only
# used when something is wrong, and then the interface is reconfigured from scratch, which
# also overwrites settings someone else left behind. If the kernel rejects the FD settings, the
# interface is brought up classic instead (never left down) and the script reports the failure.
#
#   bash setup/can_up.sh            # the motor's interface: JointConfig.channel in exo/config.py
#   bash setup/can_up.sh can0 can1  # these interfaces
#
# Without arguments the interface comes from exo/config.py, so moving the motor to the other port
# only needs that one line changed (scripts and VS Code terminals follow it). If the file cannot
# be read, can0 and can1 are both checked.
#
# Exit status: 0 if all present interfaces are OK, 1 otherwise.

BITRATE=1000000
DBITRATE=5000000
RESTART_MS=100

CONFIG="$(cd "$(dirname "$0")/.." && pwd)/exo/config.py"

# JointConfig.channel, read from the source (not imported: no packages needed, works before pixi).
config_channel() {
    python3 - "$CONFIG" <<'EOF' 2>/dev/null
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "JointConfig":
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", None) == "channel":
                print(ast.literal_eval(item.value))
                sys.exit(0)
sys.exit(1)
EOF
}

ifaces=("$@")
if [ ${#ifaces[@]} -eq 0 ]; then
    channel=$(config_channel)
    if [ -n "$channel" ]; then
        ifaces=("$channel")
    else
        echo "[can_up] could not read JointConfig.channel from $CONFIG; checking can0 and can1"
        ifaces=(can0 can1)
    fi
fi
label="CAN FD, $BITRATE / $DBITRATE bit/s"

# Prints what is wrong with an interface (nothing if it is fine). Exit 2 if it does not exist.
problems() {
    ip -details -json link show "$1" 2>/dev/null | python3 -c '
import json, sys
bitrate, dbitrate, restart_ms = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
try:
    link = json.load(sys.stdin)[0]
except Exception:
    sys.exit(2)
info = link.get("linkinfo", {}).get("info_data", {})
ctrl = set(info.get("ctrlmode", []))
found = []
if "UP" not in link.get("flags", []):
    found.append("down")
if "FD" not in ctrl:
    found.append("not in FD mode")
if info.get("bittiming", {}).get("bitrate") != bitrate:
    found.append("bitrate %s" % info.get("bittiming", {}).get("bitrate"))
if info.get("data_bittiming", {}).get("bitrate") != dbitrate:
    found.append("data bitrate %s" % info.get("data_bittiming", {}).get("bitrate"))
if info.get("restart_ms") != restart_ms:
    found.append("restart-ms %s" % info.get("restart_ms"))
if info.get("state") == "BUS-OFF":
    found.append("bus-off")
unwanted = ctrl & {"LOOPBACK", "LISTEN-ONLY", "FD-NON-ISO"}
if unwanted:
    found.append("modes " + ",".join(sorted(unwanted)))
print(", ".join(found))
' "$BITRATE" "$DBITRATE" "$RESTART_MS"
}

todo=()
for ifc in "${ifaces[@]}"; do
    wrong=$(problems "$ifc")
    if [ $? -eq 2 ]; then
        echo "[can_up] $ifc: no such interface, skipped"
    elif [ -z "$wrong" ]; then
        echo "[can_up] $ifc: OK ($label)"
    else
        echo "[can_up] $ifc: $wrong"
        todo+=("$ifc")
    fi
done
[ ${#todo[@]} -eq 0 ] && exit 0

common="restart-ms $RESTART_MS loopback off listen-only off"
fd="bitrate $BITRATE dbitrate $DBITRATE fd on $common"
classic="bitrate $BITRATE fd off $common"
echo "[can_up] reconfiguring ${todo[*]} ($label, restart-ms $RESTART_MS); sudo password needed:"
cmds=""
for ifc in "${todo[@]}"; do
    cmds+="ip link set $ifc down; "
    cmds+="if ! ip link set $ifc type can $fd; then "
    cmds+="echo '[can_up] $ifc: FD settings rejected, bringing it up classic so the link is not down'; "
    cmds+="ip link set $ifc type can $classic; fi; "
    cmds+="ip link set $ifc up; "
done
if ! sudo sh -c "$cmds"; then
    echo "[can_up] WARNING: could not configure ${todo[*]}; run 'pixi run can-up' to retry"
    exit 1
fi

status=0
for ifc in "${todo[@]}"; do
    wrong=$(problems "$ifc")
    if [ -z "$wrong" ]; then
        echo "[can_up] $ifc: OK ($label)"
    else
        echo "[can_up] WARNING: $ifc still not right after reconfiguring: $wrong"
        status=1
    fi
done
exit $status
