# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
# generic C2G GPU job wrapper. args: run_id model cr arms items published_dir [extra runner args...]
# optional env: EXPECT_RUNNER_SHA, EXPECT_PRESS_SHA (frozen hashes from the prereg/addendum; mismatch aborts)
set -e
WORK=$(pwd)
RUN=$1; MODEL=$2; CR=$3; ARMS=$4; ITEMS=$5; PUBDIR=$6; shift 6
P=/hfcache/payload/c2g
PAYLOAD_SHA=8a1b120b5b6be20f6ec3ee712d79b866d9a0f30077aff57603882b98b4519733
mkdir -p /tmp/w && cd /tmp/w
echo "$PAYLOAD_SHA  $P/c2g_payload_base.tgz" | sha256sum -c - >/dev/null || { echo "payload sha mismatch"; exit 6; }
tar xzf $P/c2g_payload_base.tgz
cp $P/two_grid_chunkkv_press.py kvpress/presses/
cp $P/c2g_runner.py $P/timebox.py .
[ -z "$EXPECT_RUNNER_SHA" ] || echo "$EXPECT_RUNNER_SHA  c2g_runner.py" | sha256sum -c - >/dev/null || { echo "runner sha mismatch"; exit 6; }
[ -z "$EXPECT_PRESS_SHA" ] || echo "$EXPECT_PRESS_SHA  kvpress/presses/two_grid_chunkkv_press.py" | sha256sum -c - >/dev/null || { echo "press sha mismatch"; exit 6; }
pip install --no-index --no-deps --find-links /hfcache/payload/wheels peft==0.21.0 -q 2>&1 | tail -1
export PYTHONPATH=/tmp/w
python -c "import kvpress,sys; print(kvpress.__file__)"
sha256sum c2g_runner.py kvpress/presses/two_grid_chunkkv_press.py | cut -c1-24
mkdir -p $WORK/out /hfcache/c2g_backup/$RUN
backup() { cp -r $WORK/out/${RUN}_* /hfcache/c2g_backup/$RUN/ 2>/dev/null || true; }
trap backup EXIT TERM
rc=0
python c2g_runner.py --run-id $RUN --model $MODEL --cr $CR --arms $ARMS --items $WORK/$ITEMS \
   --published lb/$PUBDIR/predictions.csv --published-metrics lb/$PUBDIR/metrics.json --payload-sha $PAYLOAD_SHA --out $WORK/out "$@" || rc=$?
exit $rc
