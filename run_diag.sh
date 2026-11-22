#!/bin/sh
/app/diag
echo "=== PIPELINE RUN 1 ===" >&2
env -u SS_QUIET /app/shopping
echo "=== PIPELINE RUN 2 ===" >&2
rm -rf /output/user_daily /output/campaign_daily
env -u SS_QUIET /app/shopping
echo "=== deliberately failing to surface logs ===" >&2
exit 1
