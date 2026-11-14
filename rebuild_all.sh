#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
# Rebuild every table and figure from data/ and compare with expected/ (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
python tests/rebuild_all.py
