# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_sglang.serve_defaults import apply_qwen4_serve_defaults

apply_qwen4_serve_defaults()

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()


def main():
    import runpy

    runpy.run_module('sglang.launch_server', run_name='__main__', alter_sys=False)


if __name__ == '__main__':
    main()
