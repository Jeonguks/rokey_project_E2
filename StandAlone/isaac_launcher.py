#!/usr/bin/env python3
"""
Isaac Sim launcher (NO rclpy).
Run with Isaac Sim's python (kit python, py3.11).
"""

import os
import sys

# --- Optional: basic args ---
USD_PATH = os.environ.get("ISAAC_USD", "/home/rokey/Documents/project/canifixit.usd")


def main():
    # NOTE: This file must be executed using Isaac Sim's python (kit/python.sh or equivalent),
    # not system python.
    from omni.isaac.kit import SimulationApp

    # You can tune these
    sim_app = SimulationApp({"headless": False})

    import omni
    import carb

    # 1) Enable ROS2 bridge extension (name may vary slightly by install)
    # Common extension id:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "omni.isaac.ros2_bridge", True
    )

    # 2) Open stage
    from omni.isaac.core.utils.stage import open_stage

    ok = open_stage(USD_PATH)
    if not ok:
        carb.log_error(f"Failed to open USD: {USD_PATH}")

    # 3) Start simulation loop (simple)
    from omni.isaac.core import World

    world = World(stage_units_in_meters=1.0)

    # Press Play in UI manually, or programmatically:
    world.reset()
    carb.log_info("Isaac Sim ready. ROS2 Bridge enabled. Stage loaded.")
    carb.log_info("Now run ROS node launcher in another terminal.")

    while sim_app.is_running():
        world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
