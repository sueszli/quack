import pytest
from mjlab.tasks.registry import list_tasks

import src.task_registry as registry

BASE_TASK_IDS = {"Mjlab-Velocity-Flat-MicroDuck", "Mjlab-Velocity-Rough-MicroDuck", "Mjlab-VelStand-Flat-MicroDuck", "Mjlab-VelStand-Rough-MicroDuck", "Mjlab-StandUp-Flat-MicroDuck", "Mjlab-StandUp-Rough-MicroDuck", "Mjlab-SitStand-Flat-MicroDuck", "Mjlab-SitStand-Rough-MicroDuck", "Mjlab-GroundPick-Flat-MicroDuck", "Mjlab-GroundPick-Rough-MicroDuck", "Mjlab-BallKick-Flat-MicroDuck", "Mjlab-Velocity-Flat-MicroDuck-Rollers", "Mjlab-Velocity-Swizzle-MicroDuck", "Mjlab-RollerCrouch-Flat-MicroDuck", "Mjlab-RollerSlope-Flat-MicroDuck", "Mjlab-RollerStandUp-Flat-MicroDuck", "Mjlab-Spin-Flat-MicroDuck", "Mjlab-Roulade-Flat-MicroDuck"}

_BACKLASH_SPEC_OF_BASE = {"get_walk_spec": "get_walk_backlash_spec", "get_standup_spec": "get_backlash_spec", "get_ground_pick_spec": "get_backlash_spec", "get_walk_rollers_spec": "get_rollers_backlash_spec"}


def test_every_base_task_is_registered():
    assert BASE_TASK_IDS <= set(list_tasks())


def test_enabled_backlash_tasks_are_registered_and_others_are_not():
    registered = set(list_tasks())
    enabled = set(registry._ENABLED_BACKLASH_TASKS)
    assert enabled <= registered, f"enabled backlash twins missing from the registry: {enabled - registered}"
    disabled = {t.task_id for t in registry._BACKLASH_TASKS} - enabled
    assert not (disabled & registered), f"disabled backlash twins leaked into the registry: {disabled & registered}"


def test_default_backlash_set_covers_all_three_robot_models():
    by_id = {t.task_id: t for t in registry._BACKLASH_TASKS}
    covered = {by_id[t].robot_cfg.spec_fn.__name__ for t in registry._DEFAULT_BACKLASH_TASKS}
    assert covered == {"get_walk_backlash_spec", "get_backlash_spec", "get_rollers_backlash_spec"}


def test_default_backlash_ids_exist_in_the_table():
    by_id = {t.task_id for t in registry._BACKLASH_TASKS}
    assert set(registry._DEFAULT_BACKLASH_TASKS) <= by_id


def test_backlash_task_ids_mirror_their_base_id():
    for task in registry._BACKLASH_TASKS:
        assert "-Backlash" in task.task_id
        base = task.task_id.replace("-Backlash", "")
        assert base in BASE_TASK_IDS, f"{task.task_id} has no base task {base}"


@pytest.mark.parametrize("task", registry._BACKLASH_TASKS, ids=lambda t: t.task_id)
def test_backlash_twin_mirrors_base_robot_model(task):
    base_spec = task.make_cfg(**task.make_kwargs).scene.entities["robot"].spec_fn.__name__
    expected = _BACKLASH_SPEC_OF_BASE[base_spec]
    assert task.robot_cfg.spec_fn.__name__ == expected, f"{task.task_id}: base uses {base_spec}, twin uses {task.robot_cfg.spec_fn.__name__}, expected {expected}"
