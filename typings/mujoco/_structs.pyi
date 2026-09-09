import collections.abc
import mujoco._enums
import numpy
import numpy.typing
import typing
from typing import Callable, ClassVar, overload

class MjContact:
    H: numpy.typing.NDArray[numpy.float64]
    dim: int
    dist: float
    efc_address: int
    elem: numpy.typing.NDArray[numpy.int32]
    exclude: int
    flex: numpy.typing.NDArray[numpy.int32]
    frame: numpy.typing.NDArray[numpy.float64]
    friction: numpy.typing.NDArray[numpy.float64]
    geom: numpy.typing.NDArray[numpy.int32]
    geom1: int
    geom2: int
    includemargin: float
    mu: float
    pos: numpy.typing.NDArray[numpy.float64]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    solreffriction: numpy.typing.NDArray[numpy.float64]
    vert: numpy.typing.NDArray[numpy.int32]
    def __init__(self) -> None: ...
    def __copy__(self) -> MjContact: ...
    def __deepcopy__(self, arg0: dict) -> MjContact: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjData:
    bind: ClassVar[Callable] = ...
    M: numpy.typing.NDArray[numpy.float64]
    act: numpy.typing.NDArray[numpy.float64]
    act_dot: numpy.typing.NDArray[numpy.float64]
    actuator_force: numpy.typing.NDArray[numpy.float64]
    actuator_length: numpy.typing.NDArray[numpy.float64]
    actuator_moment: numpy.typing.NDArray[numpy.float64]
    actuator_velocity: numpy.typing.NDArray[numpy.float64]
    body_awake: numpy.typing.NDArray[numpy.int32]
    body_awake_ind: numpy.typing.NDArray[numpy.int32]
    bvh_aabb_dyn: numpy.typing.NDArray[numpy.float64]
    bvh_active: numpy.typing.NDArray[numpy.bool]
    cacc: numpy.typing.NDArray[numpy.float64]
    cam_xmat: numpy.typing.NDArray[numpy.float64]
    cam_xpos: numpy.typing.NDArray[numpy.float64]
    cdof: numpy.typing.NDArray[numpy.float64]
    cdof_dot: numpy.typing.NDArray[numpy.float64]
    cfrc_ext: numpy.typing.NDArray[numpy.float64]
    cfrc_int: numpy.typing.NDArray[numpy.float64]
    cinert: numpy.typing.NDArray[numpy.float64]
    crb: numpy.typing.NDArray[numpy.float64]
    ctrl: numpy.typing.NDArray[numpy.float64]
    cvel: numpy.typing.NDArray[numpy.float64]
    dof_awake_ind: numpy.typing.NDArray[numpy.int32]
    energy: numpy.typing.NDArray[numpy.float64]
    eq_active: numpy.typing.NDArray[numpy.bool]
    flexedge_J: numpy.typing.NDArray[numpy.float64]
    flexedge_length: numpy.typing.NDArray[numpy.float64]
    flexedge_velocity: numpy.typing.NDArray[numpy.float64]
    flexelem_aabb: numpy.typing.NDArray[numpy.float64]
    flexvert_J: numpy.typing.NDArray[numpy.float64]
    flexvert_length: numpy.typing.NDArray[numpy.float64]
    flexvert_xpos: numpy.typing.NDArray[numpy.float64]
    flg_energypos: bool
    flg_energyvel: bool
    flg_rnepost: bool
    flg_subtreevel: bool
    geom_xmat: numpy.typing.NDArray[numpy.float64]
    geom_xpos: numpy.typing.NDArray[numpy.float64]
    history: numpy.typing.NDArray[numpy.float64]
    light_xdir: numpy.typing.NDArray[numpy.float64]
    light_xpos: numpy.typing.NDArray[numpy.float64]
    maxuse_arena: int
    maxuse_con: int
    maxuse_efc: int
    maxuse_stack: int
    mocap_pos: numpy.typing.NDArray[numpy.float64]
    mocap_quat: numpy.typing.NDArray[numpy.float64]
    moment_colind: numpy.typing.NDArray[numpy.int32]
    moment_rowadr: numpy.typing.NDArray[numpy.int32]
    moment_rownnz: numpy.typing.NDArray[numpy.int32]
    nA: int
    nJ: int
    nY: int
    narena: int
    nbody_awake: int
    nbuffer: int
    ncon: int
    ne: int
    nefc: int
    nf: int
    nidof: int
    nisland: int
    nl: int
    nparent_awake: int
    nplugin: int
    ntree_awake: int
    nv_awake: int
    parena: int
    parent_awake_ind: numpy.typing.NDArray[numpy.int32]
    pbase: int
    plugin: numpy.typing.NDArray[numpy.int32]
    plugin_data: numpy.typing.NDArray[numpy.uint64]
    plugin_state: numpy.typing.NDArray[numpy.float64]
    pstack: int
    qDeriv: numpy.typing.NDArray[numpy.float64]
    qH: numpy.typing.NDArray[numpy.float64]
    qHDiagInv: numpy.typing.NDArray[numpy.float64]
    qLD: numpy.typing.NDArray[numpy.float64]
    qLDiagInv: numpy.typing.NDArray[numpy.float64]
    qLU: numpy.typing.NDArray[numpy.float64]
    qM: numpy.typing.NDArray[numpy.float64]
    qacc: numpy.typing.NDArray[numpy.float64]
    qacc_smooth: numpy.typing.NDArray[numpy.float64]
    qacc_warmstart: numpy.typing.NDArray[numpy.float64]
    qfrc_actuator: numpy.typing.NDArray[numpy.float64]
    qfrc_applied: numpy.typing.NDArray[numpy.float64]
    qfrc_bias: numpy.typing.NDArray[numpy.float64]
    qfrc_constraint: numpy.typing.NDArray[numpy.float64]
    qfrc_damper: numpy.typing.NDArray[numpy.float64]
    qfrc_fluid: numpy.typing.NDArray[numpy.float64]
    qfrc_gravcomp: numpy.typing.NDArray[numpy.float64]
    qfrc_inverse: numpy.typing.NDArray[numpy.float64]
    qfrc_passive: numpy.typing.NDArray[numpy.float64]
    qfrc_smooth: numpy.typing.NDArray[numpy.float64]
    qfrc_spring: numpy.typing.NDArray[numpy.float64]
    qpos: numpy.typing.NDArray[numpy.float64]
    qvel: numpy.typing.NDArray[numpy.float64]
    sensordata: numpy.typing.NDArray[numpy.float64]
    site_xmat: numpy.typing.NDArray[numpy.float64]
    site_xpos: numpy.typing.NDArray[numpy.float64]
    solver_fwdinv: numpy.typing.NDArray[numpy.float64]
    solver_niter: numpy.typing.NDArray[numpy.int32]
    solver_nnz: numpy.typing.NDArray[numpy.int32]
    subtree_angmom: numpy.typing.NDArray[numpy.float64]
    subtree_com: numpy.typing.NDArray[numpy.float64]
    subtree_linvel: numpy.typing.NDArray[numpy.float64]
    ten_J: numpy.typing.NDArray[numpy.float64]
    ten_length: numpy.typing.NDArray[numpy.float64]
    ten_velocity: numpy.typing.NDArray[numpy.float64]
    ten_wrapadr: numpy.typing.NDArray[numpy.int32]
    ten_wrapnum: numpy.typing.NDArray[numpy.int32]
    threadpool: int
    time: float
    tree_asleep: numpy.typing.NDArray[numpy.int32]
    tree_awake: numpy.typing.NDArray[numpy.int32]
    userdata: numpy.typing.NDArray[numpy.float64]
    wrap_obj: numpy.typing.NDArray[numpy.int32]
    wrap_xpos: numpy.typing.NDArray[numpy.float64]
    xanchor: numpy.typing.NDArray[numpy.float64]
    xaxis: numpy.typing.NDArray[numpy.float64]
    xfrc_applied: numpy.typing.NDArray[numpy.float64]
    ximat: numpy.typing.NDArray[numpy.float64]
    xipos: numpy.typing.NDArray[numpy.float64]
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    xquat: numpy.typing.NDArray[numpy.float64]
    def __init__(self, arg0: MjModel) -> None: ...
    def actuator(self, *args, **kwargs): ...
    def bind_scalar(self, *args, **kwargs): ...
    def body(self, *args, **kwargs): ...
    def cam(self, *args, **kwargs): ...
    def camera(self, *args, **kwargs): ...
    def geom(self, *args, **kwargs): ...
    def jnt(self, *args, **kwargs): ...
    def joint(self, *args, **kwargs): ...
    def light(self, *args, **kwargs): ...
    def sensor(self, *args, **kwargs): ...
    def site(self, *args, **kwargs): ...
    def ten(self, *args, **kwargs): ...
    def tendon(self, *args, **kwargs): ...
    def __copy__(self) -> MjData: ...
    def __deepcopy__(self, arg0: dict) -> MjData: ...
    @property
    def contact(self) -> _MjContactList: ...
    @property
    def dof_island(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_AR(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_AR_colind(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_AR_rowadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_AR_rownnz(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_D(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_J(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_J_colind(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_J_rowadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_J_rownnz(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_J_rowsuper(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_KBIP(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_R(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_Y(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_Y_colind(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_Y_rowadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_Y_rownnz(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_aref(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_b(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_diagA(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_force(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_frictionloss(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_id(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_island(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_margin(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_pos(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_state(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_type(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def efc_vel(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iacc(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iacc_smooth(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iefc_D(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iefc_R(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iefc_aref(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iefc_force(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iefc_frictionloss(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def iefc_id(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def iefc_state(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def iefc_type(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def ifrc_constraint(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def ifrc_smooth(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def island_dofadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_idofadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_iefcadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_itreeadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_ne(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_nefc(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_nf(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_ntree(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def island_nv(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def map_dof2idof(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def map_efc2iefc(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def map_idof2dof(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def map_iefc2efc(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def map_itree2tree(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def model(self) -> MjModel: ...
    @property
    def signature(self) -> int: ...
    @property
    def solver(self) -> _MjSolverStatList: ...
    @property
    def tendon_efcadr(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def timer(self) -> _MjTimerStatList: ...
    @property
    def tree_island(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def warning(self) -> _MjWarningStatList: ...

class MjLROpt:
    accel: float
    interval: float
    inttotal: float
    maxforce: float
    mode: int
    timeconst: float
    timestep: float
    tolrange: float
    useexisting: int
    uselimit: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjLROpt: ...
    def __deepcopy__(self, arg0: dict) -> MjLROpt: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjLogConfig:
    logfile: str
    logto_console: bool
    logto_file: bool
    topics: int
    def __init__(self) -> None: ...
    @staticmethod
    def get() -> MjLogConfig: ...
    def set(self) -> None: ...
    def __copy__(self) -> MjLogConfig: ...
    def __deepcopy__(self, arg0: dict) -> MjLogConfig: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjLogMessage:
    level: int
    line: int
    subject: str
    timestamp: bool
    topic: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjLogMessage: ...
    def __deepcopy__(self, arg0: dict) -> MjLogMessage: ...
    def __eq__(self, arg0: object) -> bool: ...
    @property
    def body(self) -> object: ...
    @property
    def file(self) -> object: ...
    @property
    def func(self) -> object: ...

class MjModel:
    bind: ClassVar[Callable] = ...
    B_colind: numpy.typing.NDArray[numpy.int32]
    B_rowadr: numpy.typing.NDArray[numpy.int32]
    B_rownnz: numpy.typing.NDArray[numpy.int32]
    D_colind: numpy.typing.NDArray[numpy.int32]
    D_diag: numpy.typing.NDArray[numpy.int32]
    D_rowadr: numpy.typing.NDArray[numpy.int32]
    D_rownnz: numpy.typing.NDArray[numpy.int32]
    M_colind: numpy.typing.NDArray[numpy.int32]
    M_rowadr: numpy.typing.NDArray[numpy.int32]
    M_rownnz: numpy.typing.NDArray[numpy.int32]
    actuator_acc0: numpy.typing.NDArray[numpy.float64]
    actuator_actadr: numpy.typing.NDArray[numpy.int32]
    actuator_actearly: numpy.typing.NDArray[numpy.bool]
    actuator_actlimited: numpy.typing.NDArray[numpy.bool]
    actuator_actnum: numpy.typing.NDArray[numpy.int32]
    actuator_actrange: numpy.typing.NDArray[numpy.float64]
    actuator_armature: numpy.typing.NDArray[numpy.float64]
    actuator_biasprm: numpy.typing.NDArray[numpy.float64]
    actuator_biastype: numpy.typing.NDArray[numpy.int32]
    actuator_cranklength: numpy.typing.NDArray[numpy.float64]
    actuator_ctrllimited: numpy.typing.NDArray[numpy.bool]
    actuator_ctrlrange: numpy.typing.NDArray[numpy.float64]
    actuator_damping: numpy.typing.NDArray[numpy.float64]
    actuator_dampingpoly: numpy.typing.NDArray[numpy.float64]
    actuator_delay: numpy.typing.NDArray[numpy.float64]
    actuator_dynprm: numpy.typing.NDArray[numpy.float64]
    actuator_dyntype: numpy.typing.NDArray[numpy.int32]
    actuator_forcelimited: numpy.typing.NDArray[numpy.bool]
    actuator_forcerange: numpy.typing.NDArray[numpy.float64]
    actuator_gainprm: numpy.typing.NDArray[numpy.float64]
    actuator_gaintype: numpy.typing.NDArray[numpy.int32]
    actuator_gear: numpy.typing.NDArray[numpy.float64]
    actuator_group: numpy.typing.NDArray[numpy.int32]
    actuator_history: numpy.typing.NDArray[numpy.int32]
    actuator_historyadr: numpy.typing.NDArray[numpy.int32]
    actuator_length0: numpy.typing.NDArray[numpy.float64]
    actuator_lengthrange: numpy.typing.NDArray[numpy.float64]
    actuator_plugin: numpy.typing.NDArray[numpy.int32]
    actuator_trnid: numpy.typing.NDArray[numpy.int32]
    actuator_trntype: numpy.typing.NDArray[numpy.int32]
    actuator_user: numpy.typing.NDArray[numpy.float64]
    body_bvhadr: numpy.typing.NDArray[numpy.int32]
    body_bvhnum: numpy.typing.NDArray[numpy.int32]
    body_conaffinity: numpy.typing.NDArray[numpy.int32]
    body_contype: numpy.typing.NDArray[numpy.int32]
    body_dofadr: numpy.typing.NDArray[numpy.int32]
    body_dofnum: numpy.typing.NDArray[numpy.int32]
    body_geomadr: numpy.typing.NDArray[numpy.int32]
    body_geomnum: numpy.typing.NDArray[numpy.int32]
    body_gravcomp: numpy.typing.NDArray[numpy.float64]
    body_inertia: numpy.typing.NDArray[numpy.float64]
    body_invweight0: numpy.typing.NDArray[numpy.float64]
    body_ipos: numpy.typing.NDArray[numpy.float64]
    body_iquat: numpy.typing.NDArray[numpy.float64]
    body_jntadr: numpy.typing.NDArray[numpy.int32]
    body_jntnum: numpy.typing.NDArray[numpy.int32]
    body_margin: numpy.typing.NDArray[numpy.float64]
    body_mass: numpy.typing.NDArray[numpy.float64]
    body_mocapid: numpy.typing.NDArray[numpy.int32]
    body_parentid: numpy.typing.NDArray[numpy.int32]
    body_plugin: numpy.typing.NDArray[numpy.int32]
    body_pos: numpy.typing.NDArray[numpy.float64]
    body_quat: numpy.typing.NDArray[numpy.float64]
    body_rootid: numpy.typing.NDArray[numpy.int32]
    body_sameframe: numpy.typing.NDArray[numpy.uint8]
    body_simple: numpy.typing.NDArray[numpy.uint8]
    body_subtreemass: numpy.typing.NDArray[numpy.float64]
    body_treeid: numpy.typing.NDArray[numpy.int32]
    body_user: numpy.typing.NDArray[numpy.float64]
    body_weldid: numpy.typing.NDArray[numpy.int32]
    bvh_aabb: numpy.typing.NDArray[numpy.float64]
    bvh_child: numpy.typing.NDArray[numpy.int32]
    bvh_depth: numpy.typing.NDArray[numpy.int32]
    bvh_nodeid: numpy.typing.NDArray[numpy.int32]
    cam_bodyid: numpy.typing.NDArray[numpy.int32]
    cam_fovy: numpy.typing.NDArray[numpy.float64]
    cam_intrinsic: numpy.typing.NDArray[numpy.float32]
    cam_ipd: numpy.typing.NDArray[numpy.float64]
    cam_mat0: numpy.typing.NDArray[numpy.float64]
    cam_mode: numpy.typing.NDArray[numpy.int32]
    cam_output: numpy.typing.NDArray[numpy.int32]
    cam_pos: numpy.typing.NDArray[numpy.float64]
    cam_pos0: numpy.typing.NDArray[numpy.float64]
    cam_poscom0: numpy.typing.NDArray[numpy.float64]
    cam_projection: numpy.typing.NDArray[numpy.int32]
    cam_quat: numpy.typing.NDArray[numpy.float64]
    cam_resolution: numpy.typing.NDArray[numpy.int32]
    cam_sensorsize: numpy.typing.NDArray[numpy.float32]
    cam_targetbodyid: numpy.typing.NDArray[numpy.int32]
    cam_user: numpy.typing.NDArray[numpy.float64]
    dof_M0: numpy.typing.NDArray[numpy.float64]
    dof_Madr: numpy.typing.NDArray[numpy.int32]
    dof_armature: numpy.typing.NDArray[numpy.float64]
    dof_bodyid: numpy.typing.NDArray[numpy.int32]
    dof_damping: numpy.typing.NDArray[numpy.float64]
    dof_dampingpoly: numpy.typing.NDArray[numpy.float64]
    dof_frictionloss: numpy.typing.NDArray[numpy.float64]
    dof_invweight0: numpy.typing.NDArray[numpy.float64]
    dof_jntid: numpy.typing.NDArray[numpy.int32]
    dof_length: numpy.typing.NDArray[numpy.float64]
    dof_parentid: numpy.typing.NDArray[numpy.int32]
    dof_simplenum: numpy.typing.NDArray[numpy.int32]
    dof_solimp: numpy.typing.NDArray[numpy.float64]
    dof_solref: numpy.typing.NDArray[numpy.float64]
    dof_treeid: numpy.typing.NDArray[numpy.int32]
    eq_active0: numpy.typing.NDArray[numpy.bool]
    eq_data: numpy.typing.NDArray[numpy.float64]
    eq_obj1id: numpy.typing.NDArray[numpy.int32]
    eq_obj2id: numpy.typing.NDArray[numpy.int32]
    eq_objtype: numpy.typing.NDArray[numpy.int32]
    eq_solimp: numpy.typing.NDArray[numpy.float64]
    eq_solref: numpy.typing.NDArray[numpy.float64]
    eq_type: numpy.typing.NDArray[numpy.int32]
    exclude_signature: numpy.typing.NDArray[numpy.int32]
    flex_activelayers: numpy.typing.NDArray[numpy.int32]
    flex_bending: numpy.typing.NDArray[numpy.float64]
    flex_bendingadr: numpy.typing.NDArray[numpy.int32]
    flex_bvhadr: numpy.typing.NDArray[numpy.int32]
    flex_bvhnum: numpy.typing.NDArray[numpy.int32]
    flex_cellnum: numpy.typing.NDArray[numpy.int32]
    flex_centered: numpy.typing.NDArray[numpy.bool]
    flex_conaffinity: numpy.typing.NDArray[numpy.int32]
    flex_condim: numpy.typing.NDArray[numpy.int32]
    flex_contype: numpy.typing.NDArray[numpy.int32]
    flex_damping: numpy.typing.NDArray[numpy.float64]
    flex_dim: numpy.typing.NDArray[numpy.int32]
    flex_edge: numpy.typing.NDArray[numpy.int32]
    flex_edgeadr: numpy.typing.NDArray[numpy.int32]
    flex_edgedamping: numpy.typing.NDArray[numpy.float64]
    flex_edgeequality: numpy.typing.NDArray[numpy.int32]
    flex_edgeflap: numpy.typing.NDArray[numpy.int32]
    flex_edgenum: numpy.typing.NDArray[numpy.int32]
    flex_edgestiffness: numpy.typing.NDArray[numpy.float64]
    flex_elem: numpy.typing.NDArray[numpy.int32]
    flex_elemadr: numpy.typing.NDArray[numpy.int32]
    flex_elemdataadr: numpy.typing.NDArray[numpy.int32]
    flex_elemedge: numpy.typing.NDArray[numpy.int32]
    flex_elemedgeadr: numpy.typing.NDArray[numpy.int32]
    flex_elemlayer: numpy.typing.NDArray[numpy.int32]
    flex_elemnum: numpy.typing.NDArray[numpy.int32]
    flex_elemtexcoord: numpy.typing.NDArray[numpy.int32]
    flex_evpair: numpy.typing.NDArray[numpy.int32]
    flex_evpairadr: numpy.typing.NDArray[numpy.int32]
    flex_evpairnum: numpy.typing.NDArray[numpy.int32]
    flex_flatskin: numpy.typing.NDArray[numpy.bool]
    flex_friction: numpy.typing.NDArray[numpy.float64]
    flex_gap: numpy.typing.NDArray[numpy.float64]
    flex_group: numpy.typing.NDArray[numpy.int32]
    flex_internal: numpy.typing.NDArray[numpy.bool]
    flex_interp: numpy.typing.NDArray[numpy.int32]
    flex_margin: numpy.typing.NDArray[numpy.float64]
    flex_matid: numpy.typing.NDArray[numpy.int32]
    flex_node: numpy.typing.NDArray[numpy.float64]
    flex_node0: numpy.typing.NDArray[numpy.float64]
    flex_nodeadr: numpy.typing.NDArray[numpy.int32]
    flex_nodebodyid: numpy.typing.NDArray[numpy.int32]
    flex_nodenum: numpy.typing.NDArray[numpy.int32]
    flex_passive: numpy.typing.NDArray[numpy.int32]
    flex_priority: numpy.typing.NDArray[numpy.int32]
    flex_radius: numpy.typing.NDArray[numpy.float64]
    flex_rgba: numpy.typing.NDArray[numpy.float32]
    flex_rigid: numpy.typing.NDArray[numpy.bool]
    flex_selfcollide: numpy.typing.NDArray[numpy.int32]
    flex_shell: numpy.typing.NDArray[numpy.int32]
    flex_shelldataadr: numpy.typing.NDArray[numpy.int32]
    flex_shellnum: numpy.typing.NDArray[numpy.int32]
    flex_size: numpy.typing.NDArray[numpy.float64]
    flex_solimp: numpy.typing.NDArray[numpy.float64]
    flex_solmix: numpy.typing.NDArray[numpy.float64]
    flex_solref: numpy.typing.NDArray[numpy.float64]
    flex_stiffness: numpy.typing.NDArray[numpy.float64]
    flex_stiffnessadr: numpy.typing.NDArray[numpy.int32]
    flex_texcoord: numpy.typing.NDArray[numpy.float32]
    flex_texcoordadr: numpy.typing.NDArray[numpy.int32]
    flex_vert: numpy.typing.NDArray[numpy.float64]
    flex_vert0: numpy.typing.NDArray[numpy.float64]
    flex_vertadr: numpy.typing.NDArray[numpy.int32]
    flex_vertbodyid: numpy.typing.NDArray[numpy.int32]
    flex_vertedge: numpy.typing.NDArray[numpy.int32]
    flex_vertedgeadr: numpy.typing.NDArray[numpy.int32]
    flex_vertedgenum: numpy.typing.NDArray[numpy.int32]
    flex_vertmetric: numpy.typing.NDArray[numpy.float64]
    flex_vertnum: numpy.typing.NDArray[numpy.int32]
    flexedge_J_colind: numpy.typing.NDArray[numpy.int32]
    flexedge_J_rowadr: numpy.typing.NDArray[numpy.int32]
    flexedge_J_rownnz: numpy.typing.NDArray[numpy.int32]
    flexedge_invweight0: numpy.typing.NDArray[numpy.float64]
    flexedge_length0: numpy.typing.NDArray[numpy.float64]
    flexedge_rigid: numpy.typing.NDArray[numpy.bool]
    flexvert_J_colind: numpy.typing.NDArray[numpy.int32]
    flexvert_J_rowadr: numpy.typing.NDArray[numpy.int32]
    flexvert_J_rownnz: numpy.typing.NDArray[numpy.int32]
    geom_aabb: numpy.typing.NDArray[numpy.float64]
    geom_bodyid: numpy.typing.NDArray[numpy.int32]
    geom_conaffinity: numpy.typing.NDArray[numpy.int32]
    geom_condim: numpy.typing.NDArray[numpy.int32]
    geom_contype: numpy.typing.NDArray[numpy.int32]
    geom_dataid: numpy.typing.NDArray[numpy.int32]
    geom_fluid: numpy.typing.NDArray[numpy.float64]
    geom_friction: numpy.typing.NDArray[numpy.float64]
    geom_gap: numpy.typing.NDArray[numpy.float64]
    geom_group: numpy.typing.NDArray[numpy.int32]
    geom_margin: numpy.typing.NDArray[numpy.float64]
    geom_matid: numpy.typing.NDArray[numpy.int32]
    geom_plugin: numpy.typing.NDArray[numpy.int32]
    geom_pos: numpy.typing.NDArray[numpy.float64]
    geom_priority: numpy.typing.NDArray[numpy.int32]
    geom_quat: numpy.typing.NDArray[numpy.float64]
    geom_rbound: numpy.typing.NDArray[numpy.float64]
    geom_rgba: numpy.typing.NDArray[numpy.float32]
    geom_sameframe: numpy.typing.NDArray[numpy.uint8]
    geom_size: numpy.typing.NDArray[numpy.float64]
    geom_solimp: numpy.typing.NDArray[numpy.float64]
    geom_solmix: numpy.typing.NDArray[numpy.float64]
    geom_solref: numpy.typing.NDArray[numpy.float64]
    geom_type: numpy.typing.NDArray[numpy.int32]
    geom_user: numpy.typing.NDArray[numpy.float64]
    hfield_adr: numpy.typing.NDArray[numpy.int32]
    hfield_data: numpy.typing.NDArray[numpy.float32]
    hfield_ncol: numpy.typing.NDArray[numpy.int32]
    hfield_nrow: numpy.typing.NDArray[numpy.int32]
    hfield_pathadr: numpy.typing.NDArray[numpy.int32]
    hfield_size: numpy.typing.NDArray[numpy.float64]
    jnt_actfrclimited: numpy.typing.NDArray[numpy.bool]
    jnt_actfrcrange: numpy.typing.NDArray[numpy.float64]
    jnt_actgravcomp: numpy.typing.NDArray[numpy.bool]
    jnt_actuatorid: numpy.typing.NDArray[numpy.int32]
    jnt_axis: numpy.typing.NDArray[numpy.float64]
    jnt_bodyid: numpy.typing.NDArray[numpy.int32]
    jnt_dofadr: numpy.typing.NDArray[numpy.int32]
    jnt_group: numpy.typing.NDArray[numpy.int32]
    jnt_limited: numpy.typing.NDArray[numpy.bool]
    jnt_margin: numpy.typing.NDArray[numpy.float64]
    jnt_pos: numpy.typing.NDArray[numpy.float64]
    jnt_qposadr: numpy.typing.NDArray[numpy.int32]
    jnt_range: numpy.typing.NDArray[numpy.float64]
    jnt_solimp: numpy.typing.NDArray[numpy.float64]
    jnt_solref: numpy.typing.NDArray[numpy.float64]
    jnt_stiffness: numpy.typing.NDArray[numpy.float64]
    jnt_stiffnesspoly: numpy.typing.NDArray[numpy.float64]
    jnt_type: numpy.typing.NDArray[numpy.int32]
    jnt_user: numpy.typing.NDArray[numpy.float64]
    key_act: numpy.typing.NDArray[numpy.float64]
    key_ctrl: numpy.typing.NDArray[numpy.float64]
    key_mpos: numpy.typing.NDArray[numpy.float64]
    key_mquat: numpy.typing.NDArray[numpy.float64]
    key_qpos: numpy.typing.NDArray[numpy.float64]
    key_qvel: numpy.typing.NDArray[numpy.float64]
    key_time: numpy.typing.NDArray[numpy.float64]
    light_active: numpy.typing.NDArray[numpy.bool]
    light_ambient: numpy.typing.NDArray[numpy.float32]
    light_attenuation: numpy.typing.NDArray[numpy.float32]
    light_bodyid: numpy.typing.NDArray[numpy.int32]
    light_bulbradius: numpy.typing.NDArray[numpy.float32]
    light_castshadow: numpy.typing.NDArray[numpy.bool]
    light_cutoff: numpy.typing.NDArray[numpy.float32]
    light_diffuse: numpy.typing.NDArray[numpy.float32]
    light_dir: numpy.typing.NDArray[numpy.float64]
    light_dir0: numpy.typing.NDArray[numpy.float64]
    light_exponent: numpy.typing.NDArray[numpy.float32]
    light_intensity: numpy.typing.NDArray[numpy.float32]
    light_mode: numpy.typing.NDArray[numpy.int32]
    light_pos: numpy.typing.NDArray[numpy.float64]
    light_pos0: numpy.typing.NDArray[numpy.float64]
    light_poscom0: numpy.typing.NDArray[numpy.float64]
    light_range: numpy.typing.NDArray[numpy.float32]
    light_specular: numpy.typing.NDArray[numpy.float32]
    light_targetbodyid: numpy.typing.NDArray[numpy.int32]
    light_texid: numpy.typing.NDArray[numpy.int32]
    light_type: numpy.typing.NDArray[numpy.int32]
    mapD2M: numpy.typing.NDArray[numpy.int32]
    mapM2D: numpy.typing.NDArray[numpy.int32]
    mapM2M: numpy.typing.NDArray[numpy.int32]
    mat_emission: numpy.typing.NDArray[numpy.float32]
    mat_metallic: numpy.typing.NDArray[numpy.float32]
    mat_reflectance: numpy.typing.NDArray[numpy.float32]
    mat_rgba: numpy.typing.NDArray[numpy.float32]
    mat_roughness: numpy.typing.NDArray[numpy.float32]
    mat_shininess: numpy.typing.NDArray[numpy.float32]
    mat_specular: numpy.typing.NDArray[numpy.float32]
    mat_texid: numpy.typing.NDArray[numpy.int32]
    mat_texrepeat: numpy.typing.NDArray[numpy.float32]
    mat_texuniform: numpy.typing.NDArray[numpy.bool]
    mesh_bvhadr: numpy.typing.NDArray[numpy.int32]
    mesh_bvhnum: numpy.typing.NDArray[numpy.int32]
    mesh_face: numpy.typing.NDArray[numpy.int32]
    mesh_faceadr: numpy.typing.NDArray[numpy.int32]
    mesh_facenormal: numpy.typing.NDArray[numpy.int32]
    mesh_facenum: numpy.typing.NDArray[numpy.int32]
    mesh_facetexcoord: numpy.typing.NDArray[numpy.int32]
    mesh_graph: numpy.typing.NDArray[numpy.int32]
    mesh_graphadr: numpy.typing.NDArray[numpy.int32]
    mesh_normal: numpy.typing.NDArray[numpy.float32]
    mesh_normaladr: numpy.typing.NDArray[numpy.int32]
    mesh_normalnum: numpy.typing.NDArray[numpy.int32]
    mesh_octadr: numpy.typing.NDArray[numpy.int32]
    mesh_octnum: numpy.typing.NDArray[numpy.int32]
    mesh_pathadr: numpy.typing.NDArray[numpy.int32]
    mesh_polyadr: numpy.typing.NDArray[numpy.int32]
    mesh_polymap: numpy.typing.NDArray[numpy.int32]
    mesh_polymapadr: numpy.typing.NDArray[numpy.int32]
    mesh_polymapnum: numpy.typing.NDArray[numpy.int32]
    mesh_polynormal: numpy.typing.NDArray[numpy.float64]
    mesh_polynum: numpy.typing.NDArray[numpy.int32]
    mesh_polyvert: numpy.typing.NDArray[numpy.int32]
    mesh_polyvertadr: numpy.typing.NDArray[numpy.int32]
    mesh_polyvertnum: numpy.typing.NDArray[numpy.int32]
    mesh_pos: numpy.typing.NDArray[numpy.float64]
    mesh_quat: numpy.typing.NDArray[numpy.float64]
    mesh_scale: numpy.typing.NDArray[numpy.float64]
    mesh_texcoord: numpy.typing.NDArray[numpy.float32]
    mesh_texcoordadr: numpy.typing.NDArray[numpy.int32]
    mesh_texcoordnum: numpy.typing.NDArray[numpy.int32]
    mesh_vert: numpy.typing.NDArray[numpy.float32]
    mesh_vertadr: numpy.typing.NDArray[numpy.int32]
    mesh_vertnum: numpy.typing.NDArray[numpy.int32]
    name_actuatoradr: numpy.typing.NDArray[numpy.int32]
    name_bodyadr: numpy.typing.NDArray[numpy.int32]
    name_camadr: numpy.typing.NDArray[numpy.int32]
    name_eqadr: numpy.typing.NDArray[numpy.int32]
    name_excludeadr: numpy.typing.NDArray[numpy.int32]
    name_flexadr: numpy.typing.NDArray[numpy.int32]
    name_geomadr: numpy.typing.NDArray[numpy.int32]
    name_hfieldadr: numpy.typing.NDArray[numpy.int32]
    name_jntadr: numpy.typing.NDArray[numpy.int32]
    name_keyadr: numpy.typing.NDArray[numpy.int32]
    name_lightadr: numpy.typing.NDArray[numpy.int32]
    name_matadr: numpy.typing.NDArray[numpy.int32]
    name_meshadr: numpy.typing.NDArray[numpy.int32]
    name_numericadr: numpy.typing.NDArray[numpy.int32]
    name_pairadr: numpy.typing.NDArray[numpy.int32]
    name_pluginadr: numpy.typing.NDArray[numpy.int32]
    name_sensoradr: numpy.typing.NDArray[numpy.int32]
    name_siteadr: numpy.typing.NDArray[numpy.int32]
    name_skinadr: numpy.typing.NDArray[numpy.int32]
    name_tendonadr: numpy.typing.NDArray[numpy.int32]
    name_texadr: numpy.typing.NDArray[numpy.int32]
    name_textadr: numpy.typing.NDArray[numpy.int32]
    name_tupleadr: numpy.typing.NDArray[numpy.int32]
    names_map: numpy.typing.NDArray[numpy.int32]
    numeric_adr: numpy.typing.NDArray[numpy.int32]
    numeric_data: numpy.typing.NDArray[numpy.float64]
    numeric_size: numpy.typing.NDArray[numpy.int32]
    oct_aabb: numpy.typing.NDArray[numpy.float64]
    oct_child: numpy.typing.NDArray[numpy.int32]
    oct_coeff: numpy.typing.NDArray[numpy.float64]
    oct_depth: numpy.typing.NDArray[numpy.int32]
    pair_dim: numpy.typing.NDArray[numpy.int32]
    pair_friction: numpy.typing.NDArray[numpy.float64]
    pair_gap: numpy.typing.NDArray[numpy.float64]
    pair_geom1: numpy.typing.NDArray[numpy.int32]
    pair_geom2: numpy.typing.NDArray[numpy.int32]
    pair_margin: numpy.typing.NDArray[numpy.float64]
    pair_signature: numpy.typing.NDArray[numpy.int32]
    pair_solimp: numpy.typing.NDArray[numpy.float64]
    pair_solref: numpy.typing.NDArray[numpy.float64]
    pair_solreffriction: numpy.typing.NDArray[numpy.float64]
    plugin: numpy.typing.NDArray[numpy.int32]
    plugin_attr: numpy.typing.NDArray[numpy.int8]
    plugin_attradr: numpy.typing.NDArray[numpy.int32]
    plugin_stateadr: numpy.typing.NDArray[numpy.int32]
    plugin_statenum: numpy.typing.NDArray[numpy.int32]
    qpos0: numpy.typing.NDArray[numpy.float64]
    qpos_spring: numpy.typing.NDArray[numpy.float64]
    sensor_adr: numpy.typing.NDArray[numpy.int32]
    sensor_cutoff: numpy.typing.NDArray[numpy.float64]
    sensor_datatype: numpy.typing.NDArray[numpy.int32]
    sensor_delay: numpy.typing.NDArray[numpy.float64]
    sensor_dim: numpy.typing.NDArray[numpy.int32]
    sensor_history: numpy.typing.NDArray[numpy.int32]
    sensor_historyadr: numpy.typing.NDArray[numpy.int32]
    sensor_interval: numpy.typing.NDArray[numpy.float64]
    sensor_intprm: numpy.typing.NDArray[numpy.int32]
    sensor_needstage: numpy.typing.NDArray[numpy.int32]
    sensor_noise: numpy.typing.NDArray[numpy.float64]
    sensor_objid: numpy.typing.NDArray[numpy.int32]
    sensor_objtype: numpy.typing.NDArray[numpy.int32]
    sensor_plugin: numpy.typing.NDArray[numpy.int32]
    sensor_refid: numpy.typing.NDArray[numpy.int32]
    sensor_reftype: numpy.typing.NDArray[numpy.int32]
    sensor_type: numpy.typing.NDArray[numpy.int32]
    sensor_user: numpy.typing.NDArray[numpy.float64]
    site_bodyid: numpy.typing.NDArray[numpy.int32]
    site_group: numpy.typing.NDArray[numpy.int32]
    site_matid: numpy.typing.NDArray[numpy.int32]
    site_pos: numpy.typing.NDArray[numpy.float64]
    site_quat: numpy.typing.NDArray[numpy.float64]
    site_rgba: numpy.typing.NDArray[numpy.float32]
    site_sameframe: numpy.typing.NDArray[numpy.uint8]
    site_size: numpy.typing.NDArray[numpy.float64]
    site_type: numpy.typing.NDArray[numpy.int32]
    site_user: numpy.typing.NDArray[numpy.float64]
    skin_boneadr: numpy.typing.NDArray[numpy.int32]
    skin_bonebindpos: numpy.typing.NDArray[numpy.float32]
    skin_bonebindquat: numpy.typing.NDArray[numpy.float32]
    skin_bonebodyid: numpy.typing.NDArray[numpy.int32]
    skin_bonenum: numpy.typing.NDArray[numpy.int32]
    skin_bonevertadr: numpy.typing.NDArray[numpy.int32]
    skin_bonevertid: numpy.typing.NDArray[numpy.int32]
    skin_bonevertnum: numpy.typing.NDArray[numpy.int32]
    skin_bonevertweight: numpy.typing.NDArray[numpy.float32]
    skin_face: numpy.typing.NDArray[numpy.int32]
    skin_faceadr: numpy.typing.NDArray[numpy.int32]
    skin_facenum: numpy.typing.NDArray[numpy.int32]
    skin_group: numpy.typing.NDArray[numpy.int32]
    skin_inflate: numpy.typing.NDArray[numpy.float32]
    skin_matid: numpy.typing.NDArray[numpy.int32]
    skin_pathadr: numpy.typing.NDArray[numpy.int32]
    skin_rgba: numpy.typing.NDArray[numpy.float32]
    skin_texcoord: numpy.typing.NDArray[numpy.float32]
    skin_texcoordadr: numpy.typing.NDArray[numpy.int32]
    skin_vert: numpy.typing.NDArray[numpy.float32]
    skin_vertadr: numpy.typing.NDArray[numpy.int32]
    skin_vertnum: numpy.typing.NDArray[numpy.int32]
    ten_J_colind: numpy.typing.NDArray[numpy.int32]
    ten_J_rowadr: numpy.typing.NDArray[numpy.int32]
    ten_J_rownnz: numpy.typing.NDArray[numpy.int32]
    tendon_actfrclimited: numpy.typing.NDArray[numpy.bool]
    tendon_actfrcrange: numpy.typing.NDArray[numpy.float64]
    tendon_actuatorid: numpy.typing.NDArray[numpy.int32]
    tendon_adr: numpy.typing.NDArray[numpy.int32]
    tendon_armature: numpy.typing.NDArray[numpy.float64]
    tendon_damping: numpy.typing.NDArray[numpy.float64]
    tendon_dampingpoly: numpy.typing.NDArray[numpy.float64]
    tendon_frictionloss: numpy.typing.NDArray[numpy.float64]
    tendon_group: numpy.typing.NDArray[numpy.int32]
    tendon_invweight0: numpy.typing.NDArray[numpy.float64]
    tendon_length0: numpy.typing.NDArray[numpy.float64]
    tendon_lengthspring: numpy.typing.NDArray[numpy.float64]
    tendon_limited: numpy.typing.NDArray[numpy.bool]
    tendon_margin: numpy.typing.NDArray[numpy.float64]
    tendon_matid: numpy.typing.NDArray[numpy.int32]
    tendon_num: numpy.typing.NDArray[numpy.int32]
    tendon_range: numpy.typing.NDArray[numpy.float64]
    tendon_rgba: numpy.typing.NDArray[numpy.float32]
    tendon_solimp_fri: numpy.typing.NDArray[numpy.float64]
    tendon_solimp_lim: numpy.typing.NDArray[numpy.float64]
    tendon_solref_fri: numpy.typing.NDArray[numpy.float64]
    tendon_solref_lim: numpy.typing.NDArray[numpy.float64]
    tendon_stiffness: numpy.typing.NDArray[numpy.float64]
    tendon_stiffnesspoly: numpy.typing.NDArray[numpy.float64]
    tendon_treeid: numpy.typing.NDArray[numpy.int32]
    tendon_treenum: numpy.typing.NDArray[numpy.int32]
    tendon_user: numpy.typing.NDArray[numpy.float64]
    tendon_width: numpy.typing.NDArray[numpy.float64]
    tex_adr: numpy.typing.NDArray[numpy.int64]
    tex_colorspace: numpy.typing.NDArray[numpy.int32]
    tex_data: numpy.typing.NDArray[numpy.uint8]
    tex_height: numpy.typing.NDArray[numpy.int32]
    tex_nchannel: numpy.typing.NDArray[numpy.int32]
    tex_pathadr: numpy.typing.NDArray[numpy.int32]
    tex_type: numpy.typing.NDArray[numpy.int32]
    tex_width: numpy.typing.NDArray[numpy.int32]
    text_adr: numpy.typing.NDArray[numpy.int32]
    text_size: numpy.typing.NDArray[numpy.int32]
    tree_bodyadr: numpy.typing.NDArray[numpy.int32]
    tree_bodynum: numpy.typing.NDArray[numpy.int32]
    tree_dofadr: numpy.typing.NDArray[numpy.int32]
    tree_dofnum: numpy.typing.NDArray[numpy.int32]
    tree_sleep_policy: numpy.typing.NDArray[numpy.int32]
    tuple_adr: numpy.typing.NDArray[numpy.int32]
    tuple_objid: numpy.typing.NDArray[numpy.int32]
    tuple_objprm: numpy.typing.NDArray[numpy.float64]
    tuple_objtype: numpy.typing.NDArray[numpy.int32]
    tuple_size: numpy.typing.NDArray[numpy.int32]
    wrap_objid: numpy.typing.NDArray[numpy.int32]
    wrap_prm: numpy.typing.NDArray[numpy.float64]
    wrap_type: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    def actuator(self, *args, **kwargs): ...
    def bind_scalar(self, *args, **kwargs): ...
    def body(self, *args, **kwargs): ...
    def cam(self, *args, **kwargs): ...
    def camera(self, *args, **kwargs): ...
    def eq(self, *args, **kwargs): ...
    def equality(self, *args, **kwargs): ...
    def exclude(self, *args, **kwargs): ...
    @staticmethod
    def from_binary_path(filename: str, assets: collections.abc.Mapping[str, bytes] | None = ..., vfs=...) -> MjModel: ...
    @staticmethod
    def from_xml_path(filename: str, assets: collections.abc.Mapping[str, bytes] | None = ..., vfs=...) -> MjModel: ...
    @staticmethod
    def from_xml_string(xml: str, assets: collections.abc.Mapping[str, bytes] | None = ..., vfs=...) -> MjModel: ...
    def geom(self, *args, **kwargs): ...
    def hfield(self, *args, **kwargs): ...
    def jnt(self, *args, **kwargs): ...
    def joint(self, *args, **kwargs): ...
    def key(self, *args, **kwargs): ...
    def keyframe(self, *args, **kwargs): ...
    def light(self, *args, **kwargs): ...
    def mat(self, *args, **kwargs): ...
    def material(self, *args, **kwargs): ...
    def mesh(self, *args, **kwargs): ...
    def numeric(self, *args, **kwargs): ...
    def pair(self, *args, **kwargs): ...
    def sensor(self, *args, **kwargs): ...
    def site(self, *args, **kwargs): ...
    def skin(self, *args, **kwargs): ...
    def tendon(self, *args, **kwargs): ...
    def tex(self, *args, **kwargs): ...
    def texture(self, *args, **kwargs): ...
    def tuple(self, *args, **kwargs): ...
    def __copy__(self) -> MjModel: ...
    def __deepcopy__(self, arg0: dict) -> MjModel: ...
    @property
    def nB(self) -> int: ...
    @property
    def nC(self) -> int: ...
    @property
    def nD(self) -> int: ...
    @property
    def nJfe(self) -> int: ...
    @property
    def nJfv(self) -> int: ...
    @property
    def nJmom(self) -> int: ...
    @property
    def nJten(self) -> int: ...
    @property
    def nM(self) -> int: ...
    @property
    def na(self) -> int: ...
    @property
    def names(self) -> bytes: ...
    @property
    def narena(self) -> int: ...
    @property
    def nbody(self) -> int: ...
    @property
    def nbuffer(self) -> int: ...
    @property
    def nbvh(self) -> int: ...
    @property
    def nbvhdynamic(self) -> int: ...
    @property
    def nbvhstatic(self) -> int: ...
    @property
    def ncam(self) -> int: ...
    @property
    def nconmax(self) -> int: ...
    @property
    def nemax(self) -> int: ...
    @property
    def neq(self) -> int: ...
    @property
    def nexclude(self) -> int: ...
    @property
    def nflex(self) -> int: ...
    @property
    def nflexbending(self) -> int: ...
    @property
    def nflexedge(self) -> int: ...
    @property
    def nflexelem(self) -> int: ...
    @property
    def nflexelemdata(self) -> int: ...
    @property
    def nflexelemedge(self) -> int: ...
    @property
    def nflexevpair(self) -> int: ...
    @property
    def nflexnode(self) -> int: ...
    @property
    def nflexshelldata(self) -> int: ...
    @property
    def nflexstiffness(self) -> int: ...
    @property
    def nflextexcoord(self) -> int: ...
    @property
    def nflexvert(self) -> int: ...
    @property
    def ngeom(self) -> int: ...
    @property
    def ngravcomp(self) -> int: ...
    @property
    def nhfield(self) -> int: ...
    @property
    def nhfielddata(self) -> int: ...
    @property
    def nhistory(self) -> int: ...
    @property
    def njmax(self) -> int: ...
    @property
    def njnt(self) -> int: ...
    @property
    def nkey(self) -> int: ...
    @property
    def nlight(self) -> int: ...
    @property
    def nmat(self) -> int: ...
    @property
    def nmesh(self) -> int: ...
    @property
    def nmeshface(self) -> int: ...
    @property
    def nmeshgraph(self) -> int: ...
    @property
    def nmeshnormal(self) -> int: ...
    @property
    def nmeshpoly(self) -> int: ...
    @property
    def nmeshpolymap(self) -> int: ...
    @property
    def nmeshpolyvert(self) -> int: ...
    @property
    def nmeshtexcoord(self) -> int: ...
    @property
    def nmeshvert(self) -> int: ...
    @property
    def nmocap(self) -> int: ...
    @property
    def nnames(self) -> int: ...
    @property
    def nnames_map(self) -> int: ...
    @property
    def nnumeric(self) -> int: ...
    @property
    def nnumericdata(self) -> int: ...
    @property
    def noct(self) -> int: ...
    @property
    def npair(self) -> int: ...
    @property
    def npaths(self) -> int: ...
    @property
    def nplugin(self) -> int: ...
    @property
    def npluginattr(self) -> int: ...
    @property
    def npluginstate(self) -> int: ...
    @property
    def nq(self) -> int: ...
    @property
    def nsensor(self) -> int: ...
    @property
    def nsensordata(self) -> int: ...
    @property
    def nsite(self) -> int: ...
    @property
    def nskin(self) -> int: ...
    @property
    def nskinbone(self) -> int: ...
    @property
    def nskinbonevert(self) -> int: ...
    @property
    def nskinface(self) -> int: ...
    @property
    def nskintexvert(self) -> int: ...
    @property
    def nskinvert(self) -> int: ...
    @property
    def ntendon(self) -> int: ...
    @property
    def ntex(self) -> int: ...
    @property
    def ntexdata(self) -> int: ...
    @property
    def ntext(self) -> int: ...
    @property
    def ntextdata(self) -> int: ...
    @property
    def ntree(self) -> int: ...
    @property
    def ntuple(self) -> int: ...
    @property
    def ntupledata(self) -> int: ...
    @property
    def nu(self) -> int: ...
    @property
    def nuser_actuator(self) -> int: ...
    @property
    def nuser_body(self) -> int: ...
    @property
    def nuser_cam(self) -> int: ...
    @property
    def nuser_geom(self) -> int: ...
    @property
    def nuser_jnt(self) -> int: ...
    @property
    def nuser_sensor(self) -> int: ...
    @property
    def nuser_site(self) -> int: ...
    @property
    def nuser_tendon(self) -> int: ...
    @property
    def nuserdata(self) -> int: ...
    @property
    def nv(self) -> int: ...
    @property
    def nwrap(self) -> int: ...
    @property
    def opt(self) -> MjOption: ...
    @property
    def paths(self) -> bytes: ...
    @property
    def signature(self) -> int: ...
    @property
    def stat(self): ...
    @property
    def text_data(self) -> bytes: ...
    @property
    def vis(self) -> MjVisual: ...

class MjOption:
    ccd_iterations: int
    ccd_tolerance: float
    cone: int
    density: float
    disableactuator: int
    disableflags: int
    enableflags: int
    gravity: numpy.typing.NDArray[numpy.float64]
    impratio: float
    integrator: int
    iterations: int
    jacobian: int
    ls_iterations: int
    ls_tolerance: float
    magnetic: numpy.typing.NDArray[numpy.float64]
    noslip_iterations: int
    noslip_tolerance: float
    o_friction: numpy.typing.NDArray[numpy.float64]
    o_margin: float
    o_solimp: numpy.typing.NDArray[numpy.float64]
    o_solref: numpy.typing.NDArray[numpy.float64]
    sdf_initpoints: int
    sdf_iterations: int
    sleep_tolerance: float
    solver: int
    timestep: float
    tolerance: float
    viscosity: float
    wind: numpy.typing.NDArray[numpy.float64]
    def __init__(self) -> None: ...
    def __copy__(self) -> MjOption: ...
    def __deepcopy__(self, arg0: dict) -> MjOption: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjPreContact:
    dist: float
    normal: numpy.typing.NDArray[numpy.float64]
    pos: numpy.typing.NDArray[numpy.float64]
    tangent: numpy.typing.NDArray[numpy.float64]
    def __init__(self) -> None: ...
    def __copy__(self) -> MjPreContact: ...
    def __deepcopy__(self, arg0: dict) -> MjPreContact: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjSolverStat:
    gradient: float
    improvement: float
    lineslope: float
    nactive: int
    nchange: int
    neval: int
    nupdate: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjSolverStat: ...
    def __deepcopy__(self, arg0: dict) -> MjSolverStat: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjStatistic:
    center: numpy.typing.NDArray[numpy.float64]
    extent: float
    meaninertia: float
    meanmass: float
    meansize: float
    def __init__(self) -> None: ...
    def __copy__(self) -> MjStatistic: ...
    def __deepcopy__(self, arg0: dict) -> MjStatistic: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjTimerStat:
    duration: float
    number: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjTimerStat: ...
    def __deepcopy__(self, arg0: dict) -> MjTimerStat: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjVisual:
    class Global:
        azimuth: float
        bvactive: int
        cameraid: int
        elevation: float
        ellipsoidinertia: int
        fovy: float
        glow: float
        ipd: float
        linewidth: float
        offheight: int
        offwidth: int
        orthographic: int
        realtime: float
        def __init__(self, *args, **kwargs) -> None: ...
        def __copy__(self) -> MjVisual.Global: ...
        def __deepcopy__(self, arg0: dict) -> MjVisual.Global: ...
        def __eq__(self, arg0: object) -> bool: ...

    class Headlight:
        active: int
        ambient: numpy.typing.NDArray[numpy.float32]
        diffuse: numpy.typing.NDArray[numpy.float32]
        specular: numpy.typing.NDArray[numpy.float32]
        def __init__(self, *args, **kwargs) -> None: ...
        def __copy__(self) -> MjVisual.Headlight: ...
        def __deepcopy__(self, arg0: dict) -> MjVisual.Headlight: ...
        def __eq__(self, arg0: object) -> bool: ...

    class Map:
        actuatortendon: float
        alpha: float
        fogend: float
        fogstart: float
        force: float
        haze: float
        shadowclip: float
        shadowscale: float
        stiffness: float
        stiffnessrot: float
        torque: float
        zfar: float
        znear: float
        def __init__(self, *args, **kwargs) -> None: ...
        def __copy__(self) -> MjVisual.Map: ...
        def __deepcopy__(self, arg0: dict) -> MjVisual.Map: ...
        def __eq__(self, arg0: object) -> bool: ...

    class Quality:
        numquads: int
        numslices: int
        numstacks: int
        offsamples: int
        shadowsize: int
        def __init__(self, *args, **kwargs) -> None: ...
        def __copy__(self) -> MjVisual.Quality: ...
        def __deepcopy__(self, arg0: dict) -> MjVisual.Quality: ...
        def __eq__(self, arg0: object) -> bool: ...

    class Rgba:
        actuator: numpy.typing.NDArray[numpy.float32]
        actuatornegative: numpy.typing.NDArray[numpy.float32]
        actuatorpositive: numpy.typing.NDArray[numpy.float32]
        bv: numpy.typing.NDArray[numpy.float32]
        bvactive: numpy.typing.NDArray[numpy.float32]
        camera: numpy.typing.NDArray[numpy.float32]
        com: numpy.typing.NDArray[numpy.float32]
        connect: numpy.typing.NDArray[numpy.float32]
        constraint: numpy.typing.NDArray[numpy.float32]
        contactforce: numpy.typing.NDArray[numpy.float32]
        contactfriction: numpy.typing.NDArray[numpy.float32]
        contactgap: numpy.typing.NDArray[numpy.float32]
        contactpoint: numpy.typing.NDArray[numpy.float32]
        contacttorque: numpy.typing.NDArray[numpy.float32]
        crankbroken: numpy.typing.NDArray[numpy.float32]
        fog: numpy.typing.NDArray[numpy.float32]
        force: numpy.typing.NDArray[numpy.float32]
        frustum: numpy.typing.NDArray[numpy.float32]
        haze: numpy.typing.NDArray[numpy.float32]
        inertia: numpy.typing.NDArray[numpy.float32]
        joint: numpy.typing.NDArray[numpy.float32]
        light: numpy.typing.NDArray[numpy.float32]
        rangefinder: numpy.typing.NDArray[numpy.float32]
        selectpoint: numpy.typing.NDArray[numpy.float32]
        slidercrank: numpy.typing.NDArray[numpy.float32]
        def __init__(self, *args, **kwargs) -> None: ...
        def __copy__(self) -> MjVisual.Rgba: ...
        def __deepcopy__(self, arg0: dict) -> MjVisual.Rgba: ...
        def __eq__(self, arg0: object) -> bool: ...

    class Scale:
        actuatorlength: float
        actuatorwidth: float
        camera: float
        com: float
        connect: float
        constraint: float
        contactheight: float
        contactwidth: float
        forcewidth: float
        framelength: float
        framewidth: float
        frustum: float
        jointlength: float
        jointwidth: float
        light: float
        selectpoint: float
        slidercrank: float
        def __init__(self, *args, **kwargs) -> None: ...
        def __copy__(self) -> MjVisual.Scale: ...
        def __deepcopy__(self, arg0: dict) -> MjVisual.Scale: ...
        def __eq__(self, arg0: object) -> bool: ...
    def __init__(self, *args, **kwargs) -> None: ...
    def __copy__(self) -> MjVisual: ...
    def __deepcopy__(self, arg0: dict) -> MjVisual: ...
    def __eq__(self, arg0: object) -> bool: ...
    @property
    def global_(self) -> MjVisual.Global: ...
    @property
    def headlight(self) -> MjVisual.Headlight: ...
    @property
    def map(self) -> MjVisual.Map: ...
    @property
    def quality(self) -> MjVisual.Quality: ...
    @property
    def rgba(self) -> MjVisual.Rgba: ...
    @property
    def scale(self) -> MjVisual.Scale: ...

class MjWarningStat:
    lastinfo: int
    number: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjWarningStat: ...
    def __deepcopy__(self, arg0: dict) -> MjWarningStat: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjrRect:
    bottom: int
    height: int
    left: int
    width: int
    def __init__(self, left: typing.SupportsInt | typing.SupportsIndex, bottom: typing.SupportsInt | typing.SupportsIndex, width: typing.SupportsInt | typing.SupportsIndex, height: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def __copy__(self) -> MjrRect: ...
    def __deepcopy__(self, arg0: dict) -> MjrRect: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjrVertexAttribute:
    type: int
    usage: int
    def __init__(self, usage: typing.SupportsInt | typing.SupportsIndex = ..., type: typing.SupportsInt | typing.SupportsIndex = ...) -> None: ...
    def __copy__(self) -> MjrVertexAttribute: ...
    def __deepcopy__(self, arg0: dict) -> MjrVertexAttribute: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvCamera:
    azimuth: float
    distance: float
    elevation: float
    fixedcamid: int
    lookat: numpy.typing.NDArray[numpy.float64]
    orthographic: int
    trackbodyid: int
    type: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvCamera: ...
    def __deepcopy__(self, arg0: dict) -> MjvCamera: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvFigure:
    figurergba: numpy.typing.NDArray[numpy.float32]
    flg_barplot: int
    flg_extend: int
    flg_legend: int
    flg_selection: int
    flg_symmetric: int
    flg_ticklabel: numpy.typing.NDArray[numpy.int32]
    gridrgb: numpy.typing.NDArray[numpy.float32]
    gridsize: numpy.typing.NDArray[numpy.int32]
    gridwidth: float
    highlight: numpy.typing.NDArray[numpy.int32]
    highlightid: int
    legendoffset: int
    legendrgba: numpy.typing.NDArray[numpy.float32]
    linedata: numpy.typing.NDArray[numpy.float32]
    linepnt: numpy.typing.NDArray[numpy.int32]
    linergb: numpy.typing.NDArray[numpy.float32]
    linewidth: float
    minwidth: str
    panergba: numpy.typing.NDArray[numpy.float32]
    range: numpy.typing.NDArray[numpy.float32]
    selection: float
    subplot: int
    textrgb: numpy.typing.NDArray[numpy.float32]
    title: str
    xaxisdata: numpy.typing.NDArray[numpy.float32]
    xaxispixel: numpy.typing.NDArray[numpy.int32]
    xformat: str
    xlabel: str
    yaxisdata: numpy.typing.NDArray[numpy.float32]
    yaxispixel: numpy.typing.NDArray[numpy.int32]
    yformat: str
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvFigure: ...
    def __deepcopy__(self, arg0: dict) -> MjvFigure: ...
    @property
    def linename(self) -> numpy.ndarray: ...

class MjvGLCamera:
    forward: numpy.typing.NDArray[numpy.float32]
    frustum_bottom: float
    frustum_center: float
    frustum_far: float
    frustum_near: float
    frustum_top: float
    frustum_width: float
    orthographic: int
    pos: numpy.typing.NDArray[numpy.float32]
    up: numpy.typing.NDArray[numpy.float32]
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvGLCamera: ...
    def __deepcopy__(self, arg0: dict) -> MjvGLCamera: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvGeom:
    camdist: float
    category: int
    dataid: int
    emission: float
    label: str
    mat: numpy.typing.NDArray[numpy.float32]
    matid: int
    modelrbound: float
    objid: int
    objtype: int
    pos: numpy.typing.NDArray[numpy.float32]
    reflectance: float
    rgba: numpy.typing.NDArray[numpy.float32]
    segid: int
    shininess: float
    size: numpy.typing.NDArray[numpy.float32]
    specular: float
    texcoord: int
    transparent: int
    type: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvGeom: ...
    def __deepcopy__(self, arg0: dict) -> MjvGeom: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvLight:
    ambient: numpy.typing.NDArray[numpy.float32]
    attenuation: numpy.typing.NDArray[numpy.float32]
    bulbradius: float
    castshadow: int
    cutoff: float
    diffuse: numpy.typing.NDArray[numpy.float32]
    dir: numpy.typing.NDArray[numpy.float32]
    exponent: float
    headlight: int
    id: int
    intensity: float
    pos: numpy.typing.NDArray[numpy.float32]
    range: float
    specular: numpy.typing.NDArray[numpy.float32]
    texid: int
    type: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvLight: ...
    def __deepcopy__(self, arg0: dict) -> MjvLight: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvOption:
    actuatorgroup: numpy.typing.NDArray[numpy.uint8]
    bvh_depth: int
    flags: numpy.typing.NDArray[numpy.uint8]
    flex_layer: int
    flexgroup: numpy.typing.NDArray[numpy.uint8]
    frame: int
    geomgroup: numpy.typing.NDArray[numpy.uint8]
    jointgroup: numpy.typing.NDArray[numpy.uint8]
    label: int
    sitegroup: numpy.typing.NDArray[numpy.uint8]
    skingroup: numpy.typing.NDArray[numpy.uint8]
    tendongroup: numpy.typing.NDArray[numpy.uint8]
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvOption: ...
    def __deepcopy__(self, arg0: dict) -> MjvOption: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvPerturb:
    active: int
    active2: int
    flexselect: int
    localmass: float
    localpos: numpy.typing.NDArray[numpy.float64]
    refpos: numpy.typing.NDArray[numpy.float64]
    refquat: numpy.typing.NDArray[numpy.float64]
    refselpos: numpy.typing.NDArray[numpy.float64]
    scale: float
    select: int
    skinselect: int
    def __init__(self) -> None: ...
    def __copy__(self) -> MjvPerturb: ...
    def __deepcopy__(self, arg0: dict) -> MjvPerturb: ...
    def __eq__(self, arg0: object) -> bool: ...

class MjvScene:
    enabletransform: int
    flags: numpy.typing.NDArray[numpy.uint8]
    flexedge: numpy.typing.NDArray[numpy.int32]
    flexedgeadr: numpy.typing.NDArray[numpy.int32]
    flexedgenum: numpy.typing.NDArray[numpy.int32]
    flexedgeopt: int
    flexface: numpy.typing.NDArray[numpy.float32]
    flexfaceadr: numpy.typing.NDArray[numpy.int32]
    flexfacenum: numpy.typing.NDArray[numpy.int32]
    flexfaceopt: int
    flexfaceused: numpy.typing.NDArray[numpy.int32]
    flexnormal: numpy.typing.NDArray[numpy.float32]
    flexskinopt: int
    flextexcoord: numpy.typing.NDArray[numpy.float32]
    flexvert: numpy.typing.NDArray[numpy.float32]
    flexvertadr: numpy.typing.NDArray[numpy.int32]
    flexvertnum: numpy.typing.NDArray[numpy.int32]
    flexvertopt: int
    framergb: numpy.typing.NDArray[numpy.float32]
    framewidth: int
    geomorder: numpy.typing.NDArray[numpy.int32]
    maxgeom: int
    nflex: int
    ngeom: int
    nlight: int
    nskin: int
    rotate: numpy.typing.NDArray[numpy.float32]
    scale: float
    skinfacenum: numpy.typing.NDArray[numpy.int32]
    skinnormal: numpy.typing.NDArray[numpy.float32]
    skinvert: numpy.typing.NDArray[numpy.float32]
    skinvertadr: numpy.typing.NDArray[numpy.int32]
    skinvertnum: numpy.typing.NDArray[numpy.int32]
    status: int
    stereo: int
    translate: numpy.typing.NDArray[numpy.float32]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: MjModel, maxgeom: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def __copy__(self) -> MjvScene: ...
    def __deepcopy__(self, arg0: dict) -> MjvScene: ...
    @property
    def camera(self) -> tuple: ...
    @property
    def geoms(self) -> tuple: ...
    @property
    def lights(self) -> tuple: ...

class _MjContactList:
    def __init__(self, *args, **kwargs) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    @overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> MjContact: ...
    @overload
    def __getitem__(self, arg0: slice) -> _MjContactList: ...
    def __len__(self) -> int: ...
    @property
    def H(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def dim(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def dist(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def efc_address(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def elem(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def exclude(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def flex(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def frame(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def friction(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def geom(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def geom1(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def geom2(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def includemargin(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def mu(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def pos(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def solimp(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def solref(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def solreffriction(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def vert(self) -> numpy.typing.NDArray[numpy.int32]: ...

class _MjDataActuatorViews:
    ctrl: numpy.typing.NDArray[numpy.float64]
    force: numpy.typing.NDArray[numpy.float64]
    length: numpy.typing.NDArray[numpy.float64]
    moment: numpy.typing.NDArray[numpy.float64]
    velocity: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataBodyViews:
    cacc: numpy.typing.NDArray[numpy.float64]
    cfrc_ext: numpy.typing.NDArray[numpy.float64]
    cfrc_int: numpy.typing.NDArray[numpy.float64]
    cinert: numpy.typing.NDArray[numpy.float64]
    crb: numpy.typing.NDArray[numpy.float64]
    cvel: numpy.typing.NDArray[numpy.float64]
    subtree_angmom: numpy.typing.NDArray[numpy.float64]
    subtree_com: numpy.typing.NDArray[numpy.float64]
    subtree_linvel: numpy.typing.NDArray[numpy.float64]
    xfrc_applied: numpy.typing.NDArray[numpy.float64]
    ximat: numpy.typing.NDArray[numpy.float64]
    xipos: numpy.typing.NDArray[numpy.float64]
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    xquat: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataCameraViews:
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataGeomViews:
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataJointViews:
    cdof: numpy.typing.NDArray[numpy.float64]
    cdof_dot: numpy.typing.NDArray[numpy.float64]
    qLDiagInv: numpy.typing.NDArray[numpy.float64]
    qacc: numpy.typing.NDArray[numpy.float64]
    qacc_smooth: numpy.typing.NDArray[numpy.float64]
    qacc_warmstart: numpy.typing.NDArray[numpy.float64]
    qfrc_actuator: numpy.typing.NDArray[numpy.float64]
    qfrc_applied: numpy.typing.NDArray[numpy.float64]
    qfrc_bias: numpy.typing.NDArray[numpy.float64]
    qfrc_constraint: numpy.typing.NDArray[numpy.float64]
    qfrc_inverse: numpy.typing.NDArray[numpy.float64]
    qfrc_passive: numpy.typing.NDArray[numpy.float64]
    qfrc_smooth: numpy.typing.NDArray[numpy.float64]
    qpos: numpy.typing.NDArray[numpy.float64]
    qvel: numpy.typing.NDArray[numpy.float64]
    xanchor: numpy.typing.NDArray[numpy.float64]
    xaxis: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataLightViews:
    xdir: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataSensorViews:
    data: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataSiteViews:
    xmat: numpy.typing.NDArray[numpy.float64]
    xpos: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjDataTendonViews:
    length: numpy.typing.NDArray[numpy.float64]
    velocity: numpy.typing.NDArray[numpy.float64]
    wrapadr: numpy.typing.NDArray[numpy.int32]
    wrapnum: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelActuatorViews:
    acc0: numpy.typing.NDArray[numpy.float64]
    actadr: numpy.typing.NDArray[numpy.int32]
    actlimited: numpy.typing.NDArray[numpy.uint8]
    actnum: numpy.typing.NDArray[numpy.int32]
    actrange: numpy.typing.NDArray[numpy.float64]
    biasprm: numpy.typing.NDArray[numpy.float64]
    biastype: numpy.typing.NDArray[numpy.int32]
    cranklength: numpy.typing.NDArray[numpy.float64]
    ctrllimited: numpy.typing.NDArray[numpy.uint8]
    ctrlrange: numpy.typing.NDArray[numpy.float64]
    dynprm: numpy.typing.NDArray[numpy.float64]
    dyntype: numpy.typing.NDArray[numpy.int32]
    forcelimited: numpy.typing.NDArray[numpy.uint8]
    forcerange: numpy.typing.NDArray[numpy.float64]
    gainprm: numpy.typing.NDArray[numpy.float64]
    gaintype: numpy.typing.NDArray[numpy.int32]
    gear: numpy.typing.NDArray[numpy.float64]
    group: numpy.typing.NDArray[numpy.int32]
    length0: numpy.typing.NDArray[numpy.float64]
    lengthrange: numpy.typing.NDArray[numpy.float64]
    trnid: numpy.typing.NDArray[numpy.int32]
    trntype: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelBodyViews:
    dofadr: numpy.typing.NDArray[numpy.int32]
    dofnum: numpy.typing.NDArray[numpy.int32]
    geomadr: numpy.typing.NDArray[numpy.int32]
    geomnum: numpy.typing.NDArray[numpy.int32]
    inertia: numpy.typing.NDArray[numpy.float64]
    invweight0: numpy.typing.NDArray[numpy.float64]
    ipos: numpy.typing.NDArray[numpy.float64]
    iquat: numpy.typing.NDArray[numpy.float64]
    jntadr: numpy.typing.NDArray[numpy.int32]
    jntnum: numpy.typing.NDArray[numpy.int32]
    mass: numpy.typing.NDArray[numpy.float64]
    mocapid: numpy.typing.NDArray[numpy.int32]
    parentid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    quat: numpy.typing.NDArray[numpy.float64]
    rootid: numpy.typing.NDArray[numpy.int32]
    sameframe: numpy.typing.NDArray[numpy.uint8]
    simple: numpy.typing.NDArray[numpy.uint8]
    subtreemass: numpy.typing.NDArray[numpy.float64]
    user: numpy.typing.NDArray[numpy.float64]
    weldid: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelCameraViews:
    bodyid: numpy.typing.NDArray[numpy.int32]
    fovy: numpy.typing.NDArray[numpy.float64]
    ipd: numpy.typing.NDArray[numpy.float64]
    mat0: numpy.typing.NDArray[numpy.float64]
    mode: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    pos0: numpy.typing.NDArray[numpy.float64]
    poscom0: numpy.typing.NDArray[numpy.float64]
    quat: numpy.typing.NDArray[numpy.float64]
    targetbodyid: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelEqualityViews:
    active0: numpy.typing.NDArray[numpy.uint8]
    data: numpy.typing.NDArray[numpy.float64]
    obj1id: numpy.typing.NDArray[numpy.int32]
    obj2id: numpy.typing.NDArray[numpy.int32]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelExcludeViews:
    signature: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelGeomViews:
    bodyid: numpy.typing.NDArray[numpy.int32]
    conaffinity: numpy.typing.NDArray[numpy.int32]
    condim: numpy.typing.NDArray[numpy.int32]
    contype: numpy.typing.NDArray[numpy.int32]
    dataid: numpy.typing.NDArray[numpy.int32]
    friction: numpy.typing.NDArray[numpy.float64]
    gap: numpy.typing.NDArray[numpy.float64]
    group: numpy.typing.NDArray[numpy.int32]
    margin: numpy.typing.NDArray[numpy.float64]
    matid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    priority: numpy.typing.NDArray[numpy.int32]
    quat: numpy.typing.NDArray[numpy.float64]
    rbound: numpy.typing.NDArray[numpy.float64]
    rgba: numpy.typing.NDArray[numpy.float32]
    sameframe: numpy.typing.NDArray[numpy.uint8]
    size: numpy.typing.NDArray[numpy.float64]
    solimp: numpy.typing.NDArray[numpy.float64]
    solmix: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelHfieldViews:
    adr: numpy.typing.NDArray[numpy.int32]
    data: numpy.typing.NDArray[numpy.float32]
    ncol: numpy.typing.NDArray[numpy.int32]
    nrow: numpy.typing.NDArray[numpy.int32]
    size: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelJointViews:
    M0: numpy.typing.NDArray[numpy.float64]
    Madr: numpy.typing.NDArray[numpy.int32]
    armature: numpy.typing.NDArray[numpy.float64]
    axis: numpy.typing.NDArray[numpy.float64]
    bodyid: numpy.typing.NDArray[numpy.int32]
    damping: numpy.typing.NDArray[numpy.float64]
    dampingpoly: numpy.typing.NDArray[numpy.float64]
    dofadr: numpy.typing.NDArray[numpy.int32]
    frictionloss: numpy.typing.NDArray[numpy.float64]
    group: numpy.typing.NDArray[numpy.int32]
    invweight0: numpy.typing.NDArray[numpy.float64]
    jntid: numpy.typing.NDArray[numpy.int32]
    limited: numpy.typing.NDArray[numpy.uint8]
    margin: numpy.typing.NDArray[numpy.float64]
    parentid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    qpos0: numpy.typing.NDArray[numpy.float64]
    qpos_spring: numpy.typing.NDArray[numpy.float64]
    qposadr: numpy.typing.NDArray[numpy.int32]
    range: numpy.typing.NDArray[numpy.float64]
    simplenum: numpy.typing.NDArray[numpy.int32]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    stiffness: numpy.typing.NDArray[numpy.float64]
    stiffnesspoly: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelKeyframeViews:
    act: numpy.typing.NDArray[numpy.float64]
    ctrl: numpy.typing.NDArray[numpy.float64]
    mpos: numpy.typing.NDArray[numpy.float64]
    mquat: numpy.typing.NDArray[numpy.float64]
    qpos: numpy.typing.NDArray[numpy.float64]
    qvel: numpy.typing.NDArray[numpy.float64]
    time: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelLightViews:
    active: numpy.typing.NDArray[numpy.uint8]
    ambient: numpy.typing.NDArray[numpy.float32]
    attenuation: numpy.typing.NDArray[numpy.float32]
    bodyid: numpy.typing.NDArray[numpy.int32]
    castshadow: numpy.typing.NDArray[numpy.uint8]
    cutoff: numpy.typing.NDArray[numpy.float32]
    diffuse: numpy.typing.NDArray[numpy.float32]
    dir: numpy.typing.NDArray[numpy.float64]
    dir0: numpy.typing.NDArray[numpy.float64]
    exponent: numpy.typing.NDArray[numpy.float32]
    mode: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    pos0: numpy.typing.NDArray[numpy.float64]
    poscom0: numpy.typing.NDArray[numpy.float64]
    specular: numpy.typing.NDArray[numpy.float32]
    targetbodyid: numpy.typing.NDArray[numpy.int32]
    type: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelMaterialViews:
    emission: numpy.typing.NDArray[numpy.float32]
    reflectance: numpy.typing.NDArray[numpy.float32]
    rgba: numpy.typing.NDArray[numpy.float32]
    shininess: numpy.typing.NDArray[numpy.float32]
    specular: numpy.typing.NDArray[numpy.float32]
    texid: numpy.typing.NDArray[numpy.int32]
    texrepeat: numpy.typing.NDArray[numpy.float32]
    texuniform: numpy.typing.NDArray[numpy.uint8]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelMeshViews:
    faceadr: numpy.typing.NDArray[numpy.int32]
    facenum: numpy.typing.NDArray[numpy.int32]
    graphadr: numpy.typing.NDArray[numpy.int32]
    texcoordadr: numpy.typing.NDArray[numpy.int32]
    vertadr: numpy.typing.NDArray[numpy.int32]
    vertnum: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelNumericViews:
    adr: numpy.typing.NDArray[numpy.int32]
    data: numpy.typing.NDArray[numpy.float64]
    size: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelPairViews:
    dim: numpy.typing.NDArray[numpy.int32]
    friction: numpy.typing.NDArray[numpy.float64]
    gap: numpy.typing.NDArray[numpy.float64]
    geom1: numpy.typing.NDArray[numpy.int32]
    geom2: numpy.typing.NDArray[numpy.int32]
    margin: numpy.typing.NDArray[numpy.float64]
    signature: numpy.typing.NDArray[numpy.int32]
    solimp: numpy.typing.NDArray[numpy.float64]
    solref: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelSensorViews:
    adr: numpy.typing.NDArray[numpy.int32]
    cutoff: numpy.typing.NDArray[numpy.float64]
    datatype: numpy.typing.NDArray[numpy.int32]
    dim: numpy.typing.NDArray[numpy.int32]
    needstage: numpy.typing.NDArray[numpy.int32]
    noise: numpy.typing.NDArray[numpy.float64]
    objid: numpy.typing.NDArray[numpy.int32]
    objtype: numpy.typing.NDArray[numpy.int32]
    refid: numpy.typing.NDArray[numpy.int32]
    reftype: numpy.typing.NDArray[numpy.int32]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelSiteViews:
    bodyid: numpy.typing.NDArray[numpy.int32]
    group: numpy.typing.NDArray[numpy.int32]
    matid: numpy.typing.NDArray[numpy.int32]
    pos: numpy.typing.NDArray[numpy.float64]
    quat: numpy.typing.NDArray[numpy.float64]
    rgba: numpy.typing.NDArray[numpy.float32]
    sameframe: numpy.typing.NDArray[numpy.uint8]
    size: numpy.typing.NDArray[numpy.float64]
    type: numpy.typing.NDArray[numpy.int32]
    user: numpy.typing.NDArray[numpy.float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelSkinViews:
    boneadr: numpy.typing.NDArray[numpy.int32]
    bonenum: numpy.typing.NDArray[numpy.int32]
    faceadr: numpy.typing.NDArray[numpy.int32]
    facenum: numpy.typing.NDArray[numpy.int32]
    inflate: numpy.typing.NDArray[numpy.float32]
    matid: numpy.typing.NDArray[numpy.int32]
    rgba: numpy.typing.NDArray[numpy.float32]
    texcoordadr: numpy.typing.NDArray[numpy.int32]
    vertadr: numpy.typing.NDArray[numpy.int32]
    vertnum: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelTendonViews:
    J_colind: numpy.typing.NDArray[numpy.int32]
    J_rowadr: numpy.typing.NDArray[numpy.int32]
    J_rownnz: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelTextureViews:
    adr: numpy.typing.NDArray[numpy.int32]
    data: numpy.typing.NDArray[numpy.uint8]
    height: numpy.typing.NDArray[numpy.int32]
    nchannel: numpy.typing.NDArray[numpy.int32]
    type: numpy.typing.NDArray[numpy.int32]
    width: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjModelTupleViews:
    adr: numpy.typing.NDArray[numpy.int32]
    objid: numpy.typing.NDArray[numpy.int32]
    objprm: numpy.typing.NDArray[numpy.float64]
    objtype: numpy.typing.NDArray[numpy.int32]
    size: numpy.typing.NDArray[numpy.int32]
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...

class _MjSolverStatList:
    def __init__(self, *args, **kwargs) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    @overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> MjSolverStat: ...
    @overload
    def __getitem__(self, arg0: slice) -> _MjSolverStatList: ...
    def __len__(self) -> int: ...
    @property
    def gradient(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def improvement(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def lineslope(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def nactive(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def nchange(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def neval(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def nupdate(self) -> numpy.typing.NDArray[numpy.int32]: ...

class _MjTimerStatList:
    def __init__(self, *args, **kwargs) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    @overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> MjTimerStat: ...
    @overload
    def __getitem__(self, arg0: mujoco._enums.mjtTimer) -> MjTimerStat: ...
    @overload
    def __getitem__(self, arg0: slice) -> _MjTimerStatList: ...
    def __len__(self) -> int: ...
    @property
    def duration(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def number(self) -> numpy.typing.NDArray[numpy.int32]: ...

class _MjWarningStatList:
    def __init__(self, *args, **kwargs) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    @overload
    def __getitem__(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> MjWarningStat: ...
    @overload
    def __getitem__(self, arg0: mujoco._enums.mjtWarning) -> MjWarningStat: ...
    @overload
    def __getitem__(self, arg0: slice) -> _MjWarningStatList: ...
    def __len__(self) -> int: ...
    @property
    def lastinfo(self) -> numpy.typing.NDArray[numpy.int32]: ...
    @property
    def number(self) -> numpy.typing.NDArray[numpy.int32]: ...

def mjv_averageCamera(cam1: MjvGLCamera, cam2: MjvGLCamera) -> MjvGLCamera: ...
