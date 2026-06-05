from collections.abc import Sequence
import enum

from .tensor_like import TensorLike

class dtype:
    @property
    def name(self) -> str: ...
    @property
    def code(self) -> int: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...

half: dtype = ...
float32: dtype = ...
float64: dtype = ...
int16: dtype = ...
int32: dtype = ...
half2: dtype = ...
float2: dtype = ...
complex64: dtype = ...
complex128: dtype = ...
double2: dtype = ...
short2: dtype = ...
int2: dtype = ...

class TensorView:
    @property
    def base(self) -> TensorLike: ...
    @property
    def shape(self) -> tuple: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> dtype: ...
    def __repr__(self) -> str: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(self, **kwargs) -> object: ...

def view(a: object, dtype: dtype) -> TensorView: ...

class Algorithm(enum.Enum):
    DEFAULT = 0
    ALG1 = 1
    ALG2 = 2
    ALG3 = 3
    ALG4 = 4

class InterpMode(enum.Enum):
    NEAREST = 0
    LINEAR = 1

class ComputeType(enum.Enum):
    DEFAULT = 0
    FP16 = 1
    FP32 = 2
    FP64 = 3

class Handle:
    def __init__(self) -> None: ...
    def destroy(self) -> None: ...

def event_create() -> int: ...
def event_destroy(event: int) -> None: ...
def event_record(handle: Handle, event: int) -> None: ...
def event_synchronize(event: int) -> None: ...
def event_elapsed_time(start: int, stop: int) -> float: ...

class ContractionPlan:
    pass

def create_contraction(
    handle: Handle,
    a: TensorLike,
    a_modes: Sequence[int],
    b: TensorLike,
    b_modes: Sequence[int],
    out: TensorLike,
    out_modes: Sequence[int],
) -> ContractionPlan: ...
def contraction(
    handle: Handle, plan: ContractionPlan, a: TensorLike, b: TensorLike, out: TensorLike
) -> None: ...

class InterpolationPlan:
    pass

def create_interpolation_plan(
    handle: Handle, nx: int, ny: int, nz: int, gridpos: TensorLike, mode: InterpMode
) -> InterpolationPlan: ...
def interpolation_preprocess(
    handle: Handle, plan: InterpolationPlan, querypos: TensorLike
) -> None: ...
def interpolation(
    handle: Handle,
    plan: InterpolationPlan,
    querypos: TensorLike,
    x: TensorLike,
    output: TensorLike,
    fill_value: TensorLike,
) -> None: ...
def truncate_rank(
    handle: Handle, x: TensorLike, start: int, stop: int, out: TensorLike
) -> None: ...
def das(
    handle: Handle,
    x: TensorLike,
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    srcdir: TensorLike | None,
    wavenum: float,
    beta: TensorLike,
    out: TensorLike,
    alg: Algorithm,
    compute_type: ComputeType,
    channels_trailing: bool,
) -> None: ...
def das_sparse(
    handle: Handle,
    x: TensorLike,
    srcpos: TensorLike,
    dstpos: TensorLike,
    offsets: TensorLike,
    weights: TensorLike,
    srcdir: TensorLike | None,
    wavenum: float,
    beta: TensorLike,
    out: TensorLike,
    sparse_indices: TensorLike,
    alg: Algorithm,
    compute_type: ComputeType,
    channels_trailing: bool,
) -> None: ...
def contiguous_copy(handle: Handle, x: TensorLike, out: TensorLike) -> None: ...
def gather(
    handle: Handle, x: TensorLike, out: TensorLike, mode: int, indices: TensorLike
) -> None: ...
def scatter(
    handle: Handle, x: TensorLike, out: TensorLike, mode: int, indices: TensorLike
) -> None: ...
def greens_sum(
    handle: Handle,
    srcpos: TensorLike,
    wavenums: TensorLike,
    x: TensorLike,
    dstpos: TensorLike,
    out: TensorLike,
) -> None: ...
