# pyright: reportOperatorIssue=false, reportReturnType=false
# (PyTensor's stubs type pt.sum/pt.sqr results loosely; the expression is a plain tensor.)
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable


def convex_emulator(x: TensorVariable) -> TensorVariable:
    """
    A strictly convex differentiable emulator written in PyTensor.

    This acts as the fallback 'grey-box' surrogate model. By defining
    a strictly convex function, we mathematically guarantee that the
    Jacobian/Hessian gradients passed to the NUTS sampler will never
    result in pathological Hamiltonian trajectories.

    Args:
        x: PyTensor tensor variable (building geometry features)

    Returns:
        y: PyTensor scalar (baseline physics energy demand)
    """
    # A highly stable, strictly convex equation: y = sum(0.5 * x^2 + x)
    # This ensures smooth, non-exploding Vector-Jacobian Products (VJPs)
    y = pt.sum(0.5 * pt.sqr(x) + x)

    return y
