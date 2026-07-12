import torch
from mps_deform_conv import deform_conv2d


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    device = torch.device("mps")
    torch.manual_seed(7)
    input_tensor = torch.randn(1, 4, 8, 8, device=device)
    offset = torch.randn(1, 18, 8, 8, device=device) * 0.1
    weight = torch.randn(4, 4, 3, 3, device=device)
    bias = torch.randn(4, device=device)
    mask = torch.sigmoid(torch.randn(1, 9, 8, 8, device=device))
    output = deform_conv2d(
        input_tensor,
        offset,
        weight,
        bias,
        stride=1,
        padding=1,
        dilation=1,
        mask=mask,
    )
    torch.mps.synchronize()
    if tuple(output.shape) != (1, 4, 8, 8):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("mps-deform-conv returned non-finite values")
    print("mps-deform-conv smoke test passed")


if __name__ == "__main__":
    main()
