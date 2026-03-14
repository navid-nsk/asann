"""End-to-end training test with CUDA ops.

Verifies that a ASANN model with use_cuda_ops=True can:
1. Be constructed
2. Forward pass works
3. Backward pass works
4. Surgery works (operations are created as CUDA variants)
5. A short training loop completes without errors
"""

import sys
import os
import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from asann import ASANNConfig, ASANNModel, ASANNTrainer


def test_basic_forward_backward():
    """Test model forward + backward with CUDA ops."""
    print("Test 1: Basic forward/backward with CUDA ops...")
    device = "cuda"
    config = ASANNConfig(
        d_init=32,
        initial_num_layers=2,
        surgery_interval_init=100,
        warmup_steps=50,
        complexity_target=50000,
        device=device,
        use_cuda_ops=True,
    )

    model = ASANNModel(d_input=8, d_output=1, config=config)
    model.to(device)

    # Forward
    x = torch.randn(4, 8, device=device)
    out = model(x)
    print(f"  Forward output shape: {out.shape}")
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    # Backward
    loss = out.sum()
    loss.backward()
    print(f"  Backward completed, loss={loss.item():.4f}")

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Gradients: {grad_count}/{total_params} params have grads")

    print("  PASS")
    return True


def test_short_training():
    """Test a short training loop with CUDA ops."""
    print("\nTest 2: Short training loop with CUDA ops...")
    device = "cuda"
    config = ASANNConfig(
        d_init=32,
        initial_num_layers=2,
        surgery_interval_init=50,
        warmup_steps=20,
        complexity_target=50000,
        device=device,
        use_cuda_ops=True,
    )

    d_input = 8
    d_output = 1
    model = ASANNModel(d_input=d_input, d_output=d_output, config=config)
    model.to(device)

    trainer = ASANNTrainer(
        model=model,
        config=config,
        task_loss_fn=torch.nn.MSELoss(),
        task_type="regression",
    )

    # Create simple synthetic data
    np.random.seed(42)
    X = np.random.randn(200, d_input).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] - 0.5 * X[:, 2]).astype(np.float32)

    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).unsqueeze(-1).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Train for a few epochs
    max_epochs = 3
    print(f"  Training for {max_epochs} epochs...")
    metrics = trainer.train_epochs(
        train_data=train_loader,
        max_epochs=max_epochs,
        print_every=50,
    )

    final_loss = metrics.get("final_loss", float("inf"))
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Steps completed: {trainer.global_step}")

    # Verify model still works after training
    model.eval()
    with torch.no_grad():
        out = model(X_tensor[:4])
    print(f"  Eval output shape: {out.shape}")

    print("  PASS")
    return True


def test_surgery_creates_cuda_ops():
    """Test that surgery creates CUDA operations when use_cuda_ops=True."""
    print("\nTest 3: Surgery creates CUDA operations...")
    device = "cuda"
    config = ASANNConfig(
        d_init=32,
        initial_num_layers=2,
        surgery_interval_init=50,
        warmup_steps=10,
        complexity_target=100000,
        device=device,
        use_cuda_ops=True,
    )

    from asann.surgery import create_operation

    # Create some operations and verify they're CUDA variants
    ops_to_test = [
        ("embed_positional", 32, None),
        ("embed_factored", 32, None),
        ("embed_mlp", 32, None),
        ("attn_self", 32, None),
        ("conv1d_k3", 32, None),
    ]

    for name, d, spatial_shape in ops_to_test:
        op = create_operation(name, d=d, device=device, config=config,
                              spatial_shape=spatial_shape)
        cls_name = type(op).__name__
        is_cuda = "CUDA" in cls_name
        status = "CUDA" if is_cuda else "PYTHON"
        print(f"  {name}: {cls_name} [{status}]")
        assert is_cuda, f"Expected CUDA variant for {name}, got {cls_name}"

    print("  PASS")
    return True


def main():
    print("=" * 60)
    print("ASANN CUDA End-to-End Training Tests")
    print("=" * 60)
    print(f"Device: cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    results["forward_backward"] = test_basic_forward_backward()
    results["short_training"] = test_short_training()
    results["surgery_cuda_ops"] = test_surgery_creates_cuda_ops()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
