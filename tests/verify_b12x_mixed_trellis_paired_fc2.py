#!/usr/bin/env python3
"""Verify the r18p paired-FC2 contract on the GLM-5.2 TP4 geometries.

The gate uses the production launch capacities and both observed K3/K4 expert
splits while synthesizing deterministic weights, so it needs no checkpoint.
It verifies dispatch metadata, codegen resources, frozen bit-exact outputs,
and large-M latency bounds that distinguish paired r18p from stock r18.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mixed_api
from b12x.moe._shared.kernels.w4a16.prepare import (
    prepare_trellis256_moe_weights,
)


HIDDEN = 6144
INTERMEDIATE = 512
TOPK = 8
TOTAL_EXPERTS = 256
TILE_CONFIG = (128, 128, 32, 512)
DECODE_CAPACITY = 32
PREFILL_CAPACITY = 3072
DECODE_BLOCK_M = 8
PREFILL_BLOCK_M = 32

EXPECTED_OUTPUT_SHA256 = {
    (148, 108): {
        1: "633d037ca1b1a6e14a008a3a35d19b137499b0ebc5b5e3d6ae7fe3c5106042ff",
        32: "32a78d512345e69330e6302c89d616957e2d74dd845606d1de5599baa07e8ede",
        33: "1e782220ff1955340fee59400e47bf222acd07c79bcbb924fb094be26644d409",
        221: "cdba33241760466ff2248a05d0b9721e39d544fe5f08b487bf869df208f13bf2",
        222: "18b88733832391a3b53c830d977cf06d6f97ddc10a1cbb8b8c527c51c83fc1c4",
        3072: "3b8bfd27da9d2e0c619a0657d104b6f11b8e14e7bab02ddaf9bfa167a26d19cf",
    },
    (192, 64): {
        1: "e10fb0bb8e26ac809940c5ab098794f63e5d5e66cf575f898631d90c62121a64",
        32: "9f5f1010df9263ffaae66a82e43bdf2406eceeb94cedb02039e572a57f0a267a",
        33: "d9b77aa782cad5b6582450d00600b5b0f643b431efffd05194826023c94a410d",
        221: "2799a80940a2104db6b3396451209bf9ecef862f9fe50e025b8d3bcc6a51d450",
        222: "b4160a2487bb2c332c588cab913c60e3dba16ad9881e8997e51e4f46987af2f5",
        3072: "f8e1cbd2cc7b74d3c27658c416bb9aa1525c122712be829f0f2b71c11d53411b",
    },
}

PAIRED_MEDIAN_US_LIMIT = {
    (148, 108): {33: 3900.0, 221: 6300.0, 222: 6300.0, 3072: 21500.0},
    (192, 64): {33: 3700.0, 221: 6000.0, 222: 6000.0, 3072: 21500.0},
}


@dataclass
class Result:
    m: int
    plan: str
    median_us: float
    minimum_us: float
    maximum_us: float
    output_norm: float
    output_finite: bool
    output_sha256: str
    reference_max_abs: float | None
    reference_mean_abs: float | None
    reference_rel_l2: float | None
    reference_exact_fraction: float | None
    reference_allclose: bool | None


def _prepare_tier(*, bits: int, experts: int, seed: int, device: torch.device) -> Any:
    # The BTX generation renamed the native layout without changing its wire
    # geometry.  Bound execution is the reliable cross-version discriminator.
    w13_layout = (
        "trellis_t256_proj"
        if hasattr(mixed_api, "bind_mixed_trellis")
        else "trellis3_t256_proj"
    )
    ones_h = torch.ones((1, HIDDEN), dtype=torch.float16, device=device)
    ones_i = torch.ones((experts, 3 * INTERMEDIATE), dtype=torch.float16, device=device)
    return prepare_trellis256_moe_weights(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts=experts,
        activation="silu",
        fc1_tile_n=TILE_CONFIG[1],
        fc2_tile_n=TILE_CONFIG[3],
        device=device,
        seed=seed,
        params_dtype=torch.float16,
        w13_layout=w13_layout,
        trellis_bits=bits,
        codebook="mcg",
        gate_suh=ones_h,
        up_suh=ones_h,
        intermediate_rotations=ones_i,
        down_svh=ones_h,
        tile_config=TILE_CONFIG,
    )


def _make_runner(
    *,
    capacity: int,
    block_m: int,
    tier0: Any,
    tier1: Any,
    global_map: torch.Tensor,
    descriptor: torch.Tensor,
    rotations: Any,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], Any]:
    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(capacity * TOPK, block_m, TOTAL_EXPERTS)
    compile_kwargs = {
        "size_m": capacity,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "tier0_num_experts": int(tier0.num_experts),
        "tier1_num_experts": int(tier1.num_experts),
        "tier0_bits": 3,
        "tier1_bits": 4,
        "top_k": TOPK,
        "max_m_blocks": (route_slots + block_m - 1) // block_m,
        "moe_block_size": block_m,
        "sms": int(props.multi_processor_count),
        "max_shared_mem": int(props.shared_memory_per_block_optin),
        "force_tile_config": TILE_CONFIG,
        "rotation_input_dtype": "bf16",
        "route_ids_dtype": torch.int32,
        "broadcast_suh": True,
        "broadcast_svh": True,
    }
    compile_parameters = inspect.signature(mixed_api.compile_mixed_trellis).parameters
    if "trellis_codebook" in compile_parameters:
        compile_kwargs["trellis_codebook"] = "mcg"
    if "route_num_experts" in compile_parameters:
        compile_kwargs["route_num_experts"] = TOTAL_EXPERTS
    launch = mixed_api.compile_mixed_trellis(**compile_kwargs)
    buffers = mixed_api.make_mixed_trellis_buffers(
        launch, device=device, sms=int(props.multi_processor_count)
    )
    if hasattr(mixed_api, "bind_mixed_trellis"):
        binding = mixed_api.bind_mixed_trellis(
            tier0,
            tier1,
            global_map,
            descriptor,
            rotations,
            launch,
        )

        def run(x, weights, ids):
            return mixed_api.run_bound_mixed_trellis(x, weights, ids, binding, buffers)

    else:

        def run(x, weights, ids):
            return mixed_api.run_mixed_trellis(
                x,
                tier0,
                tier1,
                weights,
                ids,
                global_map,
                descriptor,
                rotations,
                launch,
                buffers,
            )

    return run, launch


def _inputs(m: int, device: torch.device):
    cpu_generator = torch.Generator(device="cpu").manual_seed(20260819 + m)
    ids = torch.stack(
        [
            torch.randperm(TOTAL_EXPERTS, generator=cpu_generator)[:TOPK]
            for _ in range(m)
        ]
    ).to(dtype=torch.int32, device=device)
    cuda_generator = torch.Generator(device=device).manual_seed(20260819 + m)
    weights = torch.softmax(
        torch.randn(
            (m, TOPK),
            dtype=torch.float32,
            device=device,
            generator=cuda_generator,
        ),
        dim=-1,
    )
    x = (
        torch.randn(
            (m, HIDDEN),
            dtype=torch.float32,
            device=device,
            generator=cuda_generator,
        )
        * 1.0e-3
    ).to(torch.bfloat16)
    return x, weights, ids


def _time_calls(fn: Callable[[], torch.Tensor], warmup: int, repeats: int):
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        output = fn()
        end.record()
    torch.cuda.synchronize()
    elapsed = [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]
    finite = bool(torch.isfinite(output).all().item())
    norm = float(output.float().norm().item())
    return elapsed, finite, norm, output.detach().cpu().contiguous()


def _output_path(directory: Path, tier0_experts: int, tier1_experts: int, m: int):
    return directory / f"tier-{tier0_experts}-{tier1_experts}-m-{m}.pt"


def _compare_reference(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, float]:
    if output.shape != reference.shape or output.dtype != reference.dtype:
        raise AssertionError(
            "mixed-Trellis output metadata differs from reference: "
            f"output={tuple(output.shape)}/{output.dtype}, "
            f"reference={tuple(reference.shape)}/{reference.dtype}"
        )
    if not bool(torch.isfinite(reference).all().item()):
        raise AssertionError("mixed-Trellis reference contains non-finite values")
    output_f32 = output.float()
    reference_f32 = reference.float()
    difference = (output_f32 - reference_f32).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    reference_norm = float(reference_f32.norm().item())
    rel_l2 = float(difference.norm().item()) / max(reference_norm, 1.0e-12)
    exact_fraction = float((output == reference).float().mean().item())
    torch.testing.assert_close(
        output_f32,
        reference_f32,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )
    return max_abs, mean_abs, rel_l2, exact_fraction


def _launch_metadata(launch: Any) -> dict[str, int | bool]:
    return {
        "fc2_moe_block_size": int(launch.fc2_moe_block_size),
        "fc2_schedule_route_block_factor": int(launch.fc2_schedule_route_block_factor),
        "fc2_paired_m8_routes": bool(launch.fc2_paired_m8_routes),
        "blocks_per_sm": int(launch.blocks_per_sm),
        "shared_memory_bytes": int(launch.shared_memory_bytes),
        "registers_per_thread": int(launch.registers_per_thread),
        "local_memory_bytes": int(launch.local_memory_bytes),
    }


def _assert_launch_contract(
    *, launch: Any, paired: bool, route_factor: int, props: Any, name: str
) -> None:
    metadata = _launch_metadata(launch)
    expected = {
        "fc2_moe_block_size": 8,
        "fc2_schedule_route_block_factor": route_factor,
        "fc2_paired_m8_routes": paired,
    }
    for field, value in expected.items():
        if metadata[field] != value:
            raise AssertionError(
                f"{name} launch has {field}={metadata[field]!r}, expected {value!r}"
            )
    if not 0 < metadata["registers_per_thread"] <= 255:
        raise AssertionError(
            f"{name} launch has invalid register count: "
            f"{metadata['registers_per_thread']}"
        )
    if metadata["local_memory_bytes"] != 0:
        raise AssertionError(
            f"{name} launch spills {metadata['local_memory_bytes']} bytes/thread"
        )
    if (
        not 0
        < metadata["shared_memory_bytes"]
        <= int(props.shared_memory_per_block_optin)
    ):
        raise AssertionError(
            f"{name} launch has invalid shared memory: "
            f"{metadata['shared_memory_bytes']}"
        )
    if metadata["blocks_per_sm"] < 1:
        raise AssertionError(
            f"{name} launch has invalid blocks_per_sm: {metadata['blocks_per_sm']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tier0-experts", type=int, default=148)
    parser.add_argument("--tier1-experts", type=int, default=108)
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1, 4, 16, 32, 33, 221, 222, 3072]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--save-output-dir", type=Path)
    parser.add_argument("--reference-output-dir", type=Path)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-2)
    parser.add_argument("--reference-atol", type=float, default=1.0e-2)
    parser.add_argument(
        "--gate-paired-fc2",
        action="store_true",
        help="enforce the SM121 r18p dispatch, output, and latency contract",
    )
    args = parser.parse_args()
    if args.tier0_experts + args.tier1_experts != TOTAL_EXPERTS:
        raise ValueError("tier counts must sum to 256")
    if args.save_output_dir is not None:
        args.save_output_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_output_dir is not None and not args.reference_output_dir.is_dir():
        raise FileNotFoundError(
            f"reference output directory not found: {args.reference_output_dir}"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    if args.gate_paired_fc2:
        geometry = (args.tier0_experts, args.tier1_experts)
        if capability != (12, 1) or int(props.multi_processor_count) != 48:
            raise AssertionError(
                "paired-FC2 release gate requires a 48-SM SM121 device, got "
                f"capability={capability}, sms={props.multi_processor_count}"
            )
        if geometry not in EXPECTED_OUTPUT_SHA256:
            raise AssertionError(f"unsupported release-gate geometry: {geometry}")
        if set(args.m) != set(EXPECTED_OUTPUT_SHA256[geometry]):
            raise AssertionError(
                "release gate requires M={1,32,33,221,222,3072}, got "
                f"{sorted(set(args.m))}"
            )
    started = time.monotonic()
    tier0 = _prepare_tier(bits=3, experts=args.tier0_experts, seed=301, device=device)
    tier1 = _prepare_tier(bits=4, experts=args.tier1_experts, seed=401, device=device)
    global_map, descriptor = mixed_api.build_tiered_maps(
        range(args.tier0_experts),
        range(args.tier0_experts, TOTAL_EXPERTS),
        device=device,
    )
    rotations = mixed_api.MixedTrellisRotations(
        intermediate=torch.cat(
            (tier0.intermediate_rotations, tier1.intermediate_rotations), dim=0
        ).contiguous(),
        gate_suh=torch.ones((1, HIDDEN), dtype=torch.float16, device=device),
        up_suh=torch.ones((1, HIDDEN), dtype=torch.float16, device=device),
        down_svh=torch.ones((1, HIDDEN), dtype=torch.float16, device=device),
    )
    decode, decode_launch = _make_runner(
        capacity=DECODE_CAPACITY,
        block_m=DECODE_BLOCK_M,
        tier0=tier0,
        tier1=tier1,
        global_map=global_map,
        descriptor=descriptor,
        rotations=rotations,
        device=device,
    )
    prefill, prefill_launch = _make_runner(
        capacity=PREFILL_CAPACITY,
        block_m=PREFILL_BLOCK_M,
        tier0=tier0,
        tier1=tier1,
        global_map=global_map,
        descriptor=descriptor,
        rotations=rotations,
        device=device,
    )
    if args.gate_paired_fc2:
        _assert_launch_contract(
            launch=decode_launch,
            paired=False,
            route_factor=1,
            props=props,
            name="decode",
        )
        _assert_launch_contract(
            launch=prefill_launch,
            paired=True,
            route_factor=2,
            props=props,
            name="prefill",
        )
    print(
        json.dumps(
            {
                "label": args.label,
                "device": props.name,
                "capability": list(capability),
                "sms": int(props.multi_processor_count),
                "torch": torch.__version__,
                "b12x_api": (
                    "bound" if hasattr(mixed_api, "bind_mixed_trellis") else "legacy"
                ),
                "tier_counts": [args.tier0_experts, args.tier1_experts],
                "tile": TILE_CONFIG,
                "decode_launch": _launch_metadata(decode_launch),
                "prefill_launch": _launch_metadata(prefill_launch),
                "setup_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results = []
    for m in args.m:
        if m <= 0 or m > PREFILL_CAPACITY:
            raise ValueError(f"M must be in [1, {PREFILL_CAPACITY}], got {m}")
        run = decode if m <= DECODE_CAPACITY else prefill
        plan = "decode" if m <= DECODE_CAPACITY else "prefill"
        x, weights, ids = _inputs(m, device)
        elapsed, finite, norm, output_cpu = _time_calls(
            lambda: run(x, weights, ids), args.warmup, args.repeats
        )
        output_sha256 = hashlib.sha256(
            output_cpu.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        reference_max_abs = None
        reference_mean_abs = None
        reference_rel_l2 = None
        reference_exact_fraction = None
        reference_allclose = None
        output_path = (
            _output_path(
                args.save_output_dir, args.tier0_experts, args.tier1_experts, m
            )
            if args.save_output_dir is not None
            else None
        )
        if args.reference_output_dir is not None:
            reference_path = _output_path(
                args.reference_output_dir,
                args.tier0_experts,
                args.tier1_experts,
                m,
            )
            if not reference_path.is_file():
                raise FileNotFoundError(f"reference output not found: {reference_path}")
            reference = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
            (
                reference_max_abs,
                reference_mean_abs,
                reference_rel_l2,
                reference_exact_fraction,
            ) = _compare_reference(
                output_cpu,
                reference,
                rtol=args.reference_rtol,
                atol=args.reference_atol,
            )
            reference_allclose = True
        if output_path is not None:
            torch.save(output_cpu, output_path)
        result = Result(
            m=m,
            plan=plan,
            median_us=statistics.median(elapsed),
            minimum_us=min(elapsed),
            maximum_us=max(elapsed),
            output_norm=norm,
            output_finite=finite,
            output_sha256=output_sha256,
            reference_max_abs=reference_max_abs,
            reference_mean_abs=reference_mean_abs,
            reference_rel_l2=reference_rel_l2,
            reference_exact_fraction=reference_exact_fraction,
            reference_allclose=reference_allclose,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)
        if not finite:
            raise RuntimeError(f"non-finite mixed-Trellis output at M={m}")
        if args.gate_paired_fc2:
            geometry = (args.tier0_experts, args.tier1_experts)
            expected_sha256 = EXPECTED_OUTPUT_SHA256[geometry][m]
            if output_sha256 != expected_sha256:
                raise AssertionError(
                    f"mixed-Trellis output digest mismatch at geometry={geometry}, "
                    f"M={m}: got {output_sha256}, expected {expected_sha256}"
                )
            limit = PAIRED_MEDIAN_US_LIMIT[geometry].get(m)
            if limit is not None and result.median_us > limit:
                raise AssertionError(
                    f"paired-FC2 latency regression at geometry={geometry}, M={m}: "
                    f"median={result.median_us:.3f} us, limit={limit:.3f} us"
                )
    print(
        json.dumps(
            {"label": args.label, "results": [asdict(result) for result in results]},
            sort_keys=True,
        ),
        flush=True,
    )
    if args.gate_paired_fc2:
        print("B12X paired-FC2 SM121 gate: PASS", flush=True)


if __name__ == "__main__":
    main()
