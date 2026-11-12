# Copyright Xingyu Chen.
# Replays a frozen path topology for a moving transmitter with the consumer time-varying API.

"""Replay a frozen path topology for a moving transmitter with the consumer time-varying API."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import torch

from witwin.channel.importers import load_mesh
from witwin.channel.presets import itu_material
from witwin.channel.propagation import consumer
from witwin.channel.scene import compile as compile_scene

import street_canyon

TX_ID, RX_ID = 1, 2


def _batch(positions: torch.Tensor, stable_id: int, *, source: bool) -> consumer.EndpointBatch:
    rows = positions.shape[0]
    return consumer.EndpointBatch(
        stable_ids=torch.full((rows,), stable_id, dtype=torch.int64, device="cuda"),
        positions_m=positions.contiguous(),
        polarizations=torch.tensor([[0.0, 0.0, 1.0]], device="cuda").expand(rows, 3).contiguous(),
        powers_w=torch.ones(rows, device="cuda") if source else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=3.5e9)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--slot-interval", type=float, default=1e-3, help="seconds per slot")
    parser.add_argument("--speed", type=float, default=15.0, help="Tx speed along +x, m/s")
    parser.add_argument("--rediscover-every", type=int, default=32, help="slots per topology")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        obj = street_canyon.write_obj(Path(workdir) / "canyon.obj")
        imported = load_mesh(
            obj,
            {
                # Non-dispersive presets keep the option of wideband frequency-offset replay.
                "concrete": itu_material("concrete", args.frequency, dispersive=False),
                "ground": itu_material("medium_dry_ground", args.frequency, dispersive=False),
            },
        )
    compiled = compile_scene(imported.to_scene(), reference_frequency_hz=args.frequency)

    start_tx = torch.tensor([-37.3, 1.1, 10.0], device="cuda")
    rx = torch.tensor([[35.0, 4.0, 1.5]], device="cuda")
    velocity = torch.tensor([args.speed, 0.0, 0.0], device="cuda")
    times = torch.arange(args.slots, dtype=torch.float64, device="cuda") * args.slot_interval

    coefficients, delays, valid_counts = [], [], []
    begin = time.perf_counter()
    for first in range(0, args.slots, args.rediscover_every):
        block = times[first : first + args.rediscover_every]
        tx_positions = start_tx + block.float().unsqueeze(1) * velocity
        # Discover once at the first instant of the block, then replay the frozen rows.
        discovered = consumer.evaluate(
            compiled,
            consumer.PropagationRequest(
                sources=_batch(tx_positions[:1], TX_ID, source=True),
                sinks=_batch(rx, RX_ID, source=False),
                reference_frequency_hz=args.frequency,
                # Discovery also returns transmission and diffraction rows, but a frozen
                # topology can hold only capabilities().fixed_topology_components (LoS
                # and reflection), so this example discovers and replays those two.
                components=frozenset({"los", "reflection"}),
                max_depth=2,
                response="scalar_transport",
                topology_mode="discover",
                ad_mode="none",
            ),
        )
        prepared = consumer.prepare_fixed_topology(discovered.paths.topology)
        cir = consumer.evaluate_time_varying(
            compiled,
            consumer.TimeVaryingRequest(
                sources=_batch(tx_positions, TX_ID, source=True),
                sinks=_batch(rx.expand(block.shape[0], 3), RX_ID, source=False),
                reference_frequency_hz=args.frequency,
                topology=prepared,
                times_s=block,
                response="scalar_transport",
                ad_mode="none",
            ),
        )
        valid = (
            cir.row_valid
            if cir.row_valid is not None
            else torch.ones_like(cir.delay_s, dtype=torch.bool)
        )
        coefficients.append(cir.transport.coefficient.masked_fill(~valid, 0))
        delays.append(cir.delay_s)
        valid_counts.append(valid.sum(dim=1))
        print(f"block at t={float(block[0]) * 1e3:6.1f} ms: K={prepared.row_count} frozen rows")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin

    # Narrowband channel per slot and the phase rotation the motion produces.
    h = torch.cat([c.sum(dim=1) for c in coefficients])
    counts = torch.cat(valid_counts)
    gain_db = 20 * torch.log10(h.abs())
    print(f"{args.slots} slots ({args.slots * args.slot_interval * 1e3:.0f} ms) in "
          f"{1e3 * elapsed:.1f} ms; Tx moved {args.speed * float(times[-1]):.2f} m")
    for slot in range(0, args.slots, max(1, args.slots // 8)):
        print(f"  t={float(times[slot]) * 1e3:6.1f} ms  |h|^2 {float(gain_db[slot]):7.2f} dB  "
              f"phase {float(torch.angle(h[slot])):+.3f} rad  live rows {int(counts[slot])}")
    first_delays = delays[0][:, :1]
    print(f"first-row delay moved {1e9 * float(first_delays[-1] - first_delays[0]):+.3f} ns "
          f"over the first block")


if __name__ == "__main__":
    main()
