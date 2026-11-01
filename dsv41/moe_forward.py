"""Lower routing, quantization and fused MoE without intermediate copies.

Over EP4 the slab is an exported carry and its peer table filled by the
runtime. At one rank the slab, its one-entry peer table and the stats are
workspaces a pipeline stage owns, which `init` clears where the stage starts."""
from .forward import Lowered, scalar
from .head import offset
from .loading import GATE_BARRIERS
from .moe import gate, moe
from .programs import buf, call, integer


def fp8_input(workspace, *, capacity, experts=384, cubin_dir=None, rows="tokens", device_sms=152, ranks=4):
    """Where Mega mHC stores the MoE input: slab regions for FP8 x, routed and shared scales,
    and the MegaMoE block M runs of rows, which page the shared scales."""
    _, _, geometry = moe.pieces(experts, cubin_dir, rows, device_sms, capacity, ranks)
    slab = f"{workspace}.e{experts}.slab"
    regions = [offset(slab, geometry["offsets"][name]) for name in ("x", "x_sf", "shared_x_sf")]
    return regions, geometry["shared_sf_rows"], moe.segments(experts, capacity, ranks)


def forward(pieces, layer, layouts, source, output, *, rows, capacity,
            workspace, experts=384, cubin_dir=None, device_sms=152, ranks=4):
    modules, ops, geometry = moe.pieces(experts, cubin_dir, rows, device_sms, capacity, ranks)
    names = pieces.add((modules,ops))
    gate_name = f"dsv41_gate_e{experts}"
    gates = pieces.add(gate.pieces(
        rows,experts,cubin_dir,max_tokens=capacity,raw_outputs=True,device_sms=device_sms))
    slab, peers, stats = (f"{workspace}.e{experts}.{n}" for n in ("slab","peers","stats"))
    buffers = {
        **slab_buffers(workspace, experts, geometry, ranks),
        output: {"kind":"workspace","dtype":"bf16","shape":[capacity,5120]},
    }
    region = lambda name: offset(slab,geometry["offsets"][name])
    routed, shared = (layouts[layer+".ffn."+n] for n in ("experts","shared_experts"))
    tag = workspace+"."+layer
    calls = [
        call(tag+".gate",gates[gate_name],buf(source),buf(layer+".ffn.gate.weight"),
             buf(layer+".ffn.gate.bias"),region("idx"),region("weights"),scalar(rows),buf(GATE_BARRIERS)),
        call(tag+".experts",names[f"dsv41_mega_moe_e{experts}"],
             buf(output),buf(stats),scalar(rows),buf(peers),{"rank":"ep"} if ranks > 1 else integer(0),
             region("l1"),region("l1_sf"),buf(routed["w1"]),buf(routed["w1_sf"]),
             region("l2"),region("l2_sf"),buf(routed["w2"]),buf(routed["w2_sf"]),
             region("x"),region("shared_x_sf"),buf(shared["w1"]),buf(shared["w1_sf"]),
             region("shared_l2"),region("shared_l2_sf"),buf(shared["w2"]),buf(shared["w2_sf"])),
    ]
    return Lowered(buffers,calls)



def slab_buffers(workspace, experts, geometry, ranks):
    slab, peers, stats = (f"{workspace}.e{experts}.{n}" for n in ("slab","peers","stats"))
    if ranks > 1:
        return {slab: {"kind":"carry","dtype":"u8","shape":[geometry["slab_bytes"]],"export":True},
                peers: {"kind":"peer","dtype":"u64","shape":[ranks],"of":slab,"group":"ep"},
                stats: {"kind":"carry","dtype":"i32","shape":[experts//ranks]}}
    return {slab: {"kind":"workspace","dtype":"u8","shape":[geometry["slab_bytes"]]},
            peers: {"kind":"workspace","dtype":"u64","shape":[1]},
            stats: {"kind":"workspace","dtype":"i32","shape":[experts]}}


def init(pieces, workspace, *, rows, capacity, experts=384, cubin_dir=None, device_sms=152):
    """Where a stage starts, a one-rank slab's signals and stats cleared and its peer table filled."""
    _, _, geometry = moe.pieces(experts, cubin_dir, rows, device_sms, capacity, 1)
    pieces.fixed(moe.slab_init(cubin_dir))
    slab, peers, stats = (f"{workspace}.e{experts}.{n}" for n in ("slab","peers","stats"))
    return Lowered(slab_buffers(workspace, experts, geometry, 1),
                   [call(f"{workspace}.e{experts}.init", "dsv41_slab_init", buf(peers), buf(slab),
                         integer(geometry["signal_bytes"] // 8), buf(stats), integer(experts))])
