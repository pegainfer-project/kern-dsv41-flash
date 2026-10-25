"""Generate DeepSeek V4.1 Flash's serving manifest: load, prefill, batch decode and the
five-draft/six-verify round, over the cubins `scripts/build.py` and
`engram_constants.py` write into one directory as `<module>.cubin`."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import shutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dsv41.schema import SCHEMA_VERSION
from dsv41.loading import Pieces, barriers, dense_scales, expert_weights
from dsv41.auxiliary.serving import Layout, build
from dsv41.draft_context import capture, publish
from dsv41.target import forward, head
from dsv41.draft import forward as draft_forward
from dsv41.programs import Stages, assemble, spec_ops


def bundle(manifest, destination, roots):
    """Copy the pinned cubins as `<module>-<sha12>.cubin`, the registry layout.

    The manifest's module sources become those names, so nothing in a
    published manifest points at a build tree; the runtime resolves by sha256.
    """
    destination.mkdir(parents=True,exist_ok=True)
    for name, module in manifest["modules"].items():
        source = Path(module["source"])
        candidates = [source,*(root/source.name for root in roots)]
        match = next((path for path in candidates if path.is_file()
                      and hashlib.sha256(path.read_bytes()).hexdigest() == module["sha256"]),None)
        if match is None:
            raise ValueError(f"cannot find pinned cubin for module {name}")
        output = destination/f"{name}-{module['sha256'][:12]}.cubin"
        if match.resolve() != output.resolve():
            shutil.copyfile(match,output)
        if hashlib.sha256(output.read_bytes()).hexdigest() != module["sha256"]:
            raise ValueError(f"cubin changed while bundling module {name}")
        module["source"] = output.name


def engram_shards(raw):
    """Layer -> (ranks, rows per rank, total rows) for each Engram table sharded into HBM (see weights.sharded); empty for host tables."""
    shards = {}
    for layer in (1,14):
        spec = raw.get(f"layers.{layer}.engram.embed.weight",{})
        if spec.get("export"):
            ranges = spec["bind"][0]["rows"]["ranges"]
            shards[layer] = (len(ranges),spec["shape"][0],ranges[-1][1])
    return shards


def generate(raw, constants, *, kernels, capacity=128, max_seqs=16, context=32768, device_sms=152):
    cubin = lambda module: kernels/f"{module}.cubin"
    cubin_dir, auxiliary_cubin, head_cubin = kernels, cubin("dsv41_auxiliary"), cubin("dsv41_head")
    fused_cubin = cubin("dsv41_fused_decode")
    pieces = Pieces()
    projections = {name:b for name,b in raw.items()
                   if b.get("placement") != "host" and b["kind"] == "weight" and ".engram.embed." not in name}
    dense_buffers, dense_load, dense_layouts = dense_scales(projections,pieces,cubin_dir=cubin_dir,fused_attention=True)
    expert_buffers, expert_load, expert_layouts = expert_weights(projections,pieces,cubin_dir=cubin_dir)
    barrier_buffers, barrier_load = barriers(pieces,cubin_dir=cubin_dir)
    engram = engram_shards(raw)
    serving = build(auxiliary_cubin,Layout(max_tokens=capacity,max_seqs=max_seqs,max_context=context),
                    peers=cubin("dsv41_engram_peers") if engram else None)
    buffers = {**dense_buffers,**expert_buffers,**barrier_buffers,**constants["buffers"]}
    pieces.fixed((constants["modules"],constants["ops"]))
    programs = {"load":{"once":True,"calls":constants["calls"]+dense_load+expert_load+barrier_load}}
    vocab = raw["embed.weight"]["shape"][0]
    buffers["next_token"] = {"kind":"output","dtype":"i64","shape":["seqs"],
                              "fill":"tokens","domain":{"index_into":"embed.weight"}}
    forwards, contexts, commits = {}, {}, []
    for mode in ("prefill","decode","verify"):
        target = forward(pieces,serving,dense_layouts,expert_layouts,mode=mode,ids="verify_ids" if mode=="verify" else "input_ids",
                         capacity=capacity,pool_tokens=context,constants=constants,cubin_dir=cubin_dir,
                         auxiliary_cubin=auxiliary_cubin,attention_cubin=None,fused_cubin=fused_cubin,
                         head_cubin=head_cubin,dense_cubin=cubin("dsv41_paged_indexer"),
                         sparse_cubin=cubin("dsv41_sparse_indexer"),select_cubin=cubin("dsv41_deepselect"),
                         candidate_cubin=cubin("dsv41_candidate"),
                         capture_tap=lambda layer,hc: capture(pieces,hc,layer,mode=mode,rows="tokens",
                                                              capacity=capacity,auxiliary_cubin=auxiliary_cubin),
                         engram=engram,device_sms=device_sms)
        output = head(pieces,target.normalized,"verify_tokens" if mode=="verify" else "next_token",mode=mode,capacity=capacity,vocab=vocab,
                      head_cubin=head_cubin,copy_cubin=cubin("copy_rows"))
        context_stage = publish(pieces,serving,dense_layouts,mode=mode,capacity=capacity,
                                auxiliary_cubin=auxiliary_cubin,cubin_dir=cubin_dir,
                                cos_sin=constants["rope"]["window"]["interleaved"])
        for stage in (target.lowered,output,context_stage):
            buffers.update(stage.buffers)
        forwards[mode] = target.lowered.calls+output.calls
        contexts[mode] = context_stage.calls
        commits.extend(target.commit)
    draft = draft_forward(pieces,serving,dense_layouts,expert_layouts,max_seqs=max_seqs,pool_tokens=context,
                          constants=constants,cubin_dir=cubin_dir,auxiliary_cubin=auxiliary_cubin,
                          attention_cubin=None,fused_cubin=fused_cubin,head_cubin=head_cubin,vocab=vocab,device_sms=device_sms)
    buffers.update(draft.buffers)
    for name,kind,width in (("draft_ids","workspace",5),("verify_ids","workspace",6),
                            ("draft_tokens","workspace",5),("verify_tokens","output",6)):
        buffers[name] = {"kind":kind,"dtype":"i64","shape":["seqs",width],
                         "domain":{"index_into":"embed.weight"}}
    buffers["verify_tokens"]["fill"] = "tokens"
    buffers["nacc"] = {"kind":"output","dtype":"i32","shape":["seqs"],"fill":"count",
                        "domain":{"min":1,"max":6}}
    spec = cubin("dsv41_spec_round")
    pieces.fixed(({"dsv41_spec_round":{"source":spec.name,"sha256":hashlib.sha256(spec.read_bytes()).hexdigest()}},
                  spec_ops("dsv41_spec_round")))
    programs = assemble(Stages(programs["load"]["calls"],forwards["prefill"],forwards["decode"],draft.calls,
                               forwards["verify"],contexts["prefill"],contexts["decode"],contexts["verify"],commits),
                        max_seqs=max_seqs)
    aux = serving.pieces(programs)
    buffers.update(aux["buffers"])
    for name in ("input_ids","anchor_token"):
        buffers[name] = {**buffers[name],"domain":{"index_into":"embed.weight"}}
    used = {a["buf"] for p in programs.values() for c in p["calls"] for a in c["args"] if "buf" in a}
    for name in sorted(used - buffers.keys()):
        buffers[name] = raw[name]
    for b in list(buffers.values()):
        if b.get("of") in raw:
            used.add(b["of"])
            buffers.setdefault(b["of"],raw[b["of"]])
        target = b.get("domain",{}).get("index_into")
        if target in raw:
            used.add(target)
            buffers.setdefault(target,raw[target])
    used_ops = {c["op"] for p in programs.values() for c in p["calls"]}
    ops = {name:op for name,op in {**pieces.ops,**aux["ops"]}.items() if name in used_ops}
    used_modules = {launch["module"] for op in ops.values() for launch in op["impl"]["launches"] if "module" in launch}
    modules = dict(pieces.modules)
    for name,module in aux["modules"].items():
        if name in modules and modules[name] != module:
            raise ValueError(f"module changed during generation: {name}")
        modules[name] = module
    return {"schema_version":SCHEMA_VERSION,"model":"DeepSeek-V4.1-Flash",
            "topology":{"groups":{"ep":4}},"vars":aux["vars"],"states":aux["states"],
            "buffers":{name:b for name,b in buffers.items() if name in used},
            "modules":{name:m for name,m in modules.items() if name in used_modules},
            "ops":ops,"programs":programs}


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bindings",type=Path)
    source.add_argument("--checkpoint",type=Path,help="read original shard headers directly; no weight export")
    parser.add_argument("--kernels",type=Path,default=Path(__file__).resolve().parents[1]/"build",
                        help="where scripts/build.py and engram_constants.py wrote the cubins (default: build/)")
    parser.add_argument("--constants",type=Path,help="engram_constants.json (default: in --kernels)")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--engram",choices=("host","device"),default="host",
                        help="where the Engram tables live: one copy in host memory (GB300), or sharded into HBM across the EP group (HGX B300)")
    parser.add_argument("--capacity",type=int,default=128)
    parser.add_argument("--max-seqs",type=int,default=16)
    parser.add_argument("--context",type=int,default=1048576,help="page-table bound per sequence, in tokens (the model's 1M)")
    parser.add_argument("--sms",dest="device_sms",type=int,default=152,choices=(148,152),
                        help="SMs per GPU of the device served: GB300 152, B300 148")
    parser.add_argument("--bundle",type=Path,help="copy pinned cubins to a serving artifact directory")
    args = vars(parser.parse_args())
    binding_file, checkpoint, engram = args.pop("bindings"), args.pop("checkpoint"), args.pop("engram")
    if checkpoint is not None:
        from dsv41.inventory import inventory
        from dsv41.weights import bindings as bind
        config = json.loads((checkpoint/"config.json").read_text())
        if config.get("model_type") != "deepseek_v41":
            parser.error("expected a deepseek_v41 checkpoint")
        tensors, _ = inventory(checkpoint)
        gpu, host = bind(tensors,config["text_config"],ep=4,engram=engram)
        bindings = {"gpu":gpu,"host":host}
    else:
        if engram != "host":
            parser.error("--engram device reads the checkpoint (--checkpoint)")
        bindings = json.loads(binding_file.read_text())
    constants_file = args.pop("constants") or args["kernels"]/"engram_constants.json"
    constants = json.loads(constants_file.read_text())
    output = args.pop("out")
    destination = args.pop("bundle")
    host = {name:dict(spec,placement="host") for name,spec in bindings["host"].items()}
    manifest = generate({**bindings["gpu"],**host},constants,**args)
    if destination is not None:
        bundle(manifest,destination,[args["kernels"],constants_file.parent])
    output.write_text(json.dumps(manifest,indent=2)+"\n")
    print({name:len(p["calls"]) for name,p in manifest["programs"].items()})


if __name__ == "__main__":
    main()
