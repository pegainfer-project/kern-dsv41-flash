"""Forty-layer target forward, shared by prefill, decode and verification."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib

from .attention_forward import forward as attention
from .blocks import Blocks, Stream
from .compressed_attention import CompressedAttention
from .compression import publish
from .engram import inject
from .forward import Lowered, fp8_input, normalize, scalar, selected
from .head import definitions as head_ops
from .moe_forward import forward as moe, fp8_input as moe_fp8_input
from .programs import buf, call, integer


@dataclass(frozen=True)
class Target:
    lowered: Lowered
    normalized: str
    starts: str
    commit: list


DECODER = 20


def forward(pieces, serving, dense_layouts, expert_layouts, *, mode, ids,
            capacity, pool_tokens, constants, cubin_dir, auxiliary_cubin,
            attention_cubin, head_cubin, dense_cubin, sparse_cubin, select_cubin,
            candidate_cubin, capture_tap, fused_cubin=None, engram=None, device_sms=152,
            replay_capacity=None, copy_cubin=None):
    """Produce all token hidden rows; the caller chooses serving head rows.

    capture_tap(layer,materialized_hc,mode,capacity) returns a Lowered stage
    storing the attention-input mean in DSpark's concatenated target context
    allocation. Plain modes commit compressor state here; verify returns
    deferred commits. engram maps a layer to its table's shard geometry when
    the tables live in HBM (see engram.inject); None reads them from shared
    host memory.

    A prefill with `serving.layout.replay` is the causal encoder-decoder's:
    every row runs the 20 encoder layers and the decoder's global KV
    projection (layer 20's compressor and index keys over its attention
    input), then only each sequence's replay rows run the decoder, in
    `replay` mode, and the hidden rows handed back are theirs.
    """
    if mode not in ("prefill", "decode", "verify"):
        raise ValueError("target mode must be prefill, decode or verify")
    buffers, calls = {}, serving.prepare(mode,ids)
    calls += serving.engram_hash(mode,**constants["names"],
                                  compressed_pad_id=constants["metadata"]["compressed_pad_id"])
    commits = []

    def extend(stage):
        for name, spec in stage.buffers.items():
            if name in buffers and buffers[name] != spec:
                raise ValueError(f"target workspace shape collision: {name}")
            buffers[name] = spec
        calls.extend(stage.calls)

    def stage(m, cap, source=None):
        blocks = Blocks(pieces,prefix=m+".target",rows=serving.rows(m),capacity=cap,cubin_dir=cubin_dir,device_sms=device_sms)
        buffers.update(blocks.buffers())
        compressed = CompressedAttention(pieces,serving,dense_layouts,mode=m,capacity=cap,
                                         pool_tokens=pool_tokens,cos_sin=constants["rope"]["compressed"]["interleaved"],
                                         cubin_dir=cubin_dir,auxiliary_cubin=auxiliary_cubin,
                                         dense_cubin=dense_cubin,sparse_cubin=sparse_cubin,
                                         select_cubin=select_cubin,candidate_cubin=candidate_cubin,source=source)
        return m, serving.rows(m), cap, blocks, compressed

    m, rows, cap, blocks, compressed = stage(mode, capacity)
    embedding = m + ".target.embedding"
    buffers[embedding] = {"dtype":"bf16","kind":"workspace","shape":[capacity,5120]}
    embedding_op = pieces.add(selected(head_ops(head_cubin,seqs=rows),"head_embedding"))["head_embedding"]
    calls.append(call(m+".target.embed",embedding_op,buf(ids),integer(1),buf("embed.weight"),buf(embedding),integer(5120)))
    state, initial = blocks.initialize(embedding)
    calls.extend(initial)

    def replay(state, encoder):
        """Publish the decoder's global KV over every row, then move the stream to the replay rows."""
        layer = f"layers.{DECODER}"
        _, hidden, mix = encoder.sublayer(state,layer=layer,kind="attn",bank=1,
                                          update=encoder.name("attention_result"),
                                          fp8=fp8_input(mode+".target.attention",capacity)[1])
        calls.extend(mix)
        published = publish(pieces,serving,DECODER,hidden,mode=mode,capacity=capacity,
                            cos_sin=constants["rope"]["compressed"]["interleaved"],auxiliary_cubin=auxiliary_cubin)
        extend(published.lowered)
        calls.extend(serving.prepare("replay",ids))
        decoder = stage("replay",replay_capacity,source=DECODER)
        calls.extend(decoder[3].identity())
        moved = Stream(*(decoder[3].name(f.removeprefix(encoder.prefix+".")) for f in
                         (state.residual,state.update,state.post,state.comb,state.pre)))
        for source, target in zip((state.residual,state.update,state.post,state.comb,state.pre),
                                  (moved.residual,moved.update,moved.post,moved.comb,moved.pre)):
            extend(gather(pieces,source,target,buffers[source],rows=decoder[1],capacity=replay_capacity,copy_cubin=copy_cubin))
        return moved, decoder

    for layer_id in range(40):
        layer = f"layers.{layer_id}"
        if layer_id == DECODER and mode == "prefill" and serving.layout.replay:
            commits += compressed.commits
            state, (m, rows, cap, blocks, compressed) = replay(state, blocks)
        prefix = m + ".target"
        if layer_id in (1,14):
            state, flush = blocks.materialize(state,prefix+f".{layer}.engram_materialize")
            calls.extend(flush)
            output = prefix + ".engram_residual"
            extend(inject(pieces,serving,layer_id,dense_layouts,state.residual,output,
                          mode=m,capacity=cap,cubin_dir=cubin_dir,
                          shard=engram.get(layer_id) if engram else None))
            state = replace(state,residual=output)
        if layer_id in (37,38,39):
            state, flush = blocks.materialize(state,prefix+f".{layer}.tap_materialize")
            calls.extend(flush)
            extend(capture_tap(layer_id,state.residual,m,cap))
        rope = constants["rope"]["window" if layer_id < 2 else "compressed"]

        def attn(source, output):
            return attention(pieces,serving,layer,dense_layouts,source,output,mode=m,
                             prefix=prefix+".attention",capacity=cap,pool_tokens=pool_tokens,
                             cos_sin=rope["interleaved"],cubin_dir=cubin_dir,
                             auxiliary_cubin=auxiliary_cubin,attention_cubin=attention_cubin,
                             fused_cubin=fused_cubin,fused_cos_sin=rope["split"] if fused_cubin else None,
                             prepare_compressed=(lambda hidden,qr: compressed.prepare(layer_id,hidden,qr))
                             if layer_id >= 2 else None)

        def ffn(source, output):
            return moe(pieces,layer,expert_layouts,source,output,rows=rows,capacity=cap,
                       workspace=prefix+".moe",cubin_dir=cubin_dir,device_sms=device_sms)

        state, lowered = blocks.layer(state,layer,attn,ffn,
                                      attention_fp8=fp8_input(prefix+".attention",cap)[1],
                                      ffn_fp8=moe_fp8_input(prefix+".moe",capacity=cap,cubin_dir=cubin_dir,rows=rows,device_sms=device_sms))
        extend(lowered)

    collapsed, normalized = m+".target.collapsed", m+".target.head_hidden"
    buffers[collapsed] = {"dtype":"bf16","kind":"workspace","shape":[cap,5120]}
    calls.extend(blocks.head_input(state,collapsed))
    extend(normalize(pieces,m+".target.head_norm",collapsed,"norm.weight",normalized,
                     rows=rows,width=5120,capacity=cap,cubin=auxiliary_cubin))
    commits += compressed.commits
    if mode != "verify":
        calls.extend(commits)
    return Target(Lowered(buffers,calls),normalized,m+".starts",commits if mode == "verify" else [])


def gather(pieces, source, target, spec, *, rows, capacity, copy_cubin):
    """Copy a per-row buffer's replay rows out of the chunk they were staged in."""
    copy_cubin = Path(copy_cubin)
    modules = {"copy_rows":{"source":copy_cubin.name,"sha256":hashlib.sha256(copy_cubin.read_bytes()).hexdigest()}}
    dtype, width = spec["dtype"], spec["shape"][1]
    t = f"buffer<{dtype}>"
    op = {"params":[f"out {t}",f"in {t}","in buffer<i32>","i32"],
          "impl":{"launches":[{"module":"copy_rows","entry":"kern_gather_rows","grid":[rows,1,1],"block":[128,1,1]}]}}
    name = pieces.add((modules,{f"gather_rows_{dtype}":op}))[f"gather_rows_{dtype}"]
    size = {"bf16":2,"f32":4}[dtype]
    return Lowered({target:{"dtype":dtype,"kind":"workspace","shape":[capacity,width]}},
                   [call(target+".gather",name,buf(target),buf(source),buf("replay.source_rows"),integer(width*size))])


def head(pieces, hidden, output, *, mode, capacity, max_seqs, vocab, head_cubin, copy_cubin, starts=None):
    """Prefill projects each sequence's last row; decode and verify project every row."""
    prefix = mode + ".target_head"
    rows = "seqs" if mode == "prefill" else "tokens"
    head_capacity = max_seqs if mode == "prefill" else capacity
    buffers, calls = {}, []
    if mode == "prefill":
        copy_cubin = Path(copy_cubin)
        modules = {"copy_rows":{"source":copy_cubin.name,
                                 "sha256":hashlib.sha256(copy_cubin.read_bytes()).hexdigest()}}
        ops = {"last_rows":{"params":["out buffer<bf16>","in buffer<bf16>","in buffer<i32>","i32","i32"],
                            "impl":{"launches":[{"module":"copy_rows","entry":"kern_last_rows_bf16",
                                                   "grid":["seqs",1,1],"block":[256,1,1]}]}}}
        name = pieces.add((modules,ops))["last_rows"]
        last = prefix + ".last"
        buffers[last] = {"dtype":"bf16","kind":"workspace","shape":[max_seqs,5120]}
        calls.append(call(prefix+".last",name,buf(last),buf(hidden),buf(starts or mode+".starts"),integer(5120),integer(5120)))
        hidden = last
    names = pieces.add(head_ops(head_cubin,seqs=rows))
    logits = prefix + ".logits"
    buffers[logits] = {"dtype":"f32","kind":"workspace","shape":[head_capacity,vocab]}
    # The serving generator declares output geometry/fill; verification has
    # six predictions per request whereas the plain programs return one.
    calls.append(call(prefix+".project",names["head_project"],buf(hidden),buf("head.weight"),buf(logits),
                      scalar(rows),integer(vocab),integer(5120)))
    calls.append(call(prefix+".sample",names["head_plain"],buf(logits),buf(output),{"i64":vocab},integer(vocab)))
    return Lowered(buffers,calls)
