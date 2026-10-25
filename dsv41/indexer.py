"""Paged two-level Indexer call lowering, with shared candidate block indices."""
from dataclasses import dataclass

from .forward import Lowered, projection, bf16_projection, scalar, selected, align4
from .head import offset
from .programs import buf, call, integer
from .auxiliary.ops import definitions as auxiliary_ops
from .attention.paged_indexer_ops import definitions as dense_ops, weights as weight_ops
from .attention.paged_sparse_ops import definitions as sparse_ops
from .attention.select_ops import definitions as select_ops
from .attention.candidate_ops import definitions as candidate_ops


# Dense scores are f32 [rows, context / ratio]: at 32k rows over a 1M context
# that is 128 GiB a layer. They are made a query-row tile at a time into one
# buffer per width, so their memory is a tile's whatever the capacity.
TILE_BYTES = 4 << 30


def tiles(capacity, row_bytes):
    """(first row, rows) of each query-row tile whose scores fit TILE_BYTES."""
    n = max(4, min(capacity, TILE_BYTES // row_bytes // 4 * 4))
    return [(start, min(n, capacity - start)) for start in range(0, capacity, n)]


@dataclass(frozen=True)
class Indexed:
    lowered: Lowered
    logical: str
    physical: str
    candidates: str | None


def forward(pieces, serving, layer, source, layouts, hidden, qr, *, mode,
            capacity, pool_tokens, cos_sin, metadata, cubin_dir,
            auxiliary_cubin, dense_cubin, sparse_cubin, select_cubin,
            candidate_cubin, candidates=None):
    """metadata names: ends, pages, request_ids (one virtual request per row).

    Source denotes the shared index-K owner, not the current indexer layer.
    Layer20 publishes candidate blocks; later indexers consume that allocation.
    """
    rows = serving.rows(mode)
    capacity = align4(capacity)
    ratio = dict(serving.layout.sources)[source]
    page = serving.layout.page_size // ratio
    width = ((serving.layout.max_context // ratio + 255) // 256) * 256
    prefix = f"{mode}.index{layer}"
    weight = f"layers.{layer}.attn.indexer"
    buffers, calls = {}, []

    def allocate(name, dtype, columns):
        name = prefix + "." + name
        buffers[name] = {"dtype":dtype,"kind":"workspace","shape":[capacity,columns]}
        return name

    def tile(name, dtype, rows, columns):
        # A score tile: every layer of one shape runs one after another, so
        # they share it (a layer-less name); each mode's row bound is its own.
        name = f"index.{name}.{rows}x{columns}"
        buffers[name] = {"dtype":dtype,"kind":"workspace","shape":[rows,columns]}
        return name

    def arg(b):
        return buf(b) if isinstance(b, str) else b

    def extend(stage):
        buffers.update(stage.buffers)
        calls.extend(stage.calls)

    query = prefix + ".query"
    extend(projection(pieces,prefix+".query",layouts[weight+".wq_b"],qr,query,
                      rows=rows,row_capacity=capacity,workspace=prefix+".query_quant",cubin_dir=cubin_dir))
    rotated = allocate("rotated","bf16",4096)
    rope = pieces.add(selected(auxiliary_ops(auxiliary_cubin,rows=rows,heads=32),"rope"))["rope"]
    calls.append(call(prefix+".rope",rope,buf(rotated),buf(query),buf(cos_sin),
                      buf(mode+".position"),scalar(rows),integer(32),integer(128),integer(64),integer(0)))
    packed = allocate("packed","u8",2048)
    scales = allocate("scales","fp8e8m0",128)
    dequant = allocate("dequant","bf16",4096)
    quant_rows = {"mul":[rows,32]}
    quant = pieces.add(selected(auxiliary_ops(auxiliary_cubin,rows=quant_rows),"index_quant"))["index_quant"]
    calls.append(call(prefix+".quant",quant,buf(packed),buf(scales),buf(dequant),buf(rotated),scalar(quant_rows),integer(128)))
    raw_weights = prefix + ".weights_raw"
    extend(bf16_projection(pieces,prefix+".weights",hidden,weight+".weights_proj.weight",
                           raw_weights,rows=rows,capacity=capacity,n=32,k=5120))
    def dense(rows, rows_max):
        # Layer 20's candidate blocks max over 8-key blocks, so its rows read
        # past their end: only there do the tails need -infinity.
        return dense_ops(dense_cubin,rows=rows,rows_max=rows_max,kv_rows_max=width,
                         page_size=page,pages=pool_tokens//serving.layout.page_size,
                         page_cols=serving.layout.pages,clean=layer==20)
    names = pieces.add(weight_ops(dense_cubin,rows=rows))
    weights_bf16 = allocate("weights_bf16","bf16",32)
    weights_f32 = allocate("weights_f32","f32",32)
    calls.append(call(prefix+".scale_weights",names["dsv41_index_weights"],
                      buf(raw_weights),buf(weights_bf16),buf(weights_f32),scalar(rows)))
    logical = allocate("logical","i32",512)

    def select(scores, ends, output, stride, topk, dtype, label, rows=rows, rows_max=capacity):
        name = pieces.add(select_ops(select_cubin,rows=rows,rows_max=rows_max,
                          width=stride,topk=topk,dtype=dtype))["dsv41_select"]
        calls.append(call(prefix+"."+label,name,arg(scores),arg(ends),arg(output),scalar(rows)))

    if layer > 20:
        if candidates is None:
            raise ValueError("later indexers require layer20 candidate blocks")
        sparse = pieces.add(sparse_ops(sparse_cubin,rows=rows,rows_max=capacity,
                            page_size=page,page_cols=serving.layout.pages,scale_dtype="fp8e8m0"))
        scores = allocate("scores","bf16",16384)
        calls.append(call(prefix+".score",sparse["dsv41_paged_sparse_scores"],buf(packed),buf(scales),
                          buf(weights_bf16),buf(metadata["ends"]),buf(metadata["pages"]),
                          buf(metadata["request_ids"]),buf(candidates),{"state":f"index_k.{source}"},buf(scores),scalar(rows)))
        # Sparse scores use candidate-slot coordinates, so selection spans all
        # 16384 columns; sparse_positions removes masked and missing candidates.
        slots = allocate("selected_slots","i32",512)
        select(scores,metadata["sparse_ends"],slots,16384,512,"bf16","select_sparse")
        calls.append(call(prefix+".positions",sparse["dsv41_sparse_positions"],buf(scores),buf(slots),
                          buf(candidates),buf(metadata["ends"]),buf(logical),scalar(rows)))
    else:
        cache = {"state":f"index_k.{source}"}
        block_stride = ((width//8+255)//256)*256
        spans = tiles(capacity,width*4)
        scores = tile("scores","f32",spans[0][1],width)
        if layer == 20:
            block_scores = tile("block_scores","f32",spans[0][1],block_stride)
            block_ends = allocate("block_ends","i32",1)
            selected_blocks = allocate("selected_blocks","i32",2048)
            candidates = allocate("candidates","i32",2048)
        for k, (first, n) in enumerate(spans):
            t = {"clamp":[rows,first,n]} if len(spans) > 1 else rows
            def at(name, row_bytes):
                return offset(name,first*row_bytes) if first else buf(name)
            ends = at(metadata["ends"],4)
            score = pieces.add(dense(t,n))["dsv41_paged_index_scores"]
            calls.append(call(prefix+f".score.t{k}",score,at(packed,2048),at(scales,128),
                              at(weights_f32,128),ends,at(metadata["pages"],serving.layout.pages*4),
                              at(mode+".index_runs",4),cache,
                              {"state":f"index_k.{source}","offset":page*64},buf(scores),scalar(t)))
            select(scores,ends,at(logical,2048),width,512,"f32",f"select_tokens.t{k}",rows=t,rows_max=n)
            if layer == 20:
                adapters = pieces.add(candidate_ops(candidate_cubin,rows=t,width=width,block_stride=block_stride))
                calls.append(call(prefix+f".block_scores.t{k}",adapters["dsv41_candidate_scores"],buf(scores),
                                  ends,buf(block_scores),at(block_ends,4),scalar(t)))
                select(block_scores,at(block_ends,4),at(selected_blocks,8192),block_stride,2048,"f32",
                       f"select_blocks.t{k}",rows=t,rows_max=n)
                calls.append(call(prefix+f".filter_blocks.t{k}",adapters["dsv41_selection_filter"],buf(block_scores),
                                  at(selected_blocks,8192),at(candidates,8192),scalar(t)))
    physical = allocate("physical","i32",512)
    calls += serving.compressed_indices(mode,ratio,logical,physical)
    return Indexed(Lowered(buffers,calls),logical,physical,candidates)
