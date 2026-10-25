"""Upstream direct paged dense MXFP4 index scoring and projection-weight adapter."""
import hashlib
from pathlib import Path


def definitions(cubin,*,rows,rows_max,kv_rows_max,page_size=64,pages,page_cols,
                cache_state=True,page_stride=None,scale_dtype='fp8e8m0',clean=False):
    """Qpacked,QS,weights F32,end,page_table,request,cache,cache_scales,out F32,rows.

    Q[rows,32,64], QS[rows,32,4] E8M0, weights[rows,32] already scaled1/64.
    Each query has a page-table row[page_cols], an end length and a request:
    a run of rows of one request scores as one varlen request through its
    first row's table out to its last row's end, so a 4-row Q block reads each
    KV split once instead of once a row.
    Cache pages: data[P,64], scales[P,4], trailing padding to512-byte stride.
    cache_scales aliases SAME state at byte offset P*64, not a separate allocation.
    Scores are logical key columns[rows,kv_rows_max]; a row's columns from its
    end on hold its request's later keys, or -infinity when `clean`.
    pages is physical allocation capacity, kv_rows_max padded256, rows_max padded4.
    An expression `rows` may be 0 (a tile past the call's rows): every launch's
    grid is then 0, the persistent ones' through `live`.
    """
    meta_smem=(2*rows_max+33)*4
    if rows_max%4 or kv_rows_max%256 or page_size not in (64,128) or meta_smem>232448:
        raise ValueError('rows capacity4 within metadata smem, key capacity256, page64/128 required')
    stride=page_stride or ((page_size*68+511)//512)*512
    if stride<page_size*68 or stride%512:raise ValueError('cache stride must align512')
    def tm(param,dtype,dims,strides,box,swizzle=0):
        return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':param,'dtype':dtype,
            'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':256}}]}}
    live=1 if isinstance(rows,int) else {'clamp':[rows,0,1]}
    cache='in state' if cache_state else 'in buffer<u8>'
    params=['in buffer<u8>',f'in buffer<{scale_dtype}>','in buffer<f32>','in buffer<i32>',
            'in buffer<i32>','in buffer<i32>',cache,cache,'out buffer<f32>','i32']
    init={'module':'dsv41_paged_indexer','entry':'dsv41_dense_initialize',
          'grid':[{'mul':[rows,kv_rows_max//256]},1,1],'block':[256,1,1],
          'params':['out buffer<f32>','i32','i32'],
          'args':[{'param':8},{'param':9},{'i32':kv_rows_max}]}
    meta={'module':'dsv41_paged_indexer','entry':'_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj1ELb0ELb1ELj256ELj148EEEvjjPKjS3_Pj','grid':[live,1,1],
          'block':[1024,1,1],'shared_mem':meta_smem,'pdl':True,
          'params':['i32','i32','in buffer<i32>','in buffer<i32>','out buffer<u8>'],
          'args':[{'i32':rows_max},{'param':9},{'param':3},{'param':5},{'scratch':'metadata'}]}
    score={'module':'dsv41_paged_indexer','entry':f'_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj32ELj128ELj{page_size}ELb1ELb0ELb1ELj3ELj10ELj256ELj16ELj128ELj256EN7cutlass12float_e2m1_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_',
           'grid':[148 if live==1 else {'mul':[live,148]},1,1],'block':[384,1,1],'shared_mem':202240,'pdl':True,
           'params':['i32','i32','i32','in buffer<i32>','out buffer<f32>','in buffer<i32>','in buffer<i32>','in buffer<u8>']+['bytes<128>']*5,
           'args':[{'param':9},{'i32':kv_rows_max},{'i32':page_cols},{'param':3},{'param':8},{'param':4},{'param':5},{'scratch':'metadata'},
                   tm(0,'u4packed',[128,rows_max*32],[64],[128,128],64),
                   tm(1,'u32',[32,rows_max],[128],[32,4]),
                   tm(6,'u4packed',[128,page_size,0 if cache_state else pages],[64,stride],[128,page_size,1],64),
                   tm(7,'u32',[page_size,0 if cache_state else pages],[stride],[page_size,1]),
                   tm(2,'f32',[32,rows_max],[128],[32,4])]}
    mask={'module':'dsv41_paged_indexer','entry':'dsv41_dense_mask',
          'grid':init['grid'],'block':[256,1,1],
          'params':['inout buffer<f32>','in buffer<i32>','i32','i32'],
          'args':[{'param':8},{'param':3},{'param':9},{'i32':kv_rows_max}]}
    op={'params':params,'impl':{'scratch':{'metadata':{'dtype':'u8','shape':[149*8]}},'launches':[init,meta,score,mask] if clean else [meta,score]}}
    return module(cubin),{'dsv41_paged_index_scores':op}


def weights(cubin,*,rows):
    """Projection BF16 -> sparse BF16 and dense F32 weights, both scaled 1/64, rows."""
    op={'params':['in buffer<bf16>','out buffer<bf16>','out buffer<f32>','i32'],
        'impl':{'launches':[{'module':'dsv41_paged_indexer','entry':'dsv41_index_weights',
               'grid':[{'ceil_div':[{'mul':[rows,32]},256]},1,1],'block':[256,1,1]}]}}
    return module(cubin),{'dsv41_index_weights':op}


def module(cubin):
    cubin=Path(cubin)
    return {'dsv41_paged_indexer':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
