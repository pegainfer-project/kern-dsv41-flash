"""Unmodified MegaGate: x, BF16 gate weight, correction bias, indices, weights, n."""
from pathlib import Path
import hashlib

from .when import gated

SMEM_CAPACITY=232448
# (block tokens, stages, gate threads, MMA CTAs, split-K) per run of rows, by
# expert count. 384: upstream's heuristic picks (`heuristics/mega_gate.hpp`)
# at the run's typical size: split-K 8 over 16-token tiles while a call fits
# one wave, then 2-CTA tiles of 176 tokens split twice, then 224 unsplit. Stages
# fill shared memory. The 128-expert draft keeps its four-stage instance.
RUNS={384:[((16,12,128,1,8),1,1024),((176,11,896,2,2),1025,5200),((224,10,896,2,1),5201,1<<20)],
      128:[((16,4,128,1,1),1,1<<20)]}
# The split-K 1 instance that streams each group's 3.9 MB weight through one CTA.
LEGACY={384:(16,4,128,1,1),128:(16,4,128,1,1)}

def align(x,a):return (x+a-1)//a*a

def groups(experts):return 3 if experts==384 else 1

def smem_bytes(experts,block_tokens,stages,mma,split_k):
    """sizeof(SharedStorage) after stage merging: x and weight stages, barriers, tmem pointer, bias cache."""
    atoms=80//split_k
    merge=3 if stages//3>=8 and atoms%3==0 else 2 if stages//2>=8 and atoms%2==0 else 1
    rows=block_tokens//mma+experts//groups(experts)//mma
    tail=align(stages//merge*rows*64*merge*2+8*(2*(stages//merge)+5)+4,16)+4*experts
    return align(tail,1024)

def sms(experts,mma,split_k,device_sms):
    """A persistent grid of whole worker groups, one logical CTA per (group, MMA CTA, split)."""
    tasks=groups(experts)*mma*split_k
    return device_sms//tasks*tasks

def pieces(rows='tokens', experts=384, cubin_dir=None, max_tokens=8192, raw_outputs=False, config=None, device_sms=152):
    assert experts in (128,384)
    capacity=rows if isinstance(rows,int) else max_tokens
    runs=[(c,first,min(last,capacity)) for c,first,last in ([(config,1,capacity)] if config else RUNS[experts]) if first<=capacity]
    topk=6 if experts==384 else 3;g=groups(experts)
    root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[2]/'build'
    modules={name:{'source':str((Path(cubin_dir) if cubin_dir else Path('build'))/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()} for name,file in [('dsv41_gate','dsv41_gate.cubin'),('dsv41_moe_boundary','dsv41_moe_boundary.cubin')]}
    def tm(p,n,m):return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':p,'dtype':'bf16','dims':[5120,n],'strides':[10240],'box':[64,m],'swizzle':128}}]}}
    ptr=lambda i:{'param':i};nil={'i64':0}
    abi=['bytes<128>']*2+['in buffer<f32>','i64','i64','i64','i64','i64','out buffer<i64>','i64','out buffer<f32>','out buffer<u8>','inout buffer<u64>']+['i32']*4+['f32','i32','i64','i64','i64']
    def launch(c,when):
        bt,stages,threads,mma,split_k=c;n=sms(experts,mma,split_k,device_sms)
        args=[tm(0,capacity,bt//mma),tm(1,experts,experts//g//mma),ptr(2),nil,nil,nil,nil,nil,ptr(3),nil,ptr(4),{'scratch':'scores'},{'param':6},ptr(5),{'i32':experts},{'i32':0},{'i32':0},{'f32':1.5},{'i32':0},{'i64':0},nil,nil]
        entry=f'_ZN9deep_gemm25sm100_bf16_mega_gate_implILj5120ELj{experts}ELj{bt}ELj{stages}ELj{threads}ELj{mma}ELj{split_k}ELj{g}ELj{n}ELj{topk}ELj1ELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv14CUtensorMap_stS1_PKfS3_PKbS5_PKiS7_PlS8_PfPvSA_jjjjfilS5_S5_'
        l={'module':'dsv41_gate','entry':entry,'params':abi,'args':args,'block':[128+threads,1,1],'grid':[n,1,1],
           **({'cluster':[mma,1,1]} if mma>1 else {}),'shared_mem':smem_bytes(experts,bt,stages,mma,split_k),'pdl':True}
        return {**l,'when':when} if when else l
    scores=max(-(-capacity//c[0])*c[4]*c[0] for c,_,_ in runs)*experts*4
    op={'params':['in buffer<bf16>','in buffer<bf16>','in buffer<f32>','out buffer<i64>','out buffer<f32>','i32','inout buffer<u64>'],'impl':{'scratch':{'scores':{'dtype':'u8','shape':[scores]}},'launches':[launch(c,when) for c,when in gated(rows,runs)]}}
    if raw_outputs:
        # Direct writes to exported byte slab offsets; pointer ABI is unchanged.
        op['params'][3:5]=['out buffer<u8>']*2
        abi[8]=abi[10]='out buffer<u8>'
    return modules,{f'dsv41_gate_e{experts}':op}
