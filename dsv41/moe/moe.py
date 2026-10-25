"""EP4 fused routed FP4 + shared FP8 MegaMoE, private TMA ABI.

Public first five params: y, cumulative_stats, token_count, slab_peers, rank.
Then routed l1, l1_sf, w1, w1_sf, l2, l2_sf, w2, w2_sf;
then shared x, x_sf, w1, w1_sf, l2, l2_sf, w2, w2_sf.
Slab region pointers are supplied with call offsets from layout.offsets.
"""
from itertools import groupby
from pathlib import Path
import hashlib,json,math

from .when import gated

BOUNDS={384:(8192,32768),128:(8192,)}
TOPK={384:6,128:3}

def block_m(tokens,experts):
 """Upstream's block M for one rank's call of `tokens` rows (DeepGEMM get_block_config_for_mega_moe).

 Every M block reloads its expert's weight tiles, so the rows an expert expects,
 plus one sigma, go in the fewest blocks of the smallest tile that covers them.
 """
 expected=tokens*4*TOPK[experts]/experts;covered=expected+math.sqrt(expected)
 if expected<=24:
  return 16 if expected<=10 else 32
 blocks=math.ceil(covered/240)
 if blocks==1 and covered>192 and experts//4<14:
  blocks=2
 return next((b for b in (64,128,192,240) if blocks*b>=covered),240)

def segments(experts,capacity):
 """`(block_m, first, last)` for each run of rows in 1..capacity that one block M serves."""
 runs=[(b,[t for t in g]) for b,g in groupby(range(1,capacity+1),lambda t:block_m(t,experts))]
 return [(b,ts[0],ts[-1]) for b,ts in runs]

def gates(rows,experts,capacity):
 """`(block_m, when)` per launch: which instance serves a call of `rows` rows."""
 return gated(rows,segments(experts,max(capacity,rows) if isinstance(rows,int) else capacity))

def bound(experts,capacity):
 """The smallest token bound an instance is built for that holds `capacity` rows a rank."""
 fits=[t for t in BOUNDS[experts] if capacity<=t]
 if capacity<1 or not fits:
  raise ValueError(f"MegaMoE e{experts} instances hold 1..{BOUNDS[experts][-1]} rows a rank, got {capacity}")
 return fits[0]

def pieces(experts=384,cubin_dir=None,rows="tokens",device_sms=152,capacity=8192):
 root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[2]/'build'
 # The dispatch grid-syncs over every SM and the ring holds a rank's bound of
 # tokens, so the instance, the grid and the slab geometry follow both.
 tokens=bound(experts,capacity)
 lay=json.loads((root/f'moe_layout_{experts}_sm{device_sms}_t{tokens}.json').read_text());local=experts//4
 modules={'dsv41_mega_moe':{'source':str((Path(cubin_dir) if cubin_dir else Path('build'))/'dsv41_mega_moe.cubin'),'sha256':hashlib.sha256((root/'dsv41_mega_moe.cubin').read_bytes()).hexdigest()}}
 def tm(p,d,dims,strides,box,swizzle=0):return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':p,'dtype':d,'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':256}}]}}
 def maps(base,rows,sf_rows,groups,weight_dtype,b):
  # A tile's rows: half a block (A multicasts over the 2-CTA cluster), the
  # epilogue's store block, and the block padded to the 128-row SF granule.
  h,i=5120,2304;stride=lambda k:k//2 if weight_dtype=='u4' else k;load,store,sf=b//2,lay['blocks'][str(b)]['store_m'],lay['blocks'][str(b)]['sf_m']
  return [tm(base,'u8',[h,rows],[h],[128,load],128),tm(base+1,'i32',[sf_rows,h//128],[sf_rows*4],[sf,1]),tm(base+2,weight_dtype,[h,groups*2*i],[stride(h)],[128,128],128),tm(base+3,'i32',[2*i,groups*h//128],[2*i*4],[128,1]),tm(base+4,'u8',[i,rows],[i],[64,store],64),tm(base+4,'u8',[i,rows],[i],[128,load],128),tm(base+5,'i32',[sf_rows,i//128],[sf_rows*4],[sf,1]),tm(base+6,weight_dtype,[i,groups*h],[stride(i)],[128,128],128),tm(base+7,'i32',[h,groups*i//128],[h*4],[128,1])]
 params=['out buffer<bf16>','inout buffer<i32>','i32','in buffer<u64>','i32']+['inout buffer<u8>','inout buffer<u8>','in buffer<u8>','in buffer<i32>','inout buffer<u8>','inout buffer<u8>','in buffer<u8>','in buffer<i32>']*2
 params[11]='in buffer<i8>';params[19]='in buffer<fp8e4m3>'
 def launch(b,when):
  l={'module':'dsv41_mega_moe','entry':f'dsv41_mega_moe_e{experts}_r4_sm{device_sms}_t{tokens}_b{b}','params':params[:5]+['bytes<128>']*18,'args':[{'param':i} for i in range(5)]+maps(5,lay['ring'],lay['sf_ring'],local,'u4',b)+maps(13,tokens,lay['shared_sf_rows'],1,'u8',b),'block':[512,1,1],'grid':[device_sms,1,1],'cluster':[2,1,1],'shared_mem':lay['blocks'][str(b)]['smem'],'pdl':True}
  return {**l,'when':when} if when else l
 op={'params':params,'impl':{'launches':[launch(b,when) for b,when in gates(rows,experts,capacity)]}}
 modules['dsv41_moe_stage']={'source':str((Path(cubin_dir) if cubin_dir else Path('build'))/'dsv41_moe_stage.cubin'),'sha256':hashlib.sha256((root/'dsv41_moe_stage.cubin').read_bytes()).hexdigest()}
 stage={'params':['in buffer<bf16>','out buffer<u8>','out buffer<u8>','i32','i32','i32','i32','out buffer<u8>','i32'],'impl':{'launches':[{'module':'dsv41_moe_stage','entry':'dsv41_mega_quant_x','block':[256,1,1],'grid':[{'ceil_div':[{'mul':[rows,40]},8]},1,1]}]}}
 return modules,{f'dsv41_mega_moe_e{experts}':op,'dsv41_moe_quant_x':stage},lay
