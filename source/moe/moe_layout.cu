#include "mega_config.cuh"
#include <cstdio>
#include <cstdlib>
template<int E,int K,int R,int S,int T,int B>void block(){using C=dsv41::Config<E,K,R,S,T,B>;printf("\"%d\":{\"store_m\":%d,\"sf_m\":%d,\"stages\":%d,\"smem\":%d}",B,C::StoreM,C::SfM,C::Stages,C::Smem);}
template<int E,int K,int R,int S,int T>void dump(){using C=dsv41::Config<E,K,R,S,T>;using namespace deep_gemm;auto b=layout::MegaMoEBuffer(nullptr,5120,2304,R,E,T,K,C::Ring,C::SfRing,true,1);
 printf("{\"experts\":%d,\"tokens\":%d,\"topk\":%d,\"ranks\":%d,\"ring\":%d,\"sf_ring\":%d,\"slab_bytes\":%llu,\"workspace_bytes\":%llu,\"shared_sf_rows\":%d,\"blocks\":{",E,T,K,R,C::Ring,C::SfRing,(unsigned long long)b.get_num_bytes(),(unsigned long long)b.workspace.get_end_ptr(),layout::get_num_max_shared_sf_tokens(T));
 block<E,K,R,S,T,16>();printf(",");block<E,K,R,S,T,32>();printf(",");block<E,K,R,S,T,64>();printf(",");block<E,K,R,S,T,128>();printf(",");block<E,K,R,S,T,192>();printf(",");block<E,K,R,S,T,240>();
 printf("},\"offsets\":{");
#define FIELD(NAME,MEMBER) printf("\"" NAME "\":%llu,",(unsigned long long)b.MEMBER.base)
 FIELD("x",input_token_buffer);FIELD("x_sf",input_sf_buffer);FIELD("idx",input_topk_idx_buffer);FIELD("weights",input_topk_weights_buffer);FIELD("shared_x_sf",shared_l1_sf_buffer);FIELD("shared_l2",shared_l2_token_buffer);FIELD("shared_l2_sf",shared_l2_sf_buffer);FIELD("l1",l1_token_buffer);FIELD("l1_sf",l1_sf_buffer);FIELD("l2",l2_token_buffer);
 printf("\"l2_sf\":%llu}}\n",(unsigned long long)b.l2_sf_buffer.base);
}
int main(int argc,char**argv){if(argc!=4)return 2;int e=atoi(argv[1]),s=atoi(argv[2]),t=atoi(argv[3]);
#define CASE(E,K,S,T) if(e==E&&s==S&&t==T){dump<E,K,4,S,T>();return 0;}
 CASE(384,6,152,8192)CASE(128,3,152,8192)CASE(384,6,148,8192)CASE(128,3,148,8192)
 CASE(384,6,152,32768)CASE(384,6,148,32768)
 return 2;}
