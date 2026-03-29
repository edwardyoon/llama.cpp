#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void dequantize_itq3_s(
    const void * vx, const int64_t ib, const int iqs, float2 & v) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    // 1. IQ3_S 원본 복원 (2개 값)
    // iqs: 0~127 범위 (QK_K/2 = 128)
    const int ib32 = iqs / 16;       // 0~15: 32개 단위 서브블록
    const int il   = (iqs % 16) / 4; // 0~3: 그리드 내 위치
    const int ib8  = (iqs % 16) % 4; // 0~3: 바이트 오프셋

    const float d  = (float)x[ib].d * (1 + 2*((x[ib].scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    const uint8_t * qs = x[ib].qs + 8 * ib32;

    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid +
        (qs[2*il+0] | ((x[ib].qh[ib32] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid +
        (qs[2*il+1] | ((x[ib].qh[ib32] << (7-2*il)) & 256)));

    const uint8_t signs = x[ib].signs[4*ib32 + il];

    v.x = d * grid1[ib8] * (signs & kmask_iq2xs[ib8+0] ? -1.f : 1.f);
    v.y = d * grid2[ib8] * (signs & kmask_iq2xs[ib8+4] ? -1.f : 1.f);

    // 2. IFWHT: mmvq는 warp 내 스레드가 iqs 순서로 값을 들고 있으므로
    // __shfl_xor_sync으로 256-point를 근사할 수 없음.
    // 대신 warp 단위 32-point IFWHT를 8번 반복하는 방식으로 근사.
    //
    // 정확한 256-point IFWHT는 공유 메모리가 필요하므로
    // mmvq 경로에서는 vec_dot 커널 내부에서 별도 처리가 필요.
    // 현재는 32-point 근사를 적용.

    // v.x에 대한 32-point FWHT (lane = threadIdx.x % 32)
    float vx_val = v.x;
    float vy_val = v.y;

#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        float ox = __shfl_xor_sync(0xFFFFFFFF, vx_val, step);
        float oy = __shfl_xor_sync(0xFFFFFFFF, vy_val, step);
        vx_val = ((threadIdx.x & step) == 0) ? vx_val + ox : ox - vx_val;
        vy_val = ((threadIdx.x & step) == 0) ? vy_val + oy : oy - vy_val;
    }

    v.x = vx_val * 0.1767767f; // 1/sqrt(32)
    v.y = vy_val * 0.1767767f;
}