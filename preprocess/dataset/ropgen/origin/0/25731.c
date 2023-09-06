static int bayer_to_yv12_wrapper(SwsContext *c, const uint8_t* src[], int srcStride[], int srcSliceY,

                                 int srcSliceH, uint8_t* dst[], int dstStride[])

{

    const uint8_t *srcPtr= src[0];

    uint8_t *dstY= dst[0];

    uint8_t *dstU= dst[1];

    uint8_t *dstV= dst[2];

    int i;

    void (*copy)       (const uint8_t *src, int src_stride, uint8_t *dstY, uint8_t *dstU, uint8_t *dstV, int luma_stride, int width, int32_t *rgb2yuv);

    void (*interpolate)(const uint8_t *src, int src_stride, uint8_t *dstY, uint8_t *dstU, uint8_t *dstV, int luma_stride, int width, int32_t *rgb2yuv);



    switch(c->srcFormat) {

#define CASE(pixfmt, prefix) \

    case pixfmt: copy        = bayer_##prefix##_to_yv12_copy; \

                 interpolate = bayer_##prefix##_to_yv12_interpolate; \

                 break;

    CASE(AV_PIX_FMT_BAYER_BGGR8,    bggr8)

    CASE(AV_PIX_FMT_BAYER_BGGR16LE, bggr16le)

    CASE(AV_PIX_FMT_BAYER_BGGR16BE, bggr16be)

    CASE(AV_PIX_FMT_BAYER_RGGB8,    rggb8)

    CASE(AV_PIX_FMT_BAYER_RGGB16LE, rggb16le)

    CASE(AV_PIX_FMT_BAYER_RGGB16BE, rggb16be)

    CASE(AV_PIX_FMT_BAYER_GBRG8,    gbrg8)

    CASE(AV_PIX_FMT_BAYER_GBRG16LE, gbrg16le)

    CASE(AV_PIX_FMT_BAYER_GBRG16BE, gbrg16be)

    CASE(AV_PIX_FMT_BAYER_GRBG8,    grbg8)

    CASE(AV_PIX_FMT_BAYER_GRBG16LE, grbg16le)

    CASE(AV_PIX_FMT_BAYER_GRBG16BE, grbg16be)

#undef CASE

    default: return 0;

    }



    copy(srcPtr, srcStride[0], dstY, dstU, dstV, dstStride[0], c->srcW, c->input_rgb2yuv_table);

    srcPtr += 2 * srcStride[0];

    dstY   += 2 * dstStride[0];

    dstU   +=     dstStride[1];

    dstV   +=     dstStride[1];



    for (i = 2; i < srcSliceH - 2; i += 2) {

        interpolate(srcPtr, srcStride[0], dstY, dstU, dstV, dstStride[0], c->srcW, c->input_rgb2yuv_table);

        srcPtr += 2 * srcStride[0];

        dstY   += 2 * dstStride[0];

        dstU   +=     dstStride[1];

        dstV   +=     dstStride[1];

    }



    copy(srcPtr, srcStride[0], dstY, dstU, dstV, dstStride[0], c->srcW, c->input_rgb2yuv_table);

    return srcSliceH;

}
