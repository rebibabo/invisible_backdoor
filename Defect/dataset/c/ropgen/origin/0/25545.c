static int nvdec_vp9_start_frame(AVCodecContext *avctx, const uint8_t *buffer, uint32_t size)

{

    VP9SharedContext *h = avctx->priv_data;

    const AVPixFmtDescriptor *pixdesc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);



    NVDECContext      *ctx = avctx->internal->hwaccel_priv_data;

    CUVIDPICPARAMS     *pp = &ctx->pic_params;

    CUVIDVP9PICPARAMS *ppc = &pp->CodecSpecific.vp9;

    FrameDecodeData *fdd;

    NVDECFrame *cf;

    AVFrame *cur_frame = h->frames[CUR_FRAME].tf.f;



    int ret, i;



    ret = ff_nvdec_start_frame(avctx, cur_frame);

    if (ret < 0)

        return ret;



    fdd = (FrameDecodeData*)cur_frame->private_ref->data;

    cf  = (NVDECFrame*)fdd->hwaccel_priv;



    *pp = (CUVIDPICPARAMS) {

        .PicWidthInMbs     = (cur_frame->width  + 15) / 16,

        .FrameHeightInMbs  = (cur_frame->height + 15) / 16,

        .CurrPicIdx        = cf->idx,



        .CodecSpecific.vp9 = {

            .width                    = cur_frame->width,

            .height                   = cur_frame->height,



            .LastRefIdx               = get_ref_idx(h->refs[h->h.refidx[0]].f),

            .GoldenRefIdx             = get_ref_idx(h->refs[h->h.refidx[1]].f),

            .AltRefIdx                = get_ref_idx(h->refs[h->h.refidx[2]].f),



            .profile                  = h->h.profile,

            .frameContextIdx          = h->h.framectxid,

            .frameType                = !h->h.keyframe,

            .showFrame                = !h->h.invisible,

            .errorResilient           = h->h.errorres,

            .frameParallelDecoding    = h->h.parallelmode,

            .subSamplingX             = pixdesc->log2_chroma_w,

            .subSamplingY             = pixdesc->log2_chroma_h,

            .intraOnly                = h->h.intraonly,

            .allow_high_precision_mv  = h->h.keyframe ? 0 : h->h.highprecisionmvs,

            .refreshEntropyProbs      = h->h.refreshctx,



            .bitDepthMinus8Luma       = pixdesc->comp[0].depth - 8,

            .bitDepthMinus8Chroma     = pixdesc->comp[1].depth - 8,



            .loopFilterLevel          = h->h.filter.level,

            .loopFilterSharpness      = h->h.filter.sharpness,

            .modeRefLfEnabled         = h->h.lf_delta.enabled,



            .log2_tile_columns        = h->h.tiling.log2_tile_cols,

            .log2_tile_rows           = h->h.tiling.log2_tile_rows,



            .segmentEnabled           = h->h.segmentation.enabled,

            .segmentMapUpdate         = h->h.segmentation.update_map,

            .segmentMapTemporalUpdate = h->h.segmentation.temporal,

            .segmentFeatureMode       = h->h.segmentation.absolute_vals,



            .qpYAc                    = h->h.yac_qi,

            .qpYDc                    = h->h.ydc_qdelta,

            .qpChDc                   = h->h.uvdc_qdelta,

            .qpChAc                   = h->h.uvac_qdelta,



            .resetFrameContext        = h->h.resetctx,

            .mcomp_filter_type        = h->h.filtermode ^ (h->h.filtermode <= 1),



            .frameTagSize             = h->h.uncompressed_header_size,

            .offsetToDctParts         = h->h.compressed_header_size,



            .refFrameSignBias[0]      = 0,

        }

    };



    for (i = 0; i < 2; i++)

        ppc->mbModeLfDelta[i] = h->h.lf_delta.mode[i];



    for (i = 0; i < 4; i++)

        ppc->mbRefLfDelta[i] = h->h.lf_delta.ref[i];



    for (i = 0; i < 7; i++)

        ppc->mb_segment_tree_probs[i] = h->h.segmentation.prob[i];



    for (i = 0; i < 3; i++) {

        ppc->activeRefIdx[i] = h->h.refidx[i];

        ppc->segment_pred_probs[i] = h->h.segmentation.pred_prob[i];

        ppc->refFrameSignBias[i + 1] = h->h.signbias[i];

    }



    for (i = 0; i < 8; i++) {

        ppc->segmentFeatureEnable[i][0] = h->h.segmentation.feat[i].q_enabled;

        ppc->segmentFeatureEnable[i][1] = h->h.segmentation.feat[i].lf_enabled;

        ppc->segmentFeatureEnable[i][2] = h->h.segmentation.feat[i].ref_enabled;

        ppc->segmentFeatureEnable[i][3] = h->h.segmentation.feat[i].skip_enabled;



        ppc->segmentFeatureData[i][0] = h->h.segmentation.feat[i].q_val;

        ppc->segmentFeatureData[i][1] = h->h.segmentation.feat[i].lf_val;

        ppc->segmentFeatureData[i][2] = h->h.segmentation.feat[i].ref_val;

        ppc->segmentFeatureData[i][3] = 0;

    }



    switch (avctx->colorspace) {

    default:

    case AVCOL_SPC_UNSPECIFIED:

        ppc->colorSpace = 0;

        break;

    case AVCOL_SPC_BT470BG:

        ppc->colorSpace = 1;

        break;

    case AVCOL_SPC_BT709:

        ppc->colorSpace = 2;

        break;

    case AVCOL_SPC_SMPTE170M:

        ppc->colorSpace = 3;

        break;

    case AVCOL_SPC_SMPTE240M:

        ppc->colorSpace = 4;

        break;

    case AVCOL_SPC_BT2020_NCL:

        ppc->colorSpace = 5;

        break;

    case AVCOL_SPC_RESERVED:

        ppc->colorSpace = 6;

        break;

    case AVCOL_SPC_RGB:

        ppc->colorSpace = 7;

        break;

    }



    return 0;

}
