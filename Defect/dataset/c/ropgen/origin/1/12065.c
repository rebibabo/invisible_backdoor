static av_cold int avs_decode_init(AVCodecContext * avctx)

{

    avctx->pix_fmt = PIX_FMT_PAL8;


    return 0;

}