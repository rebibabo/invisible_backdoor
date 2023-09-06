static av_cold int on2avc_decode_init(AVCodecContext *avctx)

{

    On2AVCContext *c = avctx->priv_data;

    int i;



    c->avctx = avctx;

    avctx->sample_fmt     = AV_SAMPLE_FMT_FLTP;

    avctx->channel_layout = (avctx->channels == 2) ? AV_CH_LAYOUT_STEREO

                                                   : AV_CH_LAYOUT_MONO;



    c->is_av500 = (avctx->codec_tag == 0x500);

    if (c->is_av500 && avctx->channels == 2) {

        av_log(avctx, AV_LOG_ERROR, "0x500 version should be mono\n");

        return AVERROR_INVALIDDATA;






    if (avctx->channels == 2)

        av_log(avctx, AV_LOG_WARNING,

               "Stereo mode support is not good, patch is welcome\n");



    for (i = 0; i < 20; i++)

        c->scale_tab[i] = ceil(pow(10.0, i * 0.1) * 16) / 32;

    for (; i < 128; i++)

        c->scale_tab[i] = ceil(pow(10.0, i * 0.1) * 0.5);



    if (avctx->sample_rate < 32000 || avctx->channels == 1)

        memcpy(c->long_win, ff_on2avc_window_long_24000,

               1024 * sizeof(*c->long_win));

    else

        memcpy(c->long_win, ff_on2avc_window_long_32000,

               1024 * sizeof(*c->long_win));

    memcpy(c->short_win, ff_on2avc_window_short, 128 * sizeof(*c->short_win));



    c->modes = (avctx->sample_rate <= 40000) ? ff_on2avc_modes_40

                                             : ff_on2avc_modes_44;

    c->wtf   = (avctx->sample_rate <= 40000) ? wtf_40

                                             : wtf_44;



    ff_mdct_init(&c->mdct,       11, 1, 1.0 / (32768.0 * 1024.0));

    ff_mdct_init(&c->mdct_half,  10, 1, 1.0 / (32768.0 * 512.0));

    ff_mdct_init(&c->mdct_small,  8, 1, 1.0 / (32768.0 * 128.0));

    ff_fft_init(&c->fft128,  6, 0);

    ff_fft_init(&c->fft256,  7, 0);

    ff_fft_init(&c->fft512,  8, 1);

    ff_fft_init(&c->fft1024, 9, 1);

    avpriv_float_dsp_init(&c->fdsp, avctx->flags & CODEC_FLAG_BITEXACT);



    if (init_vlc(&c->scale_diff, 9, ON2AVC_SCALE_DIFFS,

                 ff_on2avc_scale_diff_bits,  1, 1,

                 ff_on2avc_scale_diff_codes, 4, 4, 0)) {

        av_log(avctx, AV_LOG_ERROR, "Cannot init VLC\n");

        return AVERROR(ENOMEM);


    for (i = 1; i < 9; i++) {

        int idx = i - 1;

        if (ff_init_vlc_sparse(&c->cb_vlc[i], 9, ff_on2avc_quad_cb_elems[idx],

                               ff_on2avc_quad_cb_bits[idx],  1, 1,

                               ff_on2avc_quad_cb_codes[idx], 4, 4,

                               ff_on2avc_quad_cb_syms[idx],  2, 2, 0)) {

            av_log(avctx, AV_LOG_ERROR, "Cannot init VLC\n");

            on2avc_free_vlcs(c);

            return AVERROR(ENOMEM);



    for (i = 9; i < 16; i++) {

        int idx = i - 9;

        if (ff_init_vlc_sparse(&c->cb_vlc[i], 9, ff_on2avc_pair_cb_elems[idx],

                               ff_on2avc_pair_cb_bits[idx],  1, 1,

                               ff_on2avc_pair_cb_codes[idx], 2, 2,

                               ff_on2avc_pair_cb_syms[idx],  2, 2, 0)) {

            av_log(avctx, AV_LOG_ERROR, "Cannot init VLC\n");

            on2avc_free_vlcs(c);

            return AVERROR(ENOMEM);





    return 0;