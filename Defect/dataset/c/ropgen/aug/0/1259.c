av_cold int ff_alsa_open(AVFormatContext *ctx, snd_pcm_stream_t mode,

                         unsigned int *sample_rate,

                         int channels, enum CodecID *codec_id)

{

    AlsaData *s = ctx->priv_data;

    const char *audio_device;

    int res, flags = 0;

    snd_pcm_format_t format;

    snd_pcm_t *h;

    snd_pcm_hw_params_t *hw_params;

    snd_pcm_uframes_t buffer_size, period_size;

    int64_t layout = ctx->streams[0]->codec->channel_layout;



    if (ctx->filename[0] == 0) audio_device = "default";

    else                       audio_device = ctx->filename;



    if (*codec_id == CODEC_ID_NONE)

        *codec_id = DEFAULT_CODEC_ID;

    format = codec_id_to_pcm_format(*codec_id);

    if (format == SND_PCM_FORMAT_UNKNOWN) {

        av_log(ctx, AV_LOG_ERROR, "sample format 0x%04x is not supported\n", *codec_id);

        return AVERROR(ENOSYS);

    }

    s->frame_size = av_get_bits_per_sample(*codec_id) / 8 * channels;



    if (ctx->flags & AVFMT_FLAG_NONBLOCK) {

        flags = SND_PCM_NONBLOCK;

    }

    res = snd_pcm_open(&h, audio_device, mode, flags);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot open audio device %s (%s)\n",

               audio_device, snd_strerror(res));

        return AVERROR(EIO);

    }



    res = snd_pcm_hw_params_malloc(&hw_params);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot allocate hardware parameter structure (%s)\n",

               snd_strerror(res));

        goto fail1;

    }



    res = snd_pcm_hw_params_any(h, hw_params);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot initialize hardware parameter structure (%s)\n",

               snd_strerror(res));

        goto fail;

    }



    res = snd_pcm_hw_params_set_access(h, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set access type (%s)\n",

               snd_strerror(res));

        goto fail;

    }



    res = snd_pcm_hw_params_set_format(h, hw_params, format);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set sample format 0x%04x %d (%s)\n",

               *codec_id, format, snd_strerror(res));

        goto fail;

    }



    res = snd_pcm_hw_params_set_rate_near(h, hw_params, sample_rate, 0);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set sample rate (%s)\n",

               snd_strerror(res));

        goto fail;

    }



    res = snd_pcm_hw_params_set_channels(h, hw_params, channels);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set channel count to %d (%s)\n",

               channels, snd_strerror(res));

        goto fail;

    }



    snd_pcm_hw_params_get_buffer_size_max(hw_params, &buffer_size);

    buffer_size = FFMIN(buffer_size, ALSA_BUFFER_SIZE_MAX);

    /* TODO: maybe use ctx->max_picture_buffer somehow */

    res = snd_pcm_hw_params_set_buffer_size_near(h, hw_params, &buffer_size);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set ALSA buffer size (%s)\n",

               snd_strerror(res));

        goto fail;

    }



    snd_pcm_hw_params_get_period_size_min(hw_params, &period_size, NULL);

    if (!period_size)

        period_size = buffer_size / 4;

    res = snd_pcm_hw_params_set_period_size_near(h, hw_params, &period_size, NULL);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set ALSA period size (%s)\n",

               snd_strerror(res));

        goto fail;

    }

    s->period_size = period_size;



    res = snd_pcm_hw_params(h, hw_params);

    if (res < 0) {

        av_log(ctx, AV_LOG_ERROR, "cannot set parameters (%s)\n",

               snd_strerror(res));

        goto fail;

    }



    snd_pcm_hw_params_free(hw_params);



    if (channels > 2 && layout) {

        if (find_reorder_func(s, *codec_id, layout, mode == SND_PCM_STREAM_PLAYBACK) < 0) {

            char name[128];

            av_get_channel_layout_string(name, sizeof(name), channels, layout);

            av_log(ctx, AV_LOG_WARNING, "ALSA channel layout unknown or unimplemented for %s %s.\n",

                   name, mode == SND_PCM_STREAM_PLAYBACK ? "playback" : "capture");

        }

        if (s->reorder_func) {

            s->reorder_buf_size = buffer_size;

            s->reorder_buf = av_malloc(s->reorder_buf_size * s->frame_size);

            if (!s->reorder_buf)

                goto fail1;

        }

    }



    s->h = h;

    return 0;



fail:

    snd_pcm_hw_params_free(hw_params);

fail1:

    snd_pcm_close(h);

    return AVERROR(EIO);

}
