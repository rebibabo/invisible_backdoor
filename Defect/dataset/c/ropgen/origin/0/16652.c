static int filter_frame(AVFilterLink *inlink, AVFilterBufferRef *buf)

{

    AVFilterContext  *ctx = inlink->dst;

    ASyncContext       *s = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    int nb_channels = av_get_channel_layout_nb_channels(buf->audio->channel_layout);

    int64_t pts = (buf->pts == AV_NOPTS_VALUE) ? buf->pts :

                  av_rescale_q(buf->pts, inlink->time_base, outlink->time_base);

    int out_size, ret;

    int64_t delta;



    /* buffer data until we get the first timestamp */

    if (s->pts == AV_NOPTS_VALUE) {

        if (pts != AV_NOPTS_VALUE) {

            s->pts = pts - get_delay(s);

        }

        return write_to_fifo(s, buf);

    }



    /* now wait for the next timestamp */

    if (pts == AV_NOPTS_VALUE) {

        return write_to_fifo(s, buf);

    }



    /* when we have two timestamps, compute how many samples would we have

     * to add/remove to get proper sync between data and timestamps */

    delta    = pts - s->pts - get_delay(s);

    out_size = avresample_available(s->avr);



    if (labs(delta) > s->min_delta) {

        av_log(ctx, AV_LOG_VERBOSE, "Discontinuity - %"PRId64" samples.\n", delta);

        out_size = av_clipl_int32((int64_t)out_size + delta);

    } else {

        if (s->resample) {

            int comp = av_clip(delta, -s->max_comp, s->max_comp);

            av_log(ctx, AV_LOG_VERBOSE, "Compensating %d samples per second.\n", comp);

            avresample_set_compensation(s->avr, delta, inlink->sample_rate);

        }

        delta = 0;

    }



    if (out_size > 0) {

        AVFilterBufferRef *buf_out = ff_get_audio_buffer(outlink, AV_PERM_WRITE,

                                                         out_size);

        if (!buf_out) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        avresample_read(s->avr, buf_out->extended_data, out_size);

        buf_out->pts = s->pts;



        if (delta > 0) {

            av_samples_set_silence(buf_out->extended_data, out_size - delta,

                                   delta, nb_channels, buf->format);

        }

        ret = ff_filter_frame(outlink, buf_out);

        if (ret < 0)

            goto fail;

        s->got_output = 1;

    } else {

        av_log(ctx, AV_LOG_WARNING, "Non-monotonous timestamps, dropping "

               "whole buffer.\n");

    }



    /* drain any remaining buffered data */

    avresample_read(s->avr, NULL, avresample_available(s->avr));



    s->pts = pts - avresample_get_delay(s->avr);

    ret = avresample_convert(s->avr, NULL, 0, 0, buf->extended_data,

                             buf->linesize[0], buf->audio->nb_samples);



fail:

    avfilter_unref_buffer(buf);



    return ret;

}