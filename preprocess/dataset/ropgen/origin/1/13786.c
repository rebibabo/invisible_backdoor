void ff_filter_samples(AVFilterLink *link, AVFilterBufferRef *samplesref)

{

    void (*filter_samples)(AVFilterLink *, AVFilterBufferRef *);

    AVFilterPad *dst = link->dstpad;



    FF_DPRINTF_START(NULL, filter_samples); ff_dlog_link(NULL, link, 1);



    if (!(filter_samples = dst->filter_samples))

        filter_samples = ff_default_filter_samples;



    /* prepare to copy the samples if the buffer has insufficient permissions */

    if ((dst->min_perms & samplesref->perms) != dst->min_perms ||

        dst->rej_perms & samplesref->perms) {

        int  i, planar = av_sample_fmt_is_planar(samplesref->format);

        int planes = !planar ? 1:

                     av_get_channel_layout_nb_channels(samplesref->audio->channel_layout);



        av_log(link->dst, AV_LOG_DEBUG,

               "Copying audio data in avfilter (have perms %x, need %x, reject %x)\n",

               samplesref->perms, link->dstpad->min_perms, link->dstpad->rej_perms);



        link->cur_buf = ff_default_get_audio_buffer(link, dst->min_perms,

                                                    samplesref->audio->nb_samples);

        link->cur_buf->pts                = samplesref->pts;

        link->cur_buf->audio->sample_rate = samplesref->audio->sample_rate;



        /* Copy actual data into new samples buffer */

        for (i = 0; i < planes; i++)

            memcpy(link->cur_buf->extended_data[i], samplesref->extended_data[i], samplesref->linesize[0]);



        avfilter_unref_buffer(samplesref);

    } else

        link->cur_buf = samplesref;



    filter_samples(link, link->cur_buf);

}