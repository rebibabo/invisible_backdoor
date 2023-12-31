void avpriv_set_pts_info(AVStream *s, int pts_wrap_bits,

                         unsigned int pts_num, unsigned int pts_den)

{

    AVRational new_tb;

    if(av_reduce(&new_tb.num, &new_tb.den, pts_num, pts_den, INT_MAX)){

        if(new_tb.num != pts_num)

            av_log(NULL, AV_LOG_DEBUG, "st:%d removing common factor %d from timebase\n", s->index, pts_num/new_tb.num);

    }else

        av_log(NULL, AV_LOG_WARNING, "st:%d has too large timebase, reducing\n", s->index);



    if(new_tb.num <= 0 || new_tb.den <= 0) {

        av_log(NULL, AV_LOG_ERROR, "Ignoring attempt to set invalid timebase for st:%d\n", s->index);

        return;

    }

    s->time_base = new_tb;

    s->pts_wrap_bits = pts_wrap_bits;

}
