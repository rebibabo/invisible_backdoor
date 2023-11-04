static void matroska_execute_seekhead(MatroskaDemuxContext *matroska)

{

    EbmlList *seekhead_list = &matroska->seekhead;

    MatroskaSeekhead *seekhead = seekhead_list->elem;

    int64_t before_pos = avio_tell(matroska->ctx->pb);

    int i;



    // we should not do any seeking in the streaming case

    if (!matroska->ctx->pb->seekable ||

        (matroska->ctx->flags & AVFMT_FLAG_IGNIDX))

        return;



    for (i = 0; i < seekhead_list->nb_elem; i++) {

        if (seekhead[i].pos <= before_pos)

            continue;



        // defer cues parsing until we actually need cue data.

        if (seekhead[i].id == MATROSKA_ID_CUES) {

            matroska->cues_parsing_deferred = 1;

            continue;

        }



        if (matroska_parse_seekhead_entry(matroska, i) < 0)

            break;

    }

}