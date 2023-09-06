static void mpegts_write_pmt(AVFormatContext *s, MpegTSService *service)

{

    MpegTSWrite *ts = s->priv_data;

    uint8_t data[SECTION_LENGTH], *q, *desc_length_ptr, *program_info_length_ptr;

    int val, stream_type, i;



    q = data;

    put16(&q, 0xe000 | service->pcr_pid);



    program_info_length_ptr = q;

    q += 2; /* patched after */



    /* put program info here */



    val = 0xf000 | (q - program_info_length_ptr - 2);

    program_info_length_ptr[0] = val >> 8;

    program_info_length_ptr[1] = val;



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MpegTSWriteStream *ts_st = st->priv_data;

        AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", NULL, 0);

        switch (st->codec->codec_id) {

        case AV_CODEC_ID_MPEG1VIDEO:

        case AV_CODEC_ID_MPEG2VIDEO:

            stream_type = STREAM_TYPE_VIDEO_MPEG2;

            break;

        case AV_CODEC_ID_MPEG4:

            stream_type = STREAM_TYPE_VIDEO_MPEG4;

            break;

        case AV_CODEC_ID_H264:

            stream_type = STREAM_TYPE_VIDEO_H264;

            break;

        case AV_CODEC_ID_HEVC:

            stream_type = STREAM_TYPE_VIDEO_HEVC;

            break;

        case AV_CODEC_ID_CAVS:

            stream_type = STREAM_TYPE_VIDEO_CAVS;

            break;

        case AV_CODEC_ID_DIRAC:

            stream_type = STREAM_TYPE_VIDEO_DIRAC;

            break;

        case AV_CODEC_ID_MP2:

        case AV_CODEC_ID_MP3:

            stream_type = STREAM_TYPE_AUDIO_MPEG1;

            break;

        case AV_CODEC_ID_AAC:

            stream_type = (ts->flags & MPEGTS_FLAG_AAC_LATM)

                          ? STREAM_TYPE_AUDIO_AAC_LATM

                          : STREAM_TYPE_AUDIO_AAC;

            break;

        case AV_CODEC_ID_AAC_LATM:

            stream_type = STREAM_TYPE_AUDIO_AAC_LATM;

            break;

        case AV_CODEC_ID_AC3:

            stream_type = STREAM_TYPE_AUDIO_AC3;

            break;

        default:

            stream_type = STREAM_TYPE_PRIVATE_DATA;

            break;

        }

        *q++ = stream_type;

        put16(&q, 0xe000 | ts_st->pid);

        desc_length_ptr = q;

        q += 2; /* patched after */



        /* write optional descriptors here */

        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if (lang) {

                char *p;

                char *next = lang->value;

                uint8_t *len_ptr;



                *q++     = 0x0a; /* ISO 639 language descriptor */

                len_ptr  = q++;

                *len_ptr = 0;



                for (p = lang->value; next && *len_ptr < 255 / 4 * 4; p = next + 1) {

                    next = strchr(p, ',');

                    if (strlen(p) != 3 && (!next || next != p + 3))

                        continue; /* not a 3-letter code */



                    *q++ = *p++;

                    *q++ = *p++;

                    *q++ = *p++;



                    if (st->disposition & AV_DISPOSITION_CLEAN_EFFECTS)

                        *q++ = 0x01;

                    else if (st->disposition & AV_DISPOSITION_HEARING_IMPAIRED)

                        *q++ = 0x02;

                    else if (st->disposition & AV_DISPOSITION_VISUAL_IMPAIRED)

                        *q++ = 0x03;

                    else

                        *q++ = 0; /* undefined type */



                    *len_ptr += 4;

                }



                if (*len_ptr == 0)

                    q -= 2; /* no language codes were written */

            }

            break;

        case AVMEDIA_TYPE_SUBTITLE:

        {

            const char *language;

            language = lang && strlen(lang->value) == 3 ? lang->value : "eng";

            *q++ = 0x59;

            *q++ = 8;

            *q++ = language[0];

            *q++ = language[1];

            *q++ = language[2];

            *q++ = 0x10; /* normal subtitles (0x20 = if hearing pb) */

            if (st->codec->extradata_size == 4) {

                memcpy(q, st->codec->extradata, 4);

                q += 4;

            } else {

                put16(&q, 1);     /* page id */

                put16(&q, 1);     /* ancillary page id */

            }

        }

        break;

        case AVMEDIA_TYPE_VIDEO:

            if (stream_type == STREAM_TYPE_VIDEO_DIRAC) {

                *q++ = 0x05; /*MPEG-2 registration descriptor*/

                *q++ = 4;

                *q++ = 'd';

                *q++ = 'r';

                *q++ = 'a';

                *q++ = 'c';

            }

            break;

        }



        val = 0xf000 | (q - desc_length_ptr - 2);

        desc_length_ptr[0] = val >> 8;

        desc_length_ptr[1] = val;

    }

    mpegts_write_section1(&service->pmt, PMT_TID, service->sid, 0, 0, 0,

                          data, q - data);

}
