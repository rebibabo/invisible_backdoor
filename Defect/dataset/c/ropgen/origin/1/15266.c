static void encode_422_bitstream(HYuvContext *s, int count){

    int i;

    

    count/=2;

    if(s->flags&CODEC_FLAG_PASS1){

        for(i=0; i<count; i++){

            s->stats[0][ s->temp[0][2*i  ] ]++;

            s->stats[1][ s->temp[1][  i  ] ]++;

            s->stats[0][ s->temp[0][2*i+1] ]++;

            s->stats[2][ s->temp[2][  i  ] ]++;

        }

    }else if(s->context){

        for(i=0; i<count; i++){

            s->stats[0][ s->temp[0][2*i  ] ]++;

            put_bits(&s->pb, s->len[0][ s->temp[0][2*i  ] ], s->bits[0][ s->temp[0][2*i  ] ]);

            s->stats[1][ s->temp[1][  i  ] ]++;

            put_bits(&s->pb, s->len[1][ s->temp[1][  i  ] ], s->bits[1][ s->temp[1][  i  ] ]);

            s->stats[0][ s->temp[0][2*i+1] ]++;

            put_bits(&s->pb, s->len[0][ s->temp[0][2*i+1] ], s->bits[0][ s->temp[0][2*i+1] ]);

            s->stats[2][ s->temp[2][  i  ] ]++;

            put_bits(&s->pb, s->len[2][ s->temp[2][  i  ] ], s->bits[2][ s->temp[2][  i  ] ]);

        }

    }else{

        for(i=0; i<count; i++){

            put_bits(&s->pb, s->len[0][ s->temp[0][2*i  ] ], s->bits[0][ s->temp[0][2*i  ] ]);

            put_bits(&s->pb, s->len[1][ s->temp[1][  i  ] ], s->bits[1][ s->temp[1][  i  ] ]);

            put_bits(&s->pb, s->len[0][ s->temp[0][2*i+1] ], s->bits[0][ s->temp[0][2*i+1] ]);

            put_bits(&s->pb, s->len[2][ s->temp[2][  i  ] ], s->bits[2][ s->temp[2][  i  ] ]);

        }

    }

}
