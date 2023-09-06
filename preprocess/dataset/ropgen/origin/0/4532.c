static int vorbis_parse_setup_hdr_floors(vorbis_context *vc) {

    GetBitContext *gb=&vc->gb;

    uint_fast16_t i,j,k;



    vc->floor_count=get_bits(gb, 6)+1;



    vc->floors=av_mallocz(vc->floor_count * sizeof(vorbis_floor));



    for (i=0;i<vc->floor_count;++i) {

        vorbis_floor *floor_setup=&vc->floors[i];



        floor_setup->floor_type=get_bits(gb, 16);



        AV_DEBUG(" %d. floor type %d \n", i, floor_setup->floor_type);



        if (floor_setup->floor_type==1) {

            uint_fast8_t maximum_class=0;

            uint_fast8_t rangebits;

            uint_fast16_t floor1_values=2;



            floor_setup->decode=vorbis_floor1_decode;



            floor_setup->data.t1.partitions=get_bits(gb, 5);



            AV_DEBUG(" %d.floor: %d partitions \n", i, floor_setup->data.t1.partitions);



            for(j=0;j<floor_setup->data.t1.partitions;++j) {

                floor_setup->data.t1.partition_class[j]=get_bits(gb, 4);

                if (floor_setup->data.t1.partition_class[j]>maximum_class) maximum_class=floor_setup->data.t1.partition_class[j];



                AV_DEBUG(" %d. floor %d partition class %d \n", i, j, floor_setup->data.t1.partition_class[j]);



            }



            AV_DEBUG(" maximum class %d \n", maximum_class);



            floor_setup->data.t1.maximum_class=maximum_class;



            for(j=0;j<=maximum_class;++j) {

                floor_setup->data.t1.class_dimensions[j]=get_bits(gb, 3)+1;

                floor_setup->data.t1.class_subclasses[j]=get_bits(gb, 2);



                AV_DEBUG(" %d floor %d class dim: %d subclasses %d \n", i, j, floor_setup->data.t1.class_dimensions[j], floor_setup->data.t1.class_subclasses[j]);



                if (floor_setup->data.t1.class_subclasses[j]) {

                    floor_setup->data.t1.class_masterbook[j]=get_bits(gb, 8);



                    AV_DEBUG("   masterbook: %d \n", floor_setup->data.t1.class_masterbook[j]);

                }



                for(k=0;k<(1<<floor_setup->data.t1.class_subclasses[j]);++k) {

                    floor_setup->data.t1.subclass_books[j][k]=(int16_t)get_bits(gb, 8)-1;



                    AV_DEBUG("    book %d. : %d \n", k, floor_setup->data.t1.subclass_books[j][k]);

                }

            }



            floor_setup->data.t1.multiplier=get_bits(gb, 2)+1;

            floor_setup->data.t1.x_list_dim=2;



            for(j=0;j<floor_setup->data.t1.partitions;++j) {

                floor_setup->data.t1.x_list_dim+=floor_setup->data.t1.class_dimensions[floor_setup->data.t1.partition_class[j]];

            }



            floor_setup->data.t1.list=av_mallocz(floor_setup->data.t1.x_list_dim * sizeof(vorbis_floor1_entry));





            rangebits=get_bits(gb, 4);

            floor_setup->data.t1.list[0].x = 0;

            floor_setup->data.t1.list[1].x = (1<<rangebits);



            for(j=0;j<floor_setup->data.t1.partitions;++j) {

                for(k=0;k<floor_setup->data.t1.class_dimensions[floor_setup->data.t1.partition_class[j]];++k,++floor1_values) {

                    floor_setup->data.t1.list[floor1_values].x=get_bits(gb, rangebits);



                    AV_DEBUG(" %d. floor1 Y coord. %d \n", floor1_values, floor_setup->data.t1.list[floor1_values].x);

                }

            }



// Precalculate order of x coordinates - needed for decode

            ff_vorbis_ready_floor1_list(floor_setup->data.t1.list, floor_setup->data.t1.x_list_dim);

        }

        else if(floor_setup->floor_type==0) {

            uint_fast8_t max_codebook_dim=0;



            floor_setup->decode=vorbis_floor0_decode;



            floor_setup->data.t0.order=get_bits(gb, 8);

            floor_setup->data.t0.rate=get_bits(gb, 16);

            floor_setup->data.t0.bark_map_size=get_bits(gb, 16);

            floor_setup->data.t0.amplitude_bits=get_bits(gb, 6);

            /* zero would result in a div by zero later *

             * 2^0 - 1 == 0                             */

            if (floor_setup->data.t0.amplitude_bits == 0) {

              av_log(vc->avccontext, AV_LOG_ERROR,

                     "Floor 0 amplitude bits is 0.\n");

              return 1;

            }

            floor_setup->data.t0.amplitude_offset=get_bits(gb, 8);

            floor_setup->data.t0.num_books=get_bits(gb, 4)+1;



            /* allocate mem for booklist */

            floor_setup->data.t0.book_list=

                av_malloc(floor_setup->data.t0.num_books);

            if(!floor_setup->data.t0.book_list) { return 1; }

            /* read book indexes */

            {

                int idx;

                uint_fast8_t book_idx;

                for (idx=0;idx<floor_setup->data.t0.num_books;++idx) {

                    book_idx=get_bits(gb, 8);

                    if (book_idx>=vc->codebook_count)

                        return 1;

                    floor_setup->data.t0.book_list[idx]=book_idx;

                    if (vc->codebooks[book_idx].dimensions > max_codebook_dim)

                        max_codebook_dim=vc->codebooks[book_idx].dimensions;

                }

            }



            create_map( vc, i );



            /* allocate mem for lsp coefficients */

            {

                /* codebook dim is for padding if codebook dim doesn't *

                 * divide order+1 then we need to read more data       */

                floor_setup->data.t0.lsp=

                    av_malloc((floor_setup->data.t0.order+1 + max_codebook_dim)

                              * sizeof(float));

                if(!floor_setup->data.t0.lsp) { return 1; }

            }



#ifdef V_DEBUG /* debug output parsed headers */

            AV_DEBUG("floor0 order: %u\n", floor_setup->data.t0.order);

            AV_DEBUG("floor0 rate: %u\n", floor_setup->data.t0.rate);

            AV_DEBUG("floor0 bark map size: %u\n",

              floor_setup->data.t0.bark_map_size);

            AV_DEBUG("floor0 amplitude bits: %u\n",

              floor_setup->data.t0.amplitude_bits);

            AV_DEBUG("floor0 amplitude offset: %u\n",

              floor_setup->data.t0.amplitude_offset);

            AV_DEBUG("floor0 number of books: %u\n",

              floor_setup->data.t0.num_books);

            AV_DEBUG("floor0 book list pointer: %p\n",

              floor_setup->data.t0.book_list);

            {

              int idx;

              for (idx=0;idx<floor_setup->data.t0.num_books;++idx) {

                AV_DEBUG( "  Book %d: %u\n",

                  idx+1,

                  floor_setup->data.t0.book_list[idx] );

              }

            }

#endif

        }

        else {

            av_log(vc->avccontext, AV_LOG_ERROR, "Invalid floor type!\n");

            return 1;

        }

    }

    return 0;

}