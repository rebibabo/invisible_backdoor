static int thread_init(AVCodecContext *avctx)

{

    int i;

    ThreadContext *c;

    int thread_count = avctx->thread_count;



    if (!thread_count) {

        int nb_cpus = get_logical_cpus(avctx);

        // use number of cores + 1 as thread count if there is motre than one

        if (nb_cpus > 1)

            thread_count = avctx->thread_count = nb_cpus + 1;

    }



    if (thread_count <= 1) {

        avctx->active_thread_type = 0;

        return 0;

    }



    c = av_mallocz(sizeof(ThreadContext));

    if (!c)

        return -1;



    c->workers = av_mallocz(sizeof(pthread_t)*thread_count);

    if (!c->workers) {

        av_free(c);

        return -1;

    }



    avctx->thread_opaque = c;

    c->current_job = 0;

    c->job_count = 0;

    c->job_size = 0;

    c->done = 0;

    pthread_cond_init(&c->current_job_cond, NULL);

    pthread_cond_init(&c->last_job_cond, NULL);

    pthread_mutex_init(&c->current_job_lock, NULL);

    pthread_mutex_lock(&c->current_job_lock);

    for (i=0; i<thread_count; i++) {

        if(pthread_create(&c->workers[i], NULL, worker, avctx)) {

           avctx->thread_count = i;

           pthread_mutex_unlock(&c->current_job_lock);

           ff_thread_free(avctx);

           return -1;

        }

    }



    avcodec_thread_park_workers(c, thread_count);



    avctx->execute = avcodec_thread_execute;

    avctx->execute2 = avcodec_thread_execute2;

    return 0;

}
