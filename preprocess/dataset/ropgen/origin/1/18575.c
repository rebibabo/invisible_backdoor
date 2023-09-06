static void spr_write_tbu(DisasContext *ctx, int sprn, int gprn)

{

    if (ctx->tb->cflags & CF_USE_ICOUNT) {

        gen_io_start();

    }

    gen_helper_store_tbu(cpu_env, cpu_gpr[gprn]);

    if (ctx->tb->cflags & CF_USE_ICOUNT) {

        gen_io_end();

        gen_stop_exception(ctx);

    }

}
