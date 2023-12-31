int bdrv_file_open(BlockDriverState **pbs, const char *filename, int flags)

{

    BlockDriverState *bs;

    int ret;



    bs = bdrv_new("");

    if (!bs)

        return -ENOMEM;

    ret = bdrv_open2(bs, filename, flags | BDRV_O_FILE, NULL);

    if (ret < 0) {

        bdrv_delete(bs);

        return ret;

    }


    *pbs = bs;

    return 0;

}