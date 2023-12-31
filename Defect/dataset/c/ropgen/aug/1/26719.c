void bdrv_close(BlockDriverState *bs)

{

    if (bs->drv) {

        if (bs->backing_hd)

            bdrv_delete(bs->backing_hd);

        bs->drv->bdrv_close(bs);

        qemu_free(bs->opaque);

#ifdef _WIN32

        if (bs->is_temporary) {

            unlink(bs->filename);

        }

#endif

        bs->opaque = NULL;

        bs->drv = NULL;



        /* call the change callback */

        bs->total_sectors = 0;

        bs->media_changed = 1;

        if (bs->change_cb)

            bs->change_cb(bs->change_opaque);

    }

}
