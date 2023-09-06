static int eject_device(Monitor *mon, BlockDriverState *bs, int force)

{

    if (!bdrv_is_removable(bs)) {

        qerror_report(QERR_DEVICE_NOT_REMOVABLE, bdrv_get_device_name(bs));

        return -1;

    }

    if (!force && bdrv_dev_is_medium_locked(bs)) {

        qerror_report(QERR_DEVICE_LOCKED, bdrv_get_device_name(bs));

        return -1;

    }

    bdrv_close(bs);

    return 0;

}
