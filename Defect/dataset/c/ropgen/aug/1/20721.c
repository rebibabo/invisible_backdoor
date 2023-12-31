static int local_link(FsContext *ctx, V9fsPath *oldpath,

                      V9fsPath *dirpath, const char *name)

{

    int ret;

    V9fsString newpath;

    char *buffer, *buffer1;

    int serrno;



    v9fs_string_init(&newpath);

    v9fs_string_sprintf(&newpath, "%s/%s", dirpath->data, name);



    buffer = rpath(ctx, oldpath->data);

    buffer1 = rpath(ctx, newpath.data);

    ret = link(buffer, buffer1);

    g_free(buffer);

    if (ret < 0) {

        goto out;

    }



    /* now link the virtfs_metadata files */

    if (ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        char *vbuffer, *vbuffer1;



        /* Link the .virtfs_metadata files. Create the metada directory */

        ret = local_create_mapped_attr_dir(ctx, newpath.data);

        if (ret < 0) {

            goto err_out;

        }

        vbuffer = local_mapped_attr_path(ctx, oldpath->data);

        vbuffer1 = local_mapped_attr_path(ctx, newpath.data);

        ret = link(vbuffer, vbuffer1);

        g_free(vbuffer);

        g_free(vbuffer1);

        if (ret < 0 && errno != ENOENT) {

            goto err_out;

        }

    }

    goto out;



err_out:

    serrno = errno;

    remove(buffer1);

    errno = serrno;

out:

    g_free(buffer1);

    v9fs_string_free(&newpath);

    return ret;

}
