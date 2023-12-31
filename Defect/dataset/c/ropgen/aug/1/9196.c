static int local_remove(FsContext *ctx, const char *path)

{

    int err;

    struct stat stbuf;

    char *buffer;



    if (ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        buffer = rpath(ctx, path);

        err =  lstat(buffer, &stbuf);

        g_free(buffer);

        if (err) {

            goto err_out;

        }

        /*

         * If directory remove .virtfs_metadata contained in the

         * directory

         */

        if (S_ISDIR(stbuf.st_mode)) {

            buffer = g_strdup_printf("%s/%s/%s", ctx->fs_root,

                                     path, VIRTFS_META_DIR);

            err = remove(buffer);

            g_free(buffer);

            if (err < 0 && errno != ENOENT) {

                /*

                 * We didn't had the .virtfs_metadata file. May be file created

                 * in non-mapped mode ?. Ignore ENOENT.

                 */

                goto err_out;

            }

        }

        /*

         * Now remove the name from parent directory

         * .virtfs_metadata directory

         */

        buffer = local_mapped_attr_path(ctx, path);

        err = remove(buffer);

        g_free(buffer);

        if (err < 0 && errno != ENOENT) {

            /*

             * We didn't had the .virtfs_metadata file. May be file created

             * in non-mapped mode ?. Ignore ENOENT.

             */

            goto err_out;

        }

    }



    buffer = rpath(ctx, path);

    err = remove(buffer);

    g_free(buffer);

err_out:

    return err;

}
