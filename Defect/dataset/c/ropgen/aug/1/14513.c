static int local_mknod(FsContext *fs_ctx, V9fsPath *dir_path,

                       const char *name, FsCred *credp)

{

    char *path;

    int err = -1;

    int serrno = 0;

    V9fsString fullname;

    char *buffer;



    v9fs_string_init(&fullname);

    v9fs_string_sprintf(&fullname, "%s/%s", dir_path->data, name);

    path = fullname.data;



    /* Determine the security model */

    if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        buffer = rpath(fs_ctx, path);

        err = mknod(buffer, SM_LOCAL_MODE_BITS|S_IFREG, 0);

        if (err == -1) {

            g_free(buffer);

            goto out;

        }

        err = local_set_xattr(buffer, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {



        buffer = rpath(fs_ctx, path);

        err = mknod(buffer, SM_LOCAL_MODE_BITS|S_IFREG, 0);

        if (err == -1) {

            g_free(buffer);

            goto out;

        }

        err = local_set_mapped_file_attr(fs_ctx, path, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if ((fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

               (fs_ctx->export_flags & V9FS_SM_NONE)) {

        buffer = rpath(fs_ctx, path);

        err = mknod(buffer, credp->fc_mode, credp->fc_rdev);

        if (err == -1) {

            g_free(buffer);

            goto out;

        }

        err = local_post_create_passthrough(fs_ctx, path, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    }

    goto out;



err_end:

    remove(buffer);

    errno = serrno;

    g_free(buffer);

out:

    v9fs_string_free(&fullname);

    return err;

}
