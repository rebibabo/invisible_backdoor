static int local_mkdir(FsContext *fs_ctx, V9fsPath *dir_path,

                       const char *name, FsCred *credp)

{

    char *path;

    int err = -1;

    int serrno = 0;

    V9fsString fullname;

    char *buffer = NULL;



    v9fs_string_init(&fullname);

    v9fs_string_sprintf(&fullname, "%s/%s", dir_path->data, name);

    path = fullname.data;



    /* Determine the security model */

    if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        buffer = rpath(fs_ctx, path);

        err = mkdir(buffer, SM_LOCAL_DIR_MODE_BITS);

        if (err == -1) {

            goto out;

        }

        credp->fc_mode = credp->fc_mode|S_IFDIR;

        err = local_set_xattr(buffer, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        buffer = rpath(fs_ctx, path);

        err = mkdir(buffer, SM_LOCAL_DIR_MODE_BITS);

        if (err == -1) {

            goto out;

        }

        credp->fc_mode = credp->fc_mode|S_IFDIR;

        err = local_set_mapped_file_attr(fs_ctx, path, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if ((fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

               (fs_ctx->export_flags & V9FS_SM_NONE)) {

        buffer = rpath(fs_ctx, path);

        err = mkdir(buffer, credp->fc_mode);

        if (err == -1) {

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

out:

    g_free(buffer);

    v9fs_string_free(&fullname);

    return err;

}
