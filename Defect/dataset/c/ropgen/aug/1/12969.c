static int local_symlink(FsContext *fs_ctx, const char *oldpath,

                         V9fsPath *dir_path, const char *name, FsCred *credp)

{

    int err = -1;

    int serrno = 0;

    char *newpath;

    V9fsString fullname;

    char *buffer;



    v9fs_string_init(&fullname);

    v9fs_string_sprintf(&fullname, "%s/%s", dir_path->data, name);

    newpath = fullname.data;



    /* Determine the security model */

    if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        int fd;

        ssize_t oldpath_size, write_size;

        buffer = rpath(fs_ctx, newpath);

        fd = open(buffer, O_CREAT|O_EXCL|O_RDWR|O_NOFOLLOW, SM_LOCAL_MODE_BITS);

        if (fd == -1) {

            g_free(buffer);

            err = fd;

            goto out;

        }

        /* Write the oldpath (target) to the file. */

        oldpath_size = strlen(oldpath);

        do {

            write_size = write(fd, (void *)oldpath, oldpath_size);

        } while (write_size == -1 && errno == EINTR);



        if (write_size != oldpath_size) {

            serrno = errno;

            close(fd);

            err = -1;

            goto err_end;

        }

        close(fd);

        /* Set cleint credentials in symlink's xattr */

        credp->fc_mode = credp->fc_mode|S_IFLNK;

        err = local_set_xattr(buffer, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        int fd;

        ssize_t oldpath_size, write_size;

        buffer = rpath(fs_ctx, newpath);

        fd = open(buffer, O_CREAT|O_EXCL|O_RDWR|O_NOFOLLOW, SM_LOCAL_MODE_BITS);

        if (fd == -1) {

            g_free(buffer);

            err = fd;

            goto out;

        }

        /* Write the oldpath (target) to the file. */

        oldpath_size = strlen(oldpath);

        do {

            write_size = write(fd, (void *)oldpath, oldpath_size);

        } while (write_size == -1 && errno == EINTR);



        if (write_size != oldpath_size) {

            serrno = errno;

            close(fd);

            err = -1;

            goto err_end;

        }

        close(fd);

        /* Set cleint credentials in symlink's xattr */

        credp->fc_mode = credp->fc_mode|S_IFLNK;

        err = local_set_mapped_file_attr(fs_ctx, newpath, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if ((fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

               (fs_ctx->export_flags & V9FS_SM_NONE)) {

        buffer = rpath(fs_ctx, newpath);

        err = symlink(oldpath, buffer);

        if (err) {

            g_free(buffer);

            goto out;

        }

        err = lchown(buffer, credp->fc_uid, credp->fc_gid);

        if (err == -1) {

            /*

             * If we fail to change ownership and if we are

             * using security model none. Ignore the error

             */

            if ((fs_ctx->export_flags & V9FS_SEC_MASK) != V9FS_SM_NONE) {

                serrno = errno;

                goto err_end;

            } else

                err = 0;

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
