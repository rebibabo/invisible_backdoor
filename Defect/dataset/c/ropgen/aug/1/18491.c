static int process_requests(int sock)

{

    int flags;

    int size = 0;

    int retval = 0;

    uint64_t offset;

    ProxyHeader header;

    int mode, uid, gid;

    V9fsString name, value;

    struct timespec spec[2];

    V9fsString oldpath, path;

    struct iovec in_iovec, out_iovec;



    in_iovec.iov_base  = g_malloc(PROXY_MAX_IO_SZ + PROXY_HDR_SZ);

    in_iovec.iov_len   = PROXY_MAX_IO_SZ + PROXY_HDR_SZ;

    out_iovec.iov_base = g_malloc(PROXY_MAX_IO_SZ + PROXY_HDR_SZ);

    out_iovec.iov_len  = PROXY_MAX_IO_SZ + PROXY_HDR_SZ;



    while (1) {

        /*

         * initialize the header type, so that we send

         * response to proper request type.

         */

        header.type = 0;

        retval = read_request(sock, &in_iovec, &header);

        if (retval < 0) {

            goto err_out;

        }



        switch (header.type) {

        case T_OPEN:

            retval = do_open(&in_iovec);

            break;

        case T_CREATE:

            retval = do_create(&in_iovec);

            break;

        case T_MKNOD:

        case T_MKDIR:

        case T_SYMLINK:

            retval = do_create_others(header.type, &in_iovec);

            break;

        case T_LINK:

            v9fs_string_init(&path);

            v9fs_string_init(&oldpath);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ,

                                     "ss", &oldpath, &path);

            if (retval > 0) {

                retval = link(oldpath.data, path.data);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&oldpath);

            v9fs_string_free(&path);

            break;

        case T_LSTAT:

        case T_STATFS:

            retval = do_stat(header.type, &in_iovec, &out_iovec);

            break;

        case T_READLINK:

            retval = do_readlink(&in_iovec, &out_iovec);

            break;

        case T_CHMOD:

            v9fs_string_init(&path);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ,

                                     "sd", &path, &mode);

            if (retval > 0) {

                retval = chmod(path.data, mode);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            break;

        case T_CHOWN:

            v9fs_string_init(&path);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ, "sdd", &path,

                                     &uid, &gid);

            if (retval > 0) {

                retval = lchown(path.data, uid, gid);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            break;

        case T_TRUNCATE:

            v9fs_string_init(&path);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ, "sq",

                                     &path, &offset);

            if (retval > 0) {

                retval = truncate(path.data, offset);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            break;

        case T_UTIME:

            v9fs_string_init(&path);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ, "sqqqq", &path,

                                     &spec[0].tv_sec, &spec[0].tv_nsec,

                                     &spec[1].tv_sec, &spec[1].tv_nsec);

            if (retval > 0) {

                retval = qemu_utimens(path.data, spec);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            break;

        case T_RENAME:

            v9fs_string_init(&path);

            v9fs_string_init(&oldpath);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ,

                                     "ss", &oldpath, &path);

            if (retval > 0) {

                retval = rename(oldpath.data, path.data);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&oldpath);

            v9fs_string_free(&path);

            break;

        case T_REMOVE:

            v9fs_string_init(&path);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ, "s", &path);

            if (retval > 0) {

                retval = remove(path.data);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            break;

        case T_LGETXATTR:

        case T_LLISTXATTR:

            retval = do_getxattr(header.type, &in_iovec, &out_iovec);

            break;

        case T_LSETXATTR:

            v9fs_string_init(&path);

            v9fs_string_init(&name);

            v9fs_string_init(&value);

            retval = proxy_unmarshal(&in_iovec, PROXY_HDR_SZ, "sssdd", &path,

                                     &name, &value, &size, &flags);

            if (retval > 0) {

                retval = lsetxattr(path.data,

                                   name.data, value.data, size, flags);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            v9fs_string_free(&name);

            v9fs_string_free(&value);

            break;

        case T_LREMOVEXATTR:

            v9fs_string_init(&path);

            v9fs_string_init(&name);

            retval = proxy_unmarshal(&in_iovec,

                                     PROXY_HDR_SZ, "ss", &path, &name);

            if (retval > 0) {

                retval = lremovexattr(path.data, name.data);

                if (retval < 0) {

                    retval = -errno;

                }

            }

            v9fs_string_free(&path);

            v9fs_string_free(&name);

            break;

        case T_GETVERSION:

            retval = do_getversion(&in_iovec, &out_iovec);

            break;

        default:

            goto err_out;

            break;

        }



        if (process_reply(sock, header.type, &out_iovec, retval) < 0) {

            goto err_out;

        }

    }

err_out:

    g_free(in_iovec.iov_base);

    g_free(out_iovec.iov_base);

    return -1;

}
