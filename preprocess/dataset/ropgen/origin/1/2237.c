static int mp_pacl_setxattr(FsContext *ctx, const char *path, const char *name,

                            void *value, size_t size, int flags)

{

    char *buffer;

    int ret;



    buffer = rpath(ctx, path);

    ret = lsetxattr(buffer, MAP_ACL_ACCESS, value, size, flags);

    g_free(buffer);

    return ret;

}
