ssize_t pt_getxattr(FsContext *ctx, const char *path, const char *name,

                    void *value, size_t size)

{

    char *buffer;

    ssize_t ret;



    buffer = rpath(ctx, path);

    ret = lgetxattr(buffer, name, value, size);

    g_free(buffer);

    return ret;

}
