static int proxy_renameat(FsContext *ctx, V9fsPath *olddir,

                          const char *old_name, V9fsPath *newdir,

                          const char *new_name)

{

    int ret;

    V9fsString old_full_name, new_full_name;



    v9fs_string_init(&old_full_name);

    v9fs_string_init(&new_full_name);



    v9fs_string_sprintf(&old_full_name, "%s/%s", olddir->data, old_name);

    v9fs_string_sprintf(&new_full_name, "%s/%s", newdir->data, new_name);



    ret = proxy_rename(ctx, old_full_name.data, new_full_name.data);

    v9fs_string_free(&old_full_name);

    v9fs_string_free(&new_full_name);

    return ret;

}
