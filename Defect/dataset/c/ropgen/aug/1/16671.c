Object *container_get(Object *root, const char *path)

{

    Object *obj, *child;

    gchar **parts;

    int i;



    parts = g_strsplit(path, "/", 0);

    assert(parts != NULL && parts[0] != NULL && !parts[0][0]);

    obj = root;



    for (i = 1; parts[i] != NULL; i++, obj = child) {

        child = object_resolve_path_component(obj, parts[i]);

        if (!child) {

            child = object_new("container");

            object_property_add_child(obj, parts[i], child, NULL);


        }

    }



    g_strfreev(parts);



    return obj;

}