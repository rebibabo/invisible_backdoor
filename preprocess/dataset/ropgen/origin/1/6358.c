static const AVClass *ff_avio_child_class_next(const AVClass *prev)

{

    return prev ? NULL : &ffurl_context_class;

}
