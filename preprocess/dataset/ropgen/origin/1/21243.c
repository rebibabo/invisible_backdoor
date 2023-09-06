static void virtio_gpu_set_scanout(VirtIOGPU *g,

                                   struct virtio_gpu_ctrl_command *cmd)

{

    struct virtio_gpu_simple_resource *res;

    struct virtio_gpu_scanout *scanout;

    pixman_format_code_t format;

    uint32_t offset;

    int bpp;

    struct virtio_gpu_set_scanout ss;



    VIRTIO_GPU_FILL_CMD(ss);

    trace_virtio_gpu_cmd_set_scanout(ss.scanout_id, ss.resource_id,

                                     ss.r.width, ss.r.height, ss.r.x, ss.r.y);



    if (ss.scanout_id >= g->conf.max_outputs) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: illegal scanout id specified %d",

                      __func__, ss.scanout_id);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID;

        return;

    }



    g->enable = 1;

    if (ss.resource_id == 0) {

        scanout = &g->scanout[ss.scanout_id];

        if (scanout->resource_id) {

            res = virtio_gpu_find_resource(g, scanout->resource_id);

            if (res) {

                res->scanout_bitmask &= ~(1 << ss.scanout_id);

            }

        }

        if (ss.scanout_id == 0) {

            qemu_log_mask(LOG_GUEST_ERROR,

                          "%s: illegal scanout id specified %d",

                          __func__, ss.scanout_id);

            cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID;

            return;

        }

        dpy_gfx_replace_surface(g->scanout[ss.scanout_id].con, NULL);

        scanout->ds = NULL;

        scanout->width = 0;

        scanout->height = 0;

        return;

    }



    /* create a surface for this scanout */

    res = virtio_gpu_find_resource(g, ss.resource_id);

    if (!res) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: illegal resource specified %d\n",

                      __func__, ss.resource_id);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;

        return;

    }



    if (ss.r.x > res->width ||

        ss.r.y > res->height ||

        ss.r.width > res->width ||

        ss.r.height > res->height ||

        ss.r.x + ss.r.width > res->width ||

        ss.r.y + ss.r.height > res->height) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: illegal scanout %d bounds for"

                      " resource %d, (%d,%d)+%d,%d vs %d %d\n",

                      __func__, ss.scanout_id, ss.resource_id, ss.r.x, ss.r.y,

                      ss.r.width, ss.r.height, res->width, res->height);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;

        return;

    }



    scanout = &g->scanout[ss.scanout_id];



    format = pixman_image_get_format(res->image);

    bpp = (PIXMAN_FORMAT_BPP(format) + 7) / 8;

    offset = (ss.r.x * bpp) + ss.r.y * pixman_image_get_stride(res->image);

    if (!scanout->ds || surface_data(scanout->ds)

        != ((uint8_t *)pixman_image_get_data(res->image) + offset) ||

        scanout->width != ss.r.width ||

        scanout->height != ss.r.height) {

        pixman_image_t *rect;

        void *ptr = (uint8_t *)pixman_image_get_data(res->image) + offset;

        rect = pixman_image_create_bits(format, ss.r.width, ss.r.height, ptr,

                                        pixman_image_get_stride(res->image));

        pixman_image_ref(res->image);

        pixman_image_set_destroy_function(rect, virtio_unref_resource,

                                          res->image);

        /* realloc the surface ptr */

        scanout->ds = qemu_create_displaysurface_pixman(rect);

        if (!scanout->ds) {

            cmd->error = VIRTIO_GPU_RESP_ERR_UNSPEC;

            return;

        }


        dpy_gfx_replace_surface(g->scanout[ss.scanout_id].con, scanout->ds);

    }



    res->scanout_bitmask |= (1 << ss.scanout_id);

    scanout->resource_id = ss.resource_id;

    scanout->x = ss.r.x;

    scanout->y = ss.r.y;

    scanout->width = ss.r.width;

    scanout->height = ss.r.height;

}