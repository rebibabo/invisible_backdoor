void AUD_del_capture (CaptureVoiceOut *cap, void *cb_opaque)

{

    struct capture_callback *cb;



    for (cb = cap->cb_head.lh_first; cb; cb = cb->entries.le_next) {

        if (cb->opaque == cb_opaque) {

            cb->ops.destroy (cb_opaque);

            LIST_REMOVE (cb, entries);

            qemu_free (cb);



            if (!cap->cb_head.lh_first) {

                SWVoiceOut *sw = cap->hw.sw_head.lh_first, *sw1;



                while (sw) {

                    SWVoiceCap *sc = (SWVoiceCap *) sw;

#ifdef DEBUG_CAPTURE

                    dolog ("freeing %s\n", sw->name);

#endif



                    sw1 = sw->entries.le_next;

                    if (sw->rate) {

                        st_rate_stop (sw->rate);

                        sw->rate = NULL;

                    }

                    LIST_REMOVE (sw, entries);

                    LIST_REMOVE (sc, entries);

                    qemu_free (sc);

                    sw = sw1;

                }

                LIST_REMOVE (cap, entries);

                qemu_free (cap);

            }

            return;

        }

    }

}
