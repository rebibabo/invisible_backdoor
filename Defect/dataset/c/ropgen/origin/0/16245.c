void hmp_sendkey(Monitor *mon, const QDict *qdict)

{

    const char *keys = qdict_get_str(qdict, "keys");

    KeyValueList *keylist, *head = NULL, *tmp = NULL;

    int has_hold_time = qdict_haskey(qdict, "hold-time");

    int hold_time = qdict_get_try_int(qdict, "hold-time", -1);

    Error *err = NULL;

    char *separator;

    int keyname_len;



    while (1) {

        separator = strchr(keys, '-');

        keyname_len = separator ? separator - keys : strlen(keys);



        /* Be compatible with old interface, convert user inputted "<" */

        if (keys[0] == '<' && keyname_len == 1) {

            keys = "less";

            keyname_len = 4;

        }



        keylist = g_malloc0(sizeof(*keylist));

        keylist->value = g_malloc0(sizeof(*keylist->value));



        if (!head) {

            head = keylist;

        }

        if (tmp) {

            tmp->next = keylist;

        }

        tmp = keylist;



        if (strstart(keys, "0x", NULL)) {

            char *endp;

            int value = strtoul(keys, &endp, 0);

            assert(endp <= keys + keyname_len);

            if (endp != keys + keyname_len) {

                goto err_out;

            }

            keylist->value->type = KEY_VALUE_KIND_NUMBER;

            keylist->value->u.number = value;

        } else {

            int idx = index_from_key(keys, keyname_len);

            if (idx == Q_KEY_CODE__MAX) {

                goto err_out;

            }

            keylist->value->type = KEY_VALUE_KIND_QCODE;

            keylist->value->u.qcode = idx;

        }



        if (!separator) {

            break;

        }

        keys = separator + 1;

    }



    qmp_send_key(head, has_hold_time, hold_time, &err);

    hmp_handle_error(mon, &err);



out:

    qapi_free_KeyValueList(head);

    return;



err_out:

    monitor_printf(mon, "invalid parameter: %.*s\n", keyname_len, keys);

    goto out;

}
