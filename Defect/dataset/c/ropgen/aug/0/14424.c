int print_insn_xtensa(bfd_vma memaddr, struct disassemble_info *info)

{

    xtensa_isa isa = info->private_data;

    xtensa_insnbuf insnbuf = xtensa_insnbuf_alloc(isa);

    xtensa_insnbuf slotbuf = xtensa_insnbuf_alloc(isa);

    bfd_byte *buffer = g_malloc(1);

    int status = info->read_memory_func(memaddr, buffer, 1, info);

    xtensa_format fmt;

    unsigned slot, slots;

    unsigned len;



    if (status) {

        info->memory_error_func(status, memaddr, info);

        len = -1;

        goto out;

    }

    len = xtensa_isa_length_from_chars(isa, buffer);

    if (len == XTENSA_UNDEFINED) {

        info->fprintf_func(info->stream, ".byte 0x%02x", buffer[0]);

        len = 1;

        goto out;

    }

    buffer = g_realloc(buffer, len);

    status = info->read_memory_func(memaddr + 1, buffer + 1, len - 1, info);

    if (status) {

        info->fprintf_func(info->stream, ".byte 0x%02x", buffer[0]);

        info->memory_error_func(status, memaddr + 1, info);

        len = 1;

        goto out;

    }



    xtensa_insnbuf_from_chars(isa, insnbuf, buffer, len);

    fmt = xtensa_format_decode(isa, insnbuf);

    if (fmt == XTENSA_UNDEFINED) {

        unsigned i;



        for (i = 0; i < len; ++i) {

            info->fprintf_func(info->stream, "%s 0x%02x",

                               i ? ", " : ".byte ", buffer[i]);

        }

        goto out;

    }

    slots = xtensa_format_num_slots(isa, fmt);



    if (slots > 1) {

        info->fprintf_func(info->stream, "{ ");

    }



    for (slot = 0; slot < slots; ++slot) {

        xtensa_opcode opc;

        unsigned opnd, vopnd, opnds;



        if (slot) {

            info->fprintf_func(info->stream, "; ");

        }

        xtensa_format_get_slot(isa, fmt, slot, insnbuf, slotbuf);

        opc = xtensa_opcode_decode(isa, fmt, slot, slotbuf);

        if (opc == XTENSA_UNDEFINED) {

            info->fprintf_func(info->stream, "???");

            continue;

        }

        opnds = xtensa_opcode_num_operands(isa, opc);



        info->fprintf_func(info->stream, "%s", xtensa_opcode_name(isa, opc));



        for (opnd = vopnd = 0; opnd < opnds; ++opnd) {

            if (xtensa_operand_is_visible(isa, opc, opnd)) {

                uint32_t v = 0xbadc0de;

                int rc;



                info->fprintf_func(info->stream, vopnd ? ", " : "\t");

                xtensa_operand_get_field(isa, opc, opnd, fmt, slot,

                                         slotbuf, &v);

                rc = xtensa_operand_decode(isa, opc, opnd, &v);

                if (rc == XTENSA_UNDEFINED) {

                    info->fprintf_func(info->stream, "???");

                } else if (xtensa_operand_is_register(isa, opc, opnd)) {

                    xtensa_regfile rf = xtensa_operand_regfile(isa, opc, opnd);



                    info->fprintf_func(info->stream, "%s%d",

                                       xtensa_regfile_shortname(isa, rf), v);

                } else if (xtensa_operand_is_PCrelative(isa, opc, opnd)) {

                    xtensa_operand_undo_reloc(isa, opc, opnd, &v, memaddr);

                    info->fprintf_func(info->stream, "0x%x", v);

                } else {

                    info->fprintf_func(info->stream, "%d", v);

                }

                ++vopnd;

            }

        }

    }

    if (slots > 1) {

        info->fprintf_func(info->stream, " }");

    }



out:

    g_free(buffer);

    xtensa_insnbuf_free(isa, insnbuf);

    xtensa_insnbuf_free(isa, slotbuf);



    return len;

}
