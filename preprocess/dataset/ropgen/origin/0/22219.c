static int megasas_init_firmware(MegasasState *s, MegasasCmd *cmd)

{

    uint32_t pa_hi, pa_lo;

    target_phys_addr_t iq_pa, initq_size;

    struct mfi_init_qinfo *initq;

    uint32_t flags;

    int ret = MFI_STAT_OK;



    pa_lo = le32_to_cpu(cmd->frame->init.qinfo_new_addr_lo);

    pa_hi = le32_to_cpu(cmd->frame->init.qinfo_new_addr_hi);

    iq_pa = (((uint64_t) pa_hi << 32) | pa_lo);

    trace_megasas_init_firmware((uint64_t)iq_pa);

    initq_size = sizeof(*initq);

    initq = cpu_physical_memory_map(iq_pa, &initq_size, 0);

    if (!initq || initq_size != sizeof(*initq)) {

        trace_megasas_initq_map_failed(cmd->index);

        s->event_count++;

        ret = MFI_STAT_MEMORY_NOT_AVAILABLE;

        goto out;

    }

    s->reply_queue_len = le32_to_cpu(initq->rq_entries) & 0xFFFF;

    if (s->reply_queue_len > s->fw_cmds) {

        trace_megasas_initq_mismatch(s->reply_queue_len, s->fw_cmds);

        s->event_count++;

        ret = MFI_STAT_INVALID_PARAMETER;

        goto out;

    }

    pa_lo = le32_to_cpu(initq->rq_addr_lo);

    pa_hi = le32_to_cpu(initq->rq_addr_hi);

    s->reply_queue_pa = ((uint64_t) pa_hi << 32) | pa_lo;

    pa_lo = le32_to_cpu(initq->ci_addr_lo);

    pa_hi = le32_to_cpu(initq->ci_addr_hi);

    s->consumer_pa = ((uint64_t) pa_hi << 32) | pa_lo;

    pa_lo = le32_to_cpu(initq->pi_addr_lo);

    pa_hi = le32_to_cpu(initq->pi_addr_hi);

    s->producer_pa = ((uint64_t) pa_hi << 32) | pa_lo;

    s->reply_queue_head = ldl_le_phys(s->producer_pa);

    s->reply_queue_tail = ldl_le_phys(s->consumer_pa);

    flags = le32_to_cpu(initq->flags);

    if (flags & MFI_QUEUE_FLAG_CONTEXT64) {

        s->flags |= MEGASAS_MASK_USE_QUEUE64;

    }

    trace_megasas_init_queue((unsigned long)s->reply_queue_pa,

                             s->reply_queue_len, s->reply_queue_head,

                             s->reply_queue_tail, flags);

    megasas_reset_frames(s);

    s->fw_state = MFI_FWSTATE_OPERATIONAL;

out:

    if (initq) {

        cpu_physical_memory_unmap(initq, initq_size, 0, 0);

    }

    return ret;

}
